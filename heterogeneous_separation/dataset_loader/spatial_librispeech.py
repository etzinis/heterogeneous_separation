# © MERL 2022
# Created by Efthymios Tzinis
"""
Conditional separation dataset loader using the Librispeech dataset with extra spatial
conditioning.
"""

import heterogeneous_separation.dataset_loader.abstract_dataset as abstract_dataset
from __config__ import LIBRISPEECH_DATA_P, LIBRISPEECH_METADATA_P

import warnings
import torch
import os
import numpy as np
import pickle
import glob2
import sys
from tqdm import tqdm
from time import time
from scipy.io import wavfile
import csv
import torchaudio
import pyroomacoustics as pra
import itertools

GENDERS = ['f', 'm']
NUM_GENDERS = len(GENDERS)
GENDER_STR_TO_INT_DIC = dict([(x, i) for i, x in enumerate(GENDERS)])
LOUDNESS_LEVELS = ['louder', 'whispering']
NUM_LOUDNESS_LEVELS = len(LOUDNESS_LEVELS)
LOUDNESS_LEVELS_STR_TO_INT_DIC = dict([(x, i) for i, x in enumerate(LOUDNESS_LEVELS)])
SPATIAL_CONDITIONS = ["near_field", "far_field"]
NUM_SPATIAL_CONDITIONS = len(SPATIAL_CONDITIONS)
SPATIAL_CONDITIONS_STR_TO_INT_DIC = dict(
    [(x, i) for i, x in enumerate(SPATIAL_CONDITIONS)])
VALID_CONDITIONS = GENDERS + LOUDNESS_LEVELS + SPATIAL_CONDITIONS
VALID_QUERIES = ["in_mix_same_gender", "in_mix_cross_gender", "out_mix_gender",
                 "near_field", "far_field", "louder", "whispering"]
# in_mix_same_gender,in_mix_cross_gender,out_mix_gender,near_field,far_field,louder,whispering
# 0.4,0.1,0.0,0.25,0.25,0.0,0.0
CONDITIONAL_EMB_DIM = NUM_GENDERS + NUM_SPATIAL_CONDITIONS + NUM_LOUDNESS_LEVELS


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for multiple conditions dependent source separation."""
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['val', 'train', 'test'])

        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[lambda x: x >= 0])

        # Create the possible permutations for the spatial conditions
        self.valid_spatial_perms = list(itertools.combinations_with_replacement(
            [True, False], 2))
        self.valid_spatial_perms += [x[::-1] for x in self.valid_spatial_perms]
        self.valid_spatial_perms = [list(x) for x in self.valid_spatial_perms]

        self.valid_queries = self.get_arg_and_check_validness(
            'valid_queries', known_type=list,
            extra_lambda_checks=[lambda y: all([(x in VALID_QUERIES) for x in y])])

        self.queries_priors = self.get_arg_and_check_validness(
            'queries_priors', known_type=list,
            extra_lambda_checks=[lambda y: (sum(y) - 1.) < 1e-5,
                                 lambda y: len(y) == len(self.valid_queries)])
        self.queries_cdf = np.cumsum(self.queries_priors).tolist()

        self.sample_rate = self.get_arg_and_check_validness('sample_rate',
                                                            known_type=int)

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.overlap_ratio = self.get_arg_and_check_validness(
            'overlap_ratio', known_type=float,
            extra_lambda_checks=[lambda x: 1 >= x >= 0])

        self.input_max_snr = self.get_arg_and_check_validness(
            'input_max_snr', known_type=float, extra_lambda_checks=[lambda x: x >= 0])

        self.time_samples = int(self.sample_rate * self.timelength)

        # Initial Spatial parameters regions
        self.room_l_region = [9., 11.]
        self.room_w_region = [9., 11.]
        self.room_h_region = [2.6, 3.5]
        self.rt60_region = [0.3, 0.6]

        # What if we do near < 1m and far field > 1m
        self.near_field_distance_region = [0.2, 0.6]
        self.far_field_distance_region = [1.7, 3.]
        self.source_h_region = [1.5, 2.0]

        # One dataset more specialist
        # self.room_l_region = [10., 10.2]
        # self.room_w_region = [10., 10.2]
        # self.room_h_region = [2.99, 3.05]
        # self.rt60_region = [0.55, 0.6]
        #
        # self.near_field_distance_region = [0.45, 0.5]
        # self.far_field_distance_region = [2.3, 2.5]
        # self.source_h_region = [1.75, 1.85]

        # # Total mismatch
        # self.room_l_region = [11., 11.2]
        # self.room_w_region = [11., 11.2]
        # self.room_h_region = [3.25, 3.55]
        # self.rt60_region = [0.55, 0.6]
        #
        # self.near_field_distance_region = [0.45, 0.5]
        # self.far_field_distance_region = [2.75, 3.0]
        # self.source_h_region = [1.75, 1.85]

        self.metadata_dic = self.get_metadata_dic()
        self.metadata_dic = self.get_only_metadata_for_split()
        self.valid_speakers = list(self.metadata_dic.keys())

    def __len__(self):
        return self.n_samples

    def get_only_metadata_for_split(self):
        processed_metadata = {}
        for speaker_id, speaker_data in self.metadata_dic.items():
            if speaker_data["split"] == self.split:
                processed_metadata[speaker_id] = self.metadata_dic[speaker_id]
        return processed_metadata

    def get_metadata_dic(self,):
        metadata_info_path = self.get_metadata_path()
        if os.path.lexists(metadata_info_path):
            with open(metadata_info_path, 'rb') as filehandle:
                return pickle.load(filehandle)
        else:
            raise IOError(f"File: {metadata_info_path} could not be parsed for metadata!")

    @staticmethod
    def get_metadata_path():
        metadata_path = LIBRISPEECH_METADATA_P
        if os.path.lexists(metadata_path):
            return metadata_path
        else:
            raise IOError('Metadata file at: {} not found!'.format(metadata_path))

    def flacread(self, path):
        waveform, fs = torchaudio.load(path)
        # Resample in case of a given sample rate
        if self.sample_rate < fs:
            waveform = torchaudio.functional.resample(
                waveform, fs, self.sample_rate, resampling_method="kaiser_window")
        elif self.sample_rate > fs:
            raise ValueError("Cannot upsample.")

        # Convert to single-channel
        if len(waveform.shape) > 1:
            waveform = waveform.sum(0)

        return (1. * waveform - waveform.mean()) / (waveform.std() + 1e-8)

    def simulate_a_source_in_a_room(self, waveform, room_params, is_near_field):
        room_dims = [room_params["length"], room_params["width"], room_params["height"]]
        e_absorption, max_order = pra.inverse_sabine(room_params["rt60"], room_dims)
        room = pra.ShoeBox(
            room_dims, fs=int(self.sample_rate), materials=pra.Material(e_absorption),
            max_order=max_order
        )

        # Put the microphone in the middle of the room
        mic_loc_x = room_params["length"] / 2.
        mic_loc_y = room_params["width"] / 2.
        mic_loc_z = room_params["height"] / 2.
        mic_locs = np.c_[[mic_loc_x, mic_loc_y, mic_loc_z],]
        room.add_microphone_array(mic_locs)

        # Add the near field source
        if is_near_field:
            theta = np.random.uniform(0, np.pi)
            dist = np.random.uniform(0.3, 0.5)
        else:
            theta = np.random.uniform(0, np.pi)
            dist = np.random.uniform(1.5, 2.5)

        source_x_loc = np.cos(theta) * dist + mic_loc_x
        source_y_loc = np.sin(theta) * dist + mic_loc_y
        # A random speaker height sampling
        source_z_loc = np.random.uniform(1.6, 1.9)

        room.add_source([source_x_loc, source_y_loc, source_z_loc],
                        signal=waveform, delay=0.0)
        room.simulate()

        return room.mic_array.signals[-1, :]

    def get_sources_tensors(self, wav_paths, room_params, input_min_snr=0.,
                            sources_are_near_field=[True, False]):

        if len(wav_paths) != 2:
            raise ValueError("This dataset loader works with only 2 sources.")

        waveform_1 = self.flacread(wav_paths[0])
        waveform_2 = self.flacread(wav_paths[1])

        waveform_1 = self.simulate_a_source_in_a_room(
            waveform_1.numpy(), room_params, is_near_field=sources_are_near_field[0])

        waveform_2 = self.simulate_a_source_in_a_room(
            waveform_2.numpy(), room_params, is_near_field=sources_are_near_field[1])

        # Mix with the specified overlap ratio
        wav1_tensor = self.get_padded_tensor(waveform_1)
        wav2_tensor = self.get_padded_tensor(waveform_2)

        # Sample a random overlap ratio between [self.overlap_ratio, 1.]
        sampled_olp_ratio = np.random.uniform(low=self.overlap_ratio, high=1.)
        non_olp_samples = int((1. - sampled_olp_ratio) * wav1_tensor.shape[0])

        delayed_wav2_tensor = torch.zeros_like(wav2_tensor)
        delayed_wav2_tensor[non_olp_samples:] = \
            wav2_tensor[:wav2_tensor.shape[0] - non_olp_samples]

        # Mix the two tensors with a specified SNR ratio
        snr_ratio = np.random.uniform(input_min_snr, self.input_max_snr)
        chosen_sign = np.random.choice([-1., 1.])
        snr_ratio = chosen_sign * snr_ratio

        source_1_tensor, source_2_tensor = self.mix_2_with_specified_snr(
            wav1_tensor, delayed_wav2_tensor, snr_ratio)

        sources_tensor = torch.stack([source_1_tensor, source_2_tensor], axis=0)

        return sources_tensor, snr_ratio

    def sample_conditional_query(self):
        random_draw = np.random.random()
        selected_query = self.valid_queries[-1]
        for query_cdf_val, query_str in zip(self.queries_cdf, self.valid_queries):
            if random_draw <= query_cdf_val:
                selected_query = query_str
                break
        return selected_query

    @staticmethod
    def sample_conditional_target(sources_tensor, selected_query,
                                  snr_ratio, speakers_genders):

        specific_query_id = None
        if selected_query == 'in_mix_cross_gender':
            selected_gender_ind = np.random.randint(0, len(speakers_genders))
            selected_gender = speakers_genders[selected_gender_ind]
            specific_query_id = selected_gender
            target_tensor = torch.stack([sources_tensor[selected_gender_ind],
                                         sources_tensor[1 - selected_gender_ind]], axis=0)

        elif selected_query == 'in_mix_same_gender':
            selected_gender_ind = np.random.randint(0, len(speakers_genders))
            selected_gender = speakers_genders[selected_gender_ind]
            specific_query_id = selected_gender
            target_tensor = torch.zeros_like(sources_tensor)
            target_tensor[0] = torch.sum(sources_tensor, axis=0)

        elif selected_query == 'out_mix_gender':
            out_of_mix_genders = [gen for gen in GENDERS if gen not in speakers_genders]
            selected_gender = np.random.choice(out_of_mix_genders)
            specific_query_id = selected_gender
            target_tensor = torch.zeros_like(sources_tensor)
            target_tensor[1] = torch.sum(sources_tensor, axis=0)

        elif selected_query == 'near_field':
            specific_query_id = selected_query
            target_tensor = sources_tensor

        elif selected_query == 'far_field':
            specific_query_id = selected_query
            target_tensor = torch.stack([sources_tensor[1], sources_tensor[0]], axis=0)

        elif selected_query == 'louder':
            specific_query_id = selected_query
            if snr_ratio >= 0:
                target_tensor = sources_tensor
            else:
                target_tensor = torch.stack([sources_tensor[1], sources_tensor[0]],
                                            axis=0)

        elif selected_query == 'whispering':
            specific_query_id = selected_query
            if snr_ratio >= 0:
                target_tensor = torch.stack(
                    [sources_tensor[1], sources_tensor[0]], axis=0)
            else:
                target_tensor = sources_tensor

        else:
            raise ValueError(f"Invalid query {selected_query}")

        return target_tensor, selected_query, specific_query_id

    def sample_room_parameters(self):
        room_params = {
            "length": np.random.uniform(self.room_l_region[0], self.room_l_region[1]),
            "width": np.random.uniform(self.room_w_region[0], self.room_w_region[1]),
            "height": np.random.uniform(self.room_h_region[0], self.room_h_region[1]),
            "rt60": np.random.uniform(self.rt60_region[0], self.rt60_region[1]),
        }
        return room_params

    @staticmethod
    def get_one_hot(q_condition_str, specific_query_id):
        """Converts to one hot encoding"""
        if "gender" in q_condition_str:
            selected_gender_str = specific_query_id.split("_")[-1]
            encoded_id = GENDER_STR_TO_INT_DIC[selected_gender_str]
        elif "field" in q_condition_str:
            offset = NUM_GENDERS
            encoded_id = SPATIAL_CONDITIONS_STR_TO_INT_DIC[q_condition_str] + offset
        else:
            offset = NUM_GENDERS + NUM_SPATIAL_CONDITIONS
            encoded_id = LOUDNESS_LEVELS_STR_TO_INT_DIC[q_condition_str] + offset

        return torch.nn.functional.one_hot(
            torch.tensor([encoded_id]).to(torch.long),
            num_classes=CONDITIONAL_EMB_DIM).to(torch.float32)[0]

    def __getitem__(self, idx):
        if self.augment:
            seed = int(np.modf(time())[0] * 100000000)
        else:
            seed = idx
        np.random.seed(seed)

        # Sample the conditional query
        sampled_query = self.sample_conditional_query()

        # In the case where we get the out_mix query condition, we need to
        # appropriately resample the second speaker.
        sampled_speaker_id_1 = np.random.choice(self.valid_speakers)
        gender_1 = self.metadata_dic[sampled_speaker_id_1]["gender"]
        if sampled_query == "out_mix_gender" or sampled_query == "in_mix_same_gender":
            sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            while (sampled_speaker_id_2 == sampled_speaker_id_1 or
                   self.metadata_dic[sampled_speaker_id_2]["gender"] != gender_1):
                sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            sampled_speaker_ids = [sampled_speaker_id_1, sampled_speaker_id_2]
        elif sampled_query == "in_mix_cross_gender":
            sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            while (sampled_speaker_id_2 == sampled_speaker_id_1 or
                   self.metadata_dic[sampled_speaker_id_2]["gender"] == gender_1):
                sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            sampled_speaker_ids = [sampled_speaker_id_1, sampled_speaker_id_2]
        else:
            # Sample two speakers
            sampled_speaker_ids = np.random.choice(self.valid_speakers, 2, replace=False)
        sampled_speaker_genders = [self.metadata_dic[s]["gender"]
                                   for s in sampled_speaker_ids]

        # Sample the files
        wav_paths = []
        for sampled_speeaker_id in sampled_speaker_ids:
            this_wav_path = np.random.choice(
                self.metadata_dic[sampled_speeaker_id]['flac_paths'], 1)[0]
            wav_paths.append(this_wav_path)

        # Sample room parameters
        room_params = self.sample_room_parameters()

        if sampled_query in ['whispering', 'louder']:
            input_min_snr = 1.
            # For the cases where we want to condition on loudness we want to
            # have sources with distinct gains.
        else:
            input_min_snr = 0.

        # To also make a general version for sampling the near/far field
        if sampled_query in ['near_field', 'far_field']:
            sources_are_near_field = [True, False]
        else:
            sampled_spatial_perm_ind = np.random.randint(
                low=0, high=len(self.valid_spatial_perms))
            sources_are_near_field = self.valid_spatial_perms[
                sampled_spatial_perm_ind]

        sources_tensor, snr_ratio = self.get_sources_tensors(
            wav_paths, room_params, input_min_snr=input_min_snr,
            sources_are_near_field=sources_are_near_field)

        # Assume that always the first speaker is the near-field and the other is
        # far-field
        (target_tensor, q_condition_str, specific_query_id) = \
            self.sample_conditional_target(
                sources_tensor, sampled_query, snr_ratio, sampled_speaker_genders)

        q_condition_one_hot = self.get_one_hot(q_condition_str, specific_query_id)

        return sources_tensor, target_tensor, q_condition_one_hot, specific_query_id

    def get_multidataset_item(self, idx):
        if self.augment:
            seed = int(np.modf(time())[0] * 100000000)
        else:
            seed = idx
        np.random.seed(seed)

        # Sample the conditional query
        sampled_query = self.sample_conditional_query()

        # In the case where we get the out_mix query condition, we need to
        # appropriately resample the second speaker.
        sampled_speaker_id_1 = np.random.choice(self.valid_speakers)
        gender_1 = self.metadata_dic[sampled_speaker_id_1]["gender"]
        if sampled_query == "out_mix_gender" or sampled_query == "in_mix_same_gender":
            sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            while (sampled_speaker_id_2 == sampled_speaker_id_1 or
                   self.metadata_dic[sampled_speaker_id_2]["gender"] != gender_1):
                sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            sampled_speaker_ids = [sampled_speaker_id_1, sampled_speaker_id_2]
        elif sampled_query == "in_mix_cross_gender":
            sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            while (sampled_speaker_id_2 == sampled_speaker_id_1 or
                   self.metadata_dic[sampled_speaker_id_2]["gender"] == gender_1):
                sampled_speaker_id_2 = np.random.choice(self.valid_speakers)
            sampled_speaker_ids = [sampled_speaker_id_1, sampled_speaker_id_2]
        else:
            # Sample two speakers
            sampled_speaker_ids = np.random.choice(self.valid_speakers, 2, replace=False)
        sampled_speaker_genders = [self.metadata_dic[s]["gender"]
                                   for s in sampled_speaker_ids]

        # Sample the files
        wav_paths = []
        for sampled_speeaker_id in sampled_speaker_ids:
            this_wav_path = np.random.choice(
                self.metadata_dic[sampled_speeaker_id]['flac_paths'], 1)[0]
            wav_paths.append(this_wav_path)

        # Sample room parameters
        room_params = self.sample_room_parameters()

        if sampled_query in ['whispering', 'louder']:
            input_min_snr = 1.
            # For the cases where we want to condition on loudness we want to
            # have sources with distinct gains.
        else:
            input_min_snr = 0.

        # To also make a general version for sampling the near/far field
        if sampled_query in ['near_field', 'far_field']:
            sources_are_near_field = [True, False]
        else:
            sampled_spatial_perm_ind = np.random.randint(
                low=0, high=len(self.valid_spatial_perms))
            sources_are_near_field = self.valid_spatial_perms[
                sampled_spatial_perm_ind]

        sources_tensor, snr_ratio = self.get_sources_tensors(
            wav_paths, room_params, input_min_snr=input_min_snr,
            sources_are_near_field=sources_are_near_field)

        # Assume that always the first speaker is the near-field and the other is
        # far-field
        (target_tensor, q_condition_str, specific_query_id) = \
            self.sample_conditional_target(
                sources_tensor, sampled_query, snr_ratio, sampled_speaker_genders)

        return sources_tensor, target_tensor, q_condition_str, specific_query_id


def test_generator():
    def get_snr(tensor_1, tensor_2):
        return 10. * torch.log10((tensor_1**2).sum(-1) / ((tensor_2**2).sum(-1) + 1e-9))

    batch_size = 4
    n_jobs=4
    sample_rate = 8000
    timelength = 5.0
    overlap_ratio = 0.5
    time_samples = int(sample_rate * timelength)
    valid_queries = ["in_mix_same_gender", "in_mix_cross_gender", "out_mix_gender",
                     "near_field", "far_field",
                     "louder", "whispering"]
    queries_priors = [0.1] + [0.3] + [0.1] + [0.2] * 2 + [0.05] * 2
    input_max_snr = 2.5
    data_loader = Dataset(
        split='train',
        input_max_snr=input_max_snr,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        timelength=timelength,
        valid_queries=valid_queries,
        queries_priors=queries_priors,
        zero_pad=True,
        augment=True,
        n_samples=10)
    generator = data_loader.get_generator(batch_size=batch_size,
                                          num_workers=n_jobs,
                                          pin_memory=False)

    before = time()
    for data in generator:
        (sources_tensor, target_tensor, q_condition_one_hot,
         q_condition_str) = data
        assert sources_tensor.shape == target_tensor.shape

        actual_snrs = get_snr(sources_tensor[:, 0], sources_tensor[:, 1])
        for b in range(actual_snrs.shape[0]):
            assert -input_max_snr <= actual_snrs[b] <= input_max_snr
        print(q_condition_str)
        print(q_condition_one_hot)
    this_time = time() - before
    print(this_time)

    # For eval scenarios we can specify the condition
    one_valid_query_l = ["in_mix_cross_gender"]
    invalid_queries = [q for q in valid_queries if q not in one_valid_query_l]
    queries_priors = [1.]
    data_loader = Dataset(
        split='test',
        input_max_snr=input_max_snr,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        timelength=timelength,
        valid_queries=one_valid_query_l,
        queries_priors=queries_priors,
        zero_pad=True,
        augment=True,
        n_samples=10)
    generator = data_loader.get_generator(batch_size=batch_size,
                                          num_workers=n_jobs,
                                          pin_memory=False)

    for data in generator:
        (sources_tensor, target_tensor, q_condition_one_hot,
         q_condition_str) = data
        assert all([q not in invalid_queries for q in q_condition_str])

if __name__ == "__main__":
    test_generator()
