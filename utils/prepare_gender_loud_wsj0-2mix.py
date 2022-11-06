"""
Prepare the WSJ0-2mix dataset files by first searching them on the big index file
"""

from __config__ import WHAM_ROOT_PATH, WHSJ02MIX_METADATA_P, WHAM_METADATA_P
import argparse
from tqdm import tqdm
import csv
import os
import glob2
from pprint import pprint
import pickle
import numpy as np

RANDOM_SEED = 7

split_map = {
    'tr': 'train',
    'tt': 'test',
    'cv': 'val'
}


def extract_metadata_file():
    np.random.seed(RANDOM_SEED)
    sample_rate = 8000
    speakers_dic = {}

    gender_info_path = WHAM_METADATA_P
    print(gender_info_path)
    gender_dic = None
    if os.path.lexists(gender_info_path):
        with open(gender_info_path, 'rb') as filehandle:
            gender_dic = dict([tuple([x.decode() for x in l.split()])
                               for l in filehandle.readlines()])
    else:
        raise IOError(f"File: {gender_info_path} could not be parsed for gender "
                      f"information.")

    for split in tqdm(split_map.keys()):
        main_dir_path = os.path.join(
            WHAM_ROOT_PATH, 'wav{}k'.format(int(sample_rate / 1000)), 'min', split)

        sources_wav_paths_1 = os.path.join(main_dir_path, 's1')
        sources_wav_paths_2 = os.path.join(main_dir_path, 's2')
        available_source_files_1 = glob2.glob(sources_wav_paths_1 + '/*.wav')
        available_source_files_2 = glob2.glob(sources_wav_paths_2 + '/*.wav')

        for i, available_source_files in enumerate(
                [available_source_files_1, available_source_files_2]):
            for file_path in available_source_files:
                speaker_info = os.path.basename(file_path).split('.wav')[0]
                speaker_info = speaker_info.split('_')[i*2**i]
                this_speaker_id = speaker_info[:3]
                utterance_id = speaker_info[3:]
                this_speaker_gender = gender_dic[this_speaker_id]
                genders_info_str = this_speaker_gender.lower()

                if split == "cv":
                    this_speaker_id += "_val"

                if this_speaker_id not in speakers_dic:
                    speakers_dic[this_speaker_id] = {
                        'gender': genders_info_str,
                        'wav_paths': [file_path],
                        'split': split_map[split],
                        'utt_ids': set([utterance_id])
                    }
                else:
                    assert genders_info_str == speakers_dic[this_speaker_id]["gender"]
                    if utterance_id in speakers_dic[this_speaker_id]["utt_ids"]:
                        continue
                    else:
                        speakers_dic[this_speaker_id]["utt_ids"].add(utterance_id)
                        speakers_dic[this_speaker_id]['wav_paths'].append(file_path)

    print(f"Saving file to: {WHSJ02MIX_METADATA_P}")
    pickle.dump(speakers_dic, open(WHSJ02MIX_METADATA_P, 'wb'))


if __name__ == "__main__":
    extract_metadata_file()
