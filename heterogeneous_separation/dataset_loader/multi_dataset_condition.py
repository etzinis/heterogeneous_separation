# © MERL 2022
# Created by Efthymios Tzinis
"""
Heterogeneous speech separation with multiple datasets.
"""

import heterogeneous_separation.dataset_loader.abstract_dataset as abstract_dataset
import heterogeneous_separation.dataset_loader.spatial_voxforge as \
    spatial_voxforge_dataset
import heterogeneous_separation.dataset_loader.spatial_librispeech as \
    spatial_librispeech_dataset
import heterogeneous_separation.dataset_loader.gender_loudness_wsj02mix as wsj02mix

import torch
import numpy as np
from time import time


VALID_DATASETS_MAP = {
    "SPATIAL_VOXFORGE": spatial_voxforge_dataset,
    "SPATIAL_LIBRISPEECH": spatial_librispeech_dataset,
    "WSJ02MIX": wsj02mix,
}


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

        self.datasets_configs = self.get_arg_and_check_validness(
            'datasets_configs', known_type=list,
            extra_lambda_checks=[
                lambda y: [(x["name"] in VALID_DATASETS_MAP) for x in y]])

        self.sample_rate = self.get_arg_and_check_validness('sample_rate',
                                                            known_type=int)

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.overlap_ratio = self.get_arg_and_check_validness(
            'overlap_ratio', known_type=float,
            extra_lambda_checks=[lambda x: 1 >= x >= 0])

        self.dataset_priors = self.get_arg_and_check_validness(
            'dataset_priors', known_type=list,
            extra_lambda_checks=[lambda y: (sum(y) - 1.) < 1e-5,
                                 lambda y: len(y) == len(self.datasets_configs)])
        self.datasets_cdf = np.cumsum(self.dataset_priors).tolist()

        # Define the dataset loaders for the individual datasets
        self.datasets_loaders = []
        self.conditional_queries = []
        for d_config in self.datasets_configs:
            try:
                extra_kwargs = dict([(k, v) for k, v in d_config.items() if k != 'name'])
                this_dataset_loader = VALID_DATASETS_MAP[d_config["name"]].Dataset(
                    split=d_config.get("split", self.split),
                    overlap_ratio=d_config.get("overlap_ratio", self.overlap_ratio),
                    sample_rate=d_config.get("sample_rate", self.sample_rate),
                    timelength=d_config.get("timelength", self.timelength),
                    zero_pad=d_config.get("zero_pad", self.zero_pad),
                    augment=d_config.get("augment", self.augment),
                    n_samples=d_config.get("n_samples", self.n_samples),
                    **extra_kwargs
                )
                self.datasets_loaders.append(this_dataset_loader)

                for cond in VALID_DATASETS_MAP[d_config["name"]].VALID_CONDITIONS:
                    if cond not in self.conditional_queries:
                        self.conditional_queries.append(cond)
            except Exception as e:
                print(e)
                raise ValueError(f"Dataset initialization problem in: {d_config['name']}")

        # Extract the unique conditional query strings
        self.condition_str_to_int = dict(
            [(x, i) for i, x in enumerate(self.conditional_queries)])
        self.conditional_emb_dim = len(self.conditional_queries)

    def __len__(self):
        return self.n_samples

    def sample_conditional_dataset(self):
        random_draw = np.random.random()
        selected_dataset_loader = self.datasets_loaders[-1]
        selected_dataset_name = self.datasets_configs[-1]["name"]
        for j, (dset_cdf_val, dset_loader) in enumerate(zip(self.datasets_cdf,
                                                            self.datasets_loaders)):
            if random_draw <= dset_cdf_val:
                selected_dataset_loader = dset_loader
                selected_dataset_name = self.datasets_configs[j]["name"]
                break
        return selected_dataset_loader, selected_dataset_name

    def get_one_hot(self, specific_query_id):
        """Converts to one hot encoding"""
        encoded_id = self.condition_str_to_int[specific_query_id]
        # print(f"THe encoded id: {encoded_id}")

        return torch.nn.functional.one_hot(
            torch.tensor([encoded_id]).to(torch.long),
            num_classes=self.conditional_emb_dim).to(torch.float32)[0]

    def __getitem__(self, idx):
        if self.augment:
            seed = int(np.modf(time())[0] * 100000000)
        else:
            seed = idx
        np.random.seed(seed)

        # Random sample from which dataset we are going to draw the signals and
        # conditions. We ignore the individual dataset's q_one_hot_encoding.
        sampled_dataset_loader, sampled_dataset_name = self.sample_conditional_dataset()
        (sources_tensor, target_tensor, q_condition_str,
         specific_query_id) = sampled_dataset_loader.get_multidataset_item(idx)

        q_condition_one_hot = self.get_one_hot(specific_query_id)

        return (sources_tensor, target_tensor, q_condition_one_hot,
                q_condition_str + "|" + specific_query_id, sampled_dataset_name)


def test_generator():
    def get_snr(tensor_1, tensor_2):
        return 10. * torch.log10((tensor_1**2).sum(-1) / ((tensor_2**2).sum(-1) + 1e-9))

    batch_size = 4
    n_jobs = 12
    sample_rate = 8000
    timelength = 5.0
    overlap_ratio = 0.5
    time_samples = int(sample_rate * timelength)

    datasets_configs = [
        {
            "name": "WSJ02MIX",
            "valid_queries": [
                "in_mix_same_gender", "in_mix_cross_gender", "out_mix_gender",
                "louder", "whispering"],
            "queries_priors": [0.1] + [0.35] + [0.05] + [0.2] * 2,
            "input_max_snr": 5.
        },
        {
            "name": "SPATIAL_VOXFORGE",
            "valid_languages": ['de', 'en', 'es', 'fr'],
            "valid_queries": ["in_mix_same_lang", "in_mix_cross_lang", "out_mix_lang",
                              "near_field", "far_field"],
            "queries_priors": [0.0, 1.0] + [0.0] * 3,
            "input_max_snr": 2.5
        },
        {
            "name": "SPATIAL_LIBRISPEECH",
            "valid_queries": [
                "in_mix_same_gender", "in_mix_cross_gender", "out_mix_gender",
                "near_field", "far_field", "louder", "whispering"],
            "queries_priors": [0.0] + [0.0] + [0.0] + [0.5] * 2 + [0.0] * 2,
            "input_max_snr": 2.5
        }
    ]
    dataset_priors = [0.5, 0.0, 0.5]
    # dataset_priors = [0., 1.]
    # dataset_priors = [1., 0.]

    data_loader = Dataset(
        split='train',
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        timelength=timelength,
        datasets_configs=datasets_configs,
        dataset_priors=dataset_priors,
        zero_pad=True,
        augment=True,
        n_samples=10)
    generator = data_loader.get_generator(batch_size=batch_size,
                                          num_workers=n_jobs,
                                          pin_memory=False)

    before = time()
    for data in generator:
        (sources_tensor, target_tensor, q_condition_one_hot,
         q_condition_str, selected_dataset_name) = data
        assert sources_tensor.shape == target_tensor.shape

        actual_snrs = get_snr(sources_tensor[:, 0], sources_tensor[:, 1])
        for b in range(actual_snrs.shape[0]):
            if selected_dataset_name[b] == "WSJ02MIX":
                this_input_max_snr = datasets_configs[0]["input_max_snr"]
            elif selected_dataset_name[b] == "SPATIAL_VOXFORGE":
                this_input_max_snr = datasets_configs[1]["input_max_snr"]
            else:
                this_input_max_snr = datasets_configs[2]["input_max_snr"]

            assert -this_input_max_snr <= actual_snrs[b] <= this_input_max_snr
        print("==="*10)
        print(f"Sampled datasets:")
        print(selected_dataset_name)
        print("===" * 10)
        print(f"Query condition string:")
        print(q_condition_str)
        print("===" * 10)
        print(f"Query condition one-hot vectors:")
        print(q_condition_one_hot)
    this_time = time() - before
    print(f"Fetched batch in: {this_time} secs")

    # For eval scenarios we can specify the condition
    datasets_configs = [
        {
            "name": "SPATIAL_VOXFORGE",
            "valid_languages": ['de', 'en', 'es', 'fr'],
            "valid_queries": ["out_mix_lang"],
            "queries_priors": [1.],
            "input_max_snr": 2.5
        },
        {
            "name": "SPATIAL_LIBRISPEECH",
            "valid_queries": [
                "in_mix_same_gender", "in_mix_cross_gender", "out_mix_gender",
                "near_field", "far_field", "louder", "whispering"],
            "queries_priors": [0.1] + [0.3] + [0.1] + [0.2] * 2 + [0.05] * 2,
            "input_max_snr": 2.5
        }
    ]
    dataset_priors = [0., 1.0]

    data_loader = Dataset(
        split='test',
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
        timelength=timelength,
        datasets_configs=datasets_configs,
        dataset_priors=dataset_priors,
        zero_pad=True,
        augment=True,
        n_samples=10)
    generator = data_loader.get_generator(batch_size=batch_size,
                                          num_workers=n_jobs,
                                          pin_memory=False)

    for data in generator:
        (sources_tensor, target_tensor, q_condition_one_hot,
         q_condition_str, selected_dataset_name) = data
        print()
        print("===" * 10)
        print(f"Sampled datasets:")
        print(selected_dataset_name)
        print("===" * 10)
        print(f"Query condition string (semantic signal characteristic| concept value):")
        print(q_condition_str)
        print("===" * 10)
        print(f"Query condition one-hot vectors:")
        print(q_condition_one_hot)

if __name__ == "__main__":
    test_generator()
