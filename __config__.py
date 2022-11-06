# © MERL 2022
# Created by Efthymios Tzinis
"""Declare path variables used in this project."""
import os

"""Fill in the path with the absolute path of the repo."""
ROOT_DIRPATH = "/home/thymios/MERL/code/github_projects/heterogeneous_separation/"

"""Datasets paths"""
WHAM_ROOT_PATH = "/db/original/public/WHAM/wham/"
WHAMR_ROOT_PATH = "/db/original/public/WHAM/whamr/"

"""Metadata text files for datasets"""
WHAM_METADATA_P = os.path.join(
    ROOT_DIRPATH, "heterogeneous_separation/dataset_loader/wham_speaker_gender_info.txt")
WHSJ02MIX_METADATA_P = os.path.join(
    ROOT_DIRPATH, "heterogeneous_separation/dataset_loader/wsj02mix_metadata.txt")

# Path fo
VOXFORGE_DATA_P = "/mnt/data/Speech//voxforge_alllang"
VOXFORGE_METADATA_P = os.path.join(
    ROOT_DIRPATH, "heterogeneous_separation/dataset_loader/voxforge_metadata.pkl")

# Point to the directory where all librispeech partitions lie in:
# e.g. train-clean-360, test-clean, etc.
LIBRISPEECH_DATA_P = "/mnt/data/Speech/librispeech"
# Do not change the path below.
LIBRISPEECH_METADATA_P = os.path.join(
    ROOT_DIRPATH, "heterogeneous_separation/dataset_loader/librispeech_metadata.pkl")
