# © MERL 2022
# Created by Efthymios Tzinis
"""Declare path variables used in this project."""
import os

"""Fill in the path with the absolute path of the repo."""
ROOT_DIRPATH = "/home/thymios/MERL/code/github_projects/heterogeneous_separation/"



# Point to the directory where all librispeech partitions lie in:
# e.g. train-clean-360, test-clean, etc.
LIBRISPEECH_DATA_P = "/mnt/data/Speech/librispeech"
# Do not change the path below.
LIBRISPEECH_METADATA_P = os.path.join(
    ROOT_DIRPATH, "heterogeneous_separation/dataset_loader/librispeech_metadata.pkl")
