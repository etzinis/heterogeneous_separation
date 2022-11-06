"""
Prepare the Voxforge dataset files by first searching them on the big index file
"""

from __config__ import LIBRISPEECH_DATA_P, LIBRISPEECH_METADATA_P
import argparse
from tqdm import tqdm
import csv
import os
import glob2
from pprint import pprint
import pickle
import numpy as np

RANDOM_SEED = 7

def extract_metadata_file():
    np.random.seed(RANDOM_SEED)
    failed_files = 0
    genders_dic = {}
    speakers_dic = {}

    naive_metadata_dic_p = os.path.join(LIBRISPEECH_DATA_P, "SPEAKERS.TXT")
    with open(naive_metadata_dic_p, 'r') as filehandle:
        lines = filehandle.readlines()

    metadata_dic = {}
    for line in tqdm(lines):
        if line.startswith(";") or not len(line.split("\n")[0].split("|")) >= 5:
            continue
        this_data = line.split("\n")[0].split("|")[:5]
        speaker_id, gender_str, libri_split, _, _ = this_data

        speaker_id = speaker_id.strip()
        gender_str = gender_str.strip().lower()
        libri_split = libri_split.strip()
        if libri_split == 'SUBSET':
            continue

        split = None
        if libri_split in ["train-clean-360", "train-clean-100"]:
            split = "train"
        elif libri_split in ["train-other-500", "dev-other", "test-other"]:
            continue
        elif libri_split == "dev-clean":
            split = "val"
        elif libri_split == "test-clean":
            split = "test"
        else:
            raise ValueError(f"The split: {libri_split} is invalid.")

        # Gather all wavfiles corresponding to that one speaker.from
        this_speaker_data_dirp = os.path.join(LIBRISPEECH_DATA_P, libri_split, speaker_id)
        flac_paths = glob2.glob(this_speaker_data_dirp + '/*/*.flac')
        if not flac_paths:
            this_speaker_data_dirp2 = os.path.join(
                LIBRISPEECH_DATA_P, libri_split, libri_split, speaker_id)
            flac_paths = glob2.glob(this_speaker_data_dirp2 + '/*/*.flac')

        if not flac_paths:
            raise ValueError(f"Speaker: {speaker_id} does not have any "
                             f"associated audio files!")

        metadata_dic[speaker_id] = {
            'gender': gender_str,
            'flac_paths': flac_paths,
            'split': split
        }

    print(f"Saving file to: {LIBRISPEECH_METADATA_P}")
    pickle.dump(metadata_dic, open(LIBRISPEECH_METADATA_P, 'wb'))


if __name__ == "__main__":
    extract_metadata_file()
