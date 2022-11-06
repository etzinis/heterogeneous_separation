"""
Prepare the Voxforge dataset files by first searching them on the big index file
"""

from __config__ import VOXFORGE_DATA_P, VOXFORGE_METADATA_P
import argparse
from tqdm import tqdm
import csv
import os
import glob2
from pprint import pprint
import pickle
import numpy as np

RANDOM_SEED = 7
TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1

def extract_metadata_file():
    np.random.seed(RANDOM_SEED)
    failed_files = 0
    genders_dic = {}
    speakers_dic = {}
    languages_dirs = os.listdir(VOXFORGE_DATA_P)
    for lang_id in languages_dirs:
        if lang_id not in ['en', 'es', 'de', 'fr']:
            continue
        if lang_id not in speakers_dic:
            speakers_dic[lang_id] = {}

        sessions_dir = os.path.join(VOXFORGE_DATA_P, lang_id, 'extracted')
        session_dirs_list = os.listdir(sessions_dir)
        for session_id in tqdm(session_dirs_list):
            this_session_id_path = os.path.join(sessions_dir, session_id)
            wavs_paths = glob2.glob(this_session_id_path + '/wav/*.wav')
            metadata_filepath = os.path.join(this_session_id_path, 'etc/README')
            try:
                with open(metadata_filepath, 'rb') as f:
                    metadata_lines = f.read().decode('ISO-8859-1').split('\n')
            except Exception as e:
                failed_files += 1
                continue

            gender = None
            username = session_id.split('-')[0]
            for line in metadata_lines:
                if line.startswith('Gender:'):
                    gender = line.split('\n')[0].split()[-1]

            if gender not in genders_dic:
                genders_dic[gender] = 1
            else:
                genders_dic[gender] += 1

            if username not in speakers_dic[lang_id]:
                random_draw = np.random.random()
                if random_draw <= TRAIN_PERCENTAGE:
                    split = 'train'
                elif random_draw <= TRAIN_PERCENTAGE + TEST_PERCENTAGE:
                    split = 'test'
                else:
                    split = 'val'
                speakers_dic[lang_id][username] = {
                    'gender': gender,
                    'language': lang_id,
                    'wav_paths': wavs_paths,
                    'split': split
                }
            else:
                speakers_dic[lang_id][username]['wav_paths'] += wavs_paths

    print(f"Failed files: {failed_files}")
    print(sorted(speakers_dic.keys()))

    print(f"Saving file to: {VOXFORGE_METADATA_P}")
    pickle.dump(speakers_dic, open(VOXFORGE_METADATA_P, 'wb'))

if __name__ == "__main__":
    extract_metadata_file()
