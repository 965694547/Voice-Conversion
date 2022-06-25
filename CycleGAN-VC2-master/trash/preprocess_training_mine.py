#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 15:25

import os
import time
import argparse
import numpy as np
import pickle
import librosa
import threading

from tqdm import tqdm

# Custom Classes
import preprocess_mine

num_mcep = 36
sampling_rate = 16000
frame_period = 5.0
n_frames = 128
chunk_size = 10



def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_for_training(train_A_dir, cache_folder, class_id):
    print("Starting to prepocess data.......")
    start_time = time.time()

    wavs_A = list()
    thread_list = list()
    i=0
    run_i = len(os.listdir(train_A_dir)) // chunk_size

    f0s_A
    timeaxes_A
    sps_A
    aps_A
    coded_sps_A

    for file_A in tqdm(os.listdir(train_A_dir)):
        file_path_A = os.path.join(train_A_dir, file_A)
        wav_A, _ = librosa.load(file_path_A, sr=sampling_rate, mono=True)
        # wav = wav.astype(np.float64)
        wavs_A.append(wav_A)
        i = i +1
        if(i % run_i==0):
            t = threading.Thread(target=chunk_process, args=(wavs_A, i // chunk_size, cache_folder, class_id))
            thread_list.append(t)
            del wavs_A
            wavs_A = list()

    t = threading.Thread(target=chunk_process, args=(wavs_A, i // chunk_size, cache_folder, class_id))
    thread_list.append(t)
    del wavs_A

    for t in thread_list:
        t.setDaemon(True)
        t.start()
    for t in thread_list:
        t.join()
    del thread_list

    end_time = time.time()
    print("Preprocessing finsihed!! see your directory ../cache for cached preprocessed data")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))

def chunk_process(wavs_A, i, cache_folder, class_id):
    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess_mine.world_encode_data(
        wave=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)



    log_f0s_mean_A, log_f0s_std_A = preprocess_mine.logf0_statistics(f0s=f0s_A)

    print("Log Pitch")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))

    coded_sps_A_transposed = preprocess_mine.transpose_in_list(lst=coded_sps_A)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess_mine.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_A_transposed)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    np.savez(os.path.join(cache_folder, str(i) + '_' + class_id + '_' + 'logf0s_normalization.npz'),
             mean=log_f0s_mean_A,
             std=log_f0s_std_A
             )

    np.savez(os.path.join(cache_folder, str(i) + '_' + class_id + '_' + 'mcep_normalization.npz'),
             mean=coded_sps_A_mean,
             std=coded_sps_A_std,
             )

    save_pickle(variable=coded_sps_A_norm,
                fileName=os.path.join(cache_folder, str(i) + '_' + 'coded_sps_' + class_id + '_norm.pickle'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for training Cycle GAN using PyTorch')
    #train_A_dir_default = './data/S0913/'
    #train_B_dir_default = './data/gaoxiaosong/'
    #cache_folder_default = './cache_mine/'
    train_A_dir_default = './engb/BrE/'
    train_B_dir_default = './engb/AmE/'
    cache_folder_default = './cache_mine/'


    parser.add_argument('--train_A_dir', type=str,
                        help="Directory for source voice sample", default=train_A_dir_default)
    parser.add_argument('--train_B_dir', type=str,
                        help="Directory for target voice sample", default=train_B_dir_default)
    parser.add_argument('--cache_folder', type=str,
                        help="Store preprocessed data in cache folders", default=cache_folder_default)
    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    cache_folder = argv.cache_folder

    preprocess_for_training(train_A_dir, cache_folder, 'A')
    preprocess_for_training(train_B_dir, cache_folder, 'B')
