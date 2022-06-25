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

# Custom Classes
import preprocess_mine
from tqdm import tqdm

num_mcep = 36
sampling_rate = 16000
frame_period = 5.0
n_frames = 128
thread_num = 32


class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(self.args)

    def get_result(self):
        try:
            return self.result   # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def load_process(train_dir, class_id):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()
    for file in tqdm(os.listdir(os.path.join(train_dir, 'raw_' + class_id))):
        if "f0s" in file:
            f0s.extend()

        file_path = os.path.join(train_dir, file)
        wav, _ = librosa.load(file_path, sr=sampling_rate, mono=True)
        wavs.append(wav)
        i = i + 1
        if i % run_i == 0:
            t = MyThread(func=chunk_process, args=(wavs))
            thread_list.append(t)
            wavs = list()
            if i % (run_i * cpu_num) == 0:
                for t in thread_list:
                    t.setDaemon(True)
                    t.start()
                for t in thread_list:
                    t.join()
                for t in thread_list:
                    f0s_, timeaxes_, sps_, aps_, coded_sps_ = t.get_result()
                    f0s.extend(f0s_)
                    timeaxes.extend(timeaxes_)
                    sps.extend(sps_)
                    aps.extend(aps_)
                    coded_sps.extend(coded_sps_)
                thread_list = list()

    t = MyThread(func=chunk_process, args=(wavs))
    thread_list.append(t)
    wavs = list()

    for t in thread_list:
        t.setDaemon(True)
        t.start()
    for t in thread_list:
        t.join()
    for t in thread_list:
        f0s_, timeaxes_, sps_, aps_, coded_sps_ = t.get_result()
        f0s.extend(f0s_)
        timeaxes.extend(timeaxes_)
        sps.extend(sps_)
        aps.extend(aps_)
        coded_sps.extend(coded_sps_)
    thread_list = list()

    log_f0s_mean, log_f0s_std = preprocess_mine.logf0_statistics(f0s=f0s)
    coded_sps_norm, coded_sps_mean, coded_sps_std = preprocess_mine.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    save_pickle(variable=coded_sps_norm,
                fileName=os.path.join(cache_folder, str(i) + '_' + 'coded_sps_' + class_id + '_norm.pickle'))

    return log_f0s_mean, log_f0s_std, coded_sps_mean, coded_sps_std


def chunk_process(wavs):
    f0s, timeaxes, sps, aps, coded_sps = preprocess_mine.world_encode_data(
        wave=wavs, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    coded_sps_transposed = preprocess_mine.transpose_in_list(lst=coded_sps)
    return f0s, timeaxes, sps, aps, coded_sps_transposed

def preprocess_for_training(train_A_dir, train_B_dir, cache_folder):
    print("Starting to prepocess data.......")
    start_time = time.time()

    log_f0s_mean_A, log_f0s_std_A, coded_sps_A_mean, coded_sps_A_std = load_process(train_A_dir, 'A')
    log_f0s_mean_B, log_f0s_std_B, coded_sps_B_mean, coded_sps_B_std = load_process(train_B_dir, 'B')

    np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
             mean_A=log_f0s_mean_A,
             std_A=log_f0s_std_A,
             mean_B=log_f0s_mean_B,
             std_B=log_f0s_std_B)

    np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
             mean_A=coded_sps_A_mean,
             std_A=coded_sps_A_std,
             mean_B=coded_sps_B_mean,
             std_B=coded_sps_B_std)

    end_time = time.time()
    print("Preprocessing finsihed!! see your directory ../cache for cached preprocessed data")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for training Cycle GAN using PyTorch')
    train_A_dir_default = './data/S0913/'
    train_B_dir_default = './data/gaoxiaosong/'
    cache_folder_default = './cache/'
    #train_A_dir_default = './engb/BrE/'
    #train_B_dir_default = './engb/AmE/'
    #cache_folder_default = './cache_mine/'


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

    preprocess_for_training(train_A_dir, train_B_dir, cache_folder)
