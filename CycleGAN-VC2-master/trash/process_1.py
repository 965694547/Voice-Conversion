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
import shutil

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


def load_process(train_dir, cache_folder, class_id, start, length):
    i = 0
    wavs = list()

    for file in tqdm(os.listdir(train_dir)[start:start+length]):
        file_path = os.path.join(train_dir, file)
        wav, _ = librosa.load(file_path, sr=sampling_rate, mono=True)
        wavs.append(wav)
    f0s, timeaxes, sps, aps, coded_sps_transposed = chunk_process(wavs)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    cache_folder = os.path.join(cache_folder, 'raw_' + class_id)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    else:
        shutil.rmtree(cache_folder)
    
    save_pickle(variable=f0s,
                fileName=os.path.join(cache_folder,str(start) + '_' + 'f0s.pickle'))
    save_pickle(variable=sps,
                fileName=os.path.join(cache_folder,str(start) + '_' + 'sps.pickle'))
    save_pickle(variable=aps,
                fileName=os.path.join(cache_folder,str(start) + '_' + 'aps.pickle'))
    save_pickle(variable=coded_sps_transposed,
                fileName=os.path.join(cache_folder,str(start) + '_' + 'coded_sps_transposed.pickle'))


def chunk_process(wavs):
    f0s, timeaxes, sps, aps, coded_sps = preprocess_mine.world_encode_data(
        wave=wavs, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    coded_sps_transposed = preprocess_mine.transpose_in_list(lst=coded_sps)
    return f0s, timeaxes, sps, aps, coded_sps_transposed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for training Cycle GAN using PyTorch')
    #train_A_dir_default = './data/S0913/'
    #train_B_dir_default = './data/gaoxiaosong/'
    #cache_folder_default = './cache/'
    #train_A_dir_default = './engb/BrE/'
    #train_B_dir_default = './engb/AmE/'
    cache_folder_default = './cache_mine/'

    parser.add_argument('--id', type=str,
                        help="class id")
    parser.add_argument('--train_dir', type=str,
                        help="Directory for source voice sample")
    parser.add_argument('--cache_folder', type=str,
                        help="Store preprocessed data in cache folders", default=cache_folder_default)
    parser.add_argument('--start', type=int,
                        help="start to porcess")
    parser.add_argument('--len', type=int,
                        help="len to process")
    argv = parser.parse_args()

    class_id = argv.id
    train_dir = argv.train_dir
    cache_folder = argv.cache_folder
    start = argv.start
    length = argv.len

    load_process(train_dir, cache_folder, class_id, start, length)
