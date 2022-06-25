import os
from tqdm import tqdm
from shutil import copyfile, move
import librosa
import sys
import soundfile as sf
import numpy as np
import random

num_mcep = 36
sampling_rate = 16000
frame_period = 5.0
n_frames = 128
thread_num = os.cpu_count()
chunk_size = 40

train_A_dir = './engb/'
train_B_dir = './enus/'


def len_test():
    min_len=sys.maxsize
    max_len=0
    sum_len=0
    num_file=0
    for file in tqdm(os.listdir(train_A_dir)):
        for wav_name in os.listdir(os.path.join(train_A_dir, file)):
            if ".wav" not in wav_name:
                continue
            wav_name = os.path.join(train_A_dir, file, wav_name)
            wav, _ = librosa.load(wav_name, sr=sampling_rate, mono=True)
            min_len = min(len(wav), min_len)
            max_len = max(len(wav), max_len)
            num_file += 1
            sum_len += len(wav)
    sum_len /= num_file
    print(train_A_dir + " min length: %s" % (str(min_len)))
    print(train_A_dir + " max length: %s" % (str(max_len)))
    print(train_A_dir + " avg length: %s" % (str(sum_len)))

    min_len=sys.maxsize
    max_len=0
    sum_len=0
    num_file=0
    for file in tqdm(os.listdir(train_B_dir)):
        for wav_name in os.listdir(os.path.join(train_B_dir, file)):
            if ".wav" not in wav_name:
                continue
            wav_name = os.path.join(train_B_dir, file, wav_name)
            wav, _ = librosa.load(wav_name, sr=sampling_rate, mono=True)
            min_len = min(len(wav), min_len)
            max_len = max(len(wav), max_len)
            num_file += 1
            sum_len += len(wav)
    sum_len /= num_file
    print(train_B_dir + " min length: %s" % (str(min_len)))
    print(train_B_dir + " max length: %s" % (str(max_len)))
    print(train_B_dir + " avg length: %s" % (str(sum_len)))

def mv_dir(train_dir, len_min, len_max):
    target = os.path.join(train_dir, "data_disc_trash")
    for file in tqdm(os.listdir(train_dir)):
        if "data" in file:
            continue
        for wav_name in os.listdir(os.path.join(train_dir, file)):
            if ".wav" not in wav_name:
                continue
            out_file = os.path.join(train_dir, file, wav_name)
            wav, samplerate = sf.read(out_file)
            if len(wav)<=len_min:
                continue
            elif len(wav)<=len_max:
                in_file = os.path.join(target, wav_name)
                copyfile(out_file, in_file)
            else:
                cut_length = len(wav) // len_min
                i = 0
                for w in np.array_split(wav, cut_length):
                    in_file = os.path.join(target, str(i) + '_' + wav_name)
                    sf.write(in_file, w[:len_max], samplerate)
                    i = i + 1

def mv_half(train_dir):
    origin = os.path.join(train_dir, "data_disc")
    target = os.path.join(train_dir, "data_disc_trash")
    file_list = os.listdir(origin)
    length = len(file_list) // 2
    random.shuffle(file_list)
    for file in tqdm(file_list[:length]):
        move(os.path.join(origin, file), os.path.join(target, file))

def error_process(train_A_dir, train_B_dir):
    origin_A = os.path.join(train_A_dir, "data_disc")
    origin_B = os.path.join(train_B_dir, "data_disc")

    file_list_A = os.listdir(origin_A)
    random.shuffle(file_list_A)
    file_list_B = os.listdir(origin_B)
    random.shuffle(file_list_B)

    length_A = len(file_list_A) // 2
    length_B = len(file_list_B) // 2

    for file in tqdm(file_list_A[:length_A]):
        in_A =  os.path.join(train_A_dir, "data_disc", file)
        out_A = os.path.join(train_A_dir, "data_error", file)
        copyfile(in_A, out_A)

    for file in tqdm(file_list_B[:length_B]):
        in_A =  os.path.join(train_B_dir, "data_disc", file)
        out_A = os.path.join(train_A_dir, "data_error", file)
        copyfile(in_A, out_A)

    for file in tqdm(file_list_A[length_A:]):
        in_B =  os.path.join(train_A_dir, "data_disc", file)
        out_B = os.path.join(train_B_dir, "data_error", file)
        copyfile(in_B, out_B)

    for file in tqdm(file_list_B[length_B:]):
        in_B =  os.path.join(train_B_dir, "data_disc", file)
        out_B = os.path.join(train_B_dir, "data_error", file)
        copyfile(in_B, out_B)

if __name__ == '__main__':
    # len_test()
    # engb 16000  280800    159552
    # enus 834432 106041632 5130390
    # mv_dir(train_A_dir, 159552, 280800)
    # mv_dir(train_B_dir, 159552, 280800)
    # mv_half(train_B_dir)
    # mv_dir(train_A_dir, 140000, 159552)
    error_process(train_A_dir, train_B_dir)







