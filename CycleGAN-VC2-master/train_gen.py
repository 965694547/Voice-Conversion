#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 14:25

import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle
import soundfile as sf

import preprocess
from trainingDataset import trainingDataset
from model_tf import Generator, Discriminator
from tqdm import tqdm
from model_disc import Disc
from shutil import copyfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CycleGANTraining(object):
    def __init__(self,
                 log_f0s_mean_A,
                 log_f0s_std_A,
                 coded_sps_A_mean,
                 coded_sps_A_std,
                 log_f0s_mean_B,
                 log_f0s_std_B,
                 coded_sps_B_mean,
                 coded_sps_B_std,
                 coded_sps_A_norm,
                 coded_sps_B_norm,
                 model_checkpoint,
                 validation_A_dir,
                 output_A_dir,
                 validation_B_dir,
                 output_B_dir,
                 disc_model_dir=None,
                 restart_training_at=None):
        self.start_epoch = 0
        self.num_epochs = 200000  # 5000
        self.mini_batch_size = 6  # 1
        #self.dataset_A = self.loadPickleFile(coded_sps_A_norm)
        #self.dataset_B = self.loadPickleFile(coded_sps_B_norm)
        self.dataset_A = coded_sps_A_norm
        self.dataset_B = coded_sps_B_norm
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Speech Parameters
        self.log_f0s_mean_A = log_f0s_mean_A
        self.log_f0s_std_A = log_f0s_std_A
        self.log_f0s_mean_B = log_f0s_mean_B
        self.log_f0s_std_B = log_f0s_std_B

        self.coded_sps_A_mean = coded_sps_A_mean
        self.coded_sps_A_std = coded_sps_A_std
        self.coded_sps_B_mean = coded_sps_B_mean
        self.coded_sps_B_std = coded_sps_B_std

        # Generator and Discriminator
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        # Loss Functions
        criterion_mse = torch.nn.MSELoss()

        # Optimizer
        g_params = list(self.generator_A2B.parameters()) + \
                   list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
                   list(self.discriminator_B.parameters())

        # Initial learning rates
        self.generator_lr = 2e-4  # 0.0002
        self.discriminator_lr = 1e-4  # 0.0001

        # Learning rate decay
        self.generator_lr_decay = self.generator_lr / 200000
        self.discriminator_lr_decay = self.discriminator_lr / 200000

        # Starts learning rate decay from after this many iterations have passed
        self.start_decay = 10000  # 200000

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        # To Load save previously saved models
        self.modelCheckpoint = model_checkpoint
        os.makedirs(self.modelCheckpoint, exist_ok=True)

        # Validation set Parameters
        self.validation_A_dir = validation_A_dir
        self.output_A_dir = output_A_dir
        os.makedirs(self.output_A_dir, exist_ok=True)
        self.validation_B_dir = validation_B_dir
        self.output_B_dir = output_B_dir
        os.makedirs(self.output_B_dir, exist_ok=True)

        # Storing Discriminatior and Generator Loss
        self.generator_loss_store = []
        self.discriminator_loss_store = []

        self.file_name = 'log_store_non_sigmoid.txt'
        self.num_iterations = 0

        self.validation_num = 500
        self.cut_len = 1500000

        self.generator_loss = torch.tensor(0)
        self.d_loss = torch.tensor(0)

        if restart_training_at is not None:
            # Training will resume from previous checkpoint
            self.start_epoch = self.loadModel(restart_training_at)
            self.num_iterations = min(len(os.listdir(self.validation_A_dir)), len(os.listdir(self.validation_B_dir))) * self.start_epoch
            print("Training resumed")

        if disc_model_dir is not None:
            self.Disc = Disc().to(self.device)
            path = os.listdir(disc_model_dir)[-1]
            path = os.path.join(disc_model_dir, path)
            checkPoint = torch.load(path)
            self.Disc.load_state_dict(
                state_dict=checkPoint['model_state_dict'])

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def test_save(self, epoch, start_time_epoch):
        # Validation Set
        validation_start_time = time.time()
        AUC_A2B_x, AUC_A2B_y = self.validation_for_A_dir(epoch)
        print('Test Acc A2B x: {:.4f}, Test Acc A2B y: {:.4f}'.format(AUC_A2B_x, AUC_A2B_y))
        AUC_B2A_x, AUC_B2A_y = self.validation_for_B_dir(epoch)
        print('Test Acc B2A x: {:.4f}, Test Acc B2A y: {:.4f}'.format(AUC_B2A_x, AUC_B2A_y))
        validation_end_time = time.time()
        print('Test Acc x: {:.4f}, Test Acc y: {:.4f}'.format((AUC_A2B_x + AUC_B2A_x) / 2, (AUC_A2B_y + AUC_B2A_y) / 2))
        store_to_file = "Time taken for validation Set: {}".format(
            validation_end_time - validation_start_time)
        self.store_to_file(store_to_file)
        print("Time taken for validation Set: {}".format(
            validation_end_time - validation_start_time))

        end_time = time.time()
        store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
            epoch, self.generator_loss.item(), self.d_loss.item(), end_time - start_time_epoch)
        self.store_to_file(store_to_file)
        print("Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
            epoch, self.generator_loss.item(), self.d_loss.item(), end_time - start_time_epoch))

        # Save th:e Entire model
        print("Saving model Checkpoint  ......")
        store_to_file = "Saving model Checkpoint  ......"
        self.store_to_file(store_to_file)
        self.saveModelCheckPoint(epoch, '{}'.format(
            os.path.join(self.modelCheckpoint, str(epoch) + '_' + str((AUC_A2B_y + AUC_B2A_y) / 2) + '_CycleGAN_CheckPoint')))
        print("Model Saved!")


    def save_test(self, epoch, start_time_epoch):
        end_time = time.time()
        store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
            epoch, self.generator_loss.item(), self.d_loss.item(), end_time - start_time_epoch)
        self.store_to_file(store_to_file)
        print("Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
            epoch, self.generator_loss.item(), self.d_loss.item(), end_time - start_time_epoch))

        # Save the Entire model
        print("Saving model Checkpoint  ......")
        store_to_file = "Saving model Checkpoint  ......"
        self.store_to_file(store_to_file)
        self.saveModelCheckPoint(epoch, '{}'.format(
            os.path.join(self.modelCheckpoint, str(epoch) + '_CycleGAN_CheckPoint')))
 
        # Validation Set
        validation_start_time = time.time()
        AUC_A2B_x, AUC_A2B_y = self.validation_for_A_dir(epoch)
        AUC_B2A_x, AUC_B2A_y = self.validation_for_B_dir(epoch)
        validation_end_time = time.time()
        print('Test Acc x: {:.4f}, Test Acc y: {:.4f}'.format((AUC_A2B_x + AUC_B2A_x) / 2, (AUC_A2B_y + AUC_B2A_y) / 2))
        store_to_file = "Time taken for validation Set: {}".format(
            validation_end_time - validation_start_time)
        self.store_to_file(store_to_file)
        print("Time taken for validation Set: {}".format(   
            validation_end_time - validation_start_time))
    
    def train(self):
        # Training Begins
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time_epoch = time.time()

            # Preparing Dataset
            if chunk_size_A > chunk_size_B:
                mini_epoch = 1
                for data_A_, data_B_ in zip(self.dataset_A, self.dataset_B):
                    data_B = self.loadPickleFile(data_B_)
                    data_A = list()
                    for data_A__ in data_A_:
                        data_A.extend(self.loadPickleFile(data_A__))
                    self.train_batch(data_A, data_B)
                    print("Mini epoch: {}".format(mini_epoch))
                    mini_epoch += 1
                    if mini_epoch % 20 == 0:
                        self.test_save(epoch, start_time_epoch)

            if chunk_size_A < chunk_size_B:
                mini_epoch = 1
                for data_A_, data_B_ in zip(self.dataset_A, self.dataset_B):
                    data_A = self.loadPickleFile(data_A_)
                    data_B = list()
                    for data_B__ in data_B_:
                        data_B.extend(self.loadPickleFile(data_B__))
                    self.train_batch(data_A, data_B)
                    print("Mini epoch: {}".format(mini_epoch))
                    mini_epoch += 1
                    if mini_epoch % 20 == 0:
                        self.test_save(epoch, start_time_epoch)

            if chunk_size_A == chunk_size_B:
                mini_epoch = 1
                for data_A_, data_B_ in zip(self.dataset_A, self.dataset_B):
                    data_A = self.loadPickleFile(data_A_)
                    data_B = self.loadPickleFile(data_B_)
                    self.train_batch(data_A, data_B)
                    print("Mini epoch: {}".format(mini_epoch))
                    mini_epoch += 1
                    if mini_epoch % 20 == 0:
                        self.test_save(epoch, start_time_epoch)

            if epoch % 1 == 0:
                self.test_save(epoch, start_time_epoch)

    def train_batch(self, data_A, data_B):
        # Constants
        cycle_loss_lambda = 10
        identity_loss_lambda = 5

        dataset = trainingDataset(datasetA=data_A,
                                  datasetB=data_B,
                                  n_frames=400)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.mini_batch_size,
                                                   shuffle=True,
                                                   drop_last=False)

        pbar = tqdm(enumerate(train_loader))
        for i, (real_A, real_B) in enumerate(train_loader):
            #num_iterations = (n_samples // self.mini_batch_size) * epoch + i
            self.num_iterations = self.num_iterations + 1
            # print("iteration no: ", num_iterations, epoch)

            if self.num_iterations > 10000:
                identity_loss_lambda = 0
            if self.num_iterations > self.start_decay:
                self.adjust_lr_rate(
                    self.generator_optimizer, name='generator')
                self.adjust_lr_rate(
                    self.generator_optimizer, name='discriminator')

            real_A = real_A.to(self.device).float()
            real_B = real_B.to(self.device).float()

            # Generator Loss function

            fake_B = self.generator_A2B(real_A)
            cycle_A = self.generator_B2A(fake_B)

            fake_A = self.generator_B2A(real_B)
            cycle_B = self.generator_A2B(fake_A)

            identity_A = self.generator_B2A(real_A)
            identity_B = self.generator_A2B(real_B)

            d_fake_A = self.discriminator_A(fake_A)
            d_fake_B = self.discriminator_B(fake_B)

            # for the second step adverserial loss
            d_fake_cycle_A = self.discriminator_A(cycle_A)
            d_fake_cycle_B = self.discriminator_B(cycle_B)

            # Generator Cycle loss
            cycleLoss = torch.mean(
                torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

            # Generator Identity Loss
            identiyLoss = torch.mean(
                torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

            # Generator Loss
            generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
            generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

            # Total Generator Loss
            self.generator_loss = generator_loss_A2B + generator_loss_B2A + \
                             cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
            self.generator_loss_store.append(self.generator_loss.item())

            # Backprop for Generator
            self.reset_grad()
            self.generator_loss.backward()

            # if num_iterations > self.start_decay:  # Linearly decay learning rate
            #     self.adjust_lr_rate(
            #         self.generator_optimizer, name='generator')

            self.generator_optimizer.step()

            # Discriminator Loss Function

            # Discriminator Feed Forward
            d_real_A = self.discriminator_A(real_A)
            d_real_B = self.discriminator_B(real_B)

            generated_A = self.generator_B2A(real_B)
            d_fake_A = self.discriminator_A(generated_A)

            # for the second step adverserial loss
            cycled_B = self.generator_A2B(generated_A)
            d_cycled_B = self.discriminator_B(cycled_B)

            generated_B = self.generator_A2B(real_A)
            d_fake_B = self.discriminator_B(generated_B)

            # for the second step adverserial loss
            cycled_A = self.generator_B2A(generated_B)
            d_cycled_A = self.discriminator_A(cycled_A)

            # Loss Functions
            d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
            d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
            d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

            d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
            d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
            d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

            # the second step adverserial loss
            d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
            d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
            d_loss_A_2nd = (d_loss_A_real + d_loss_A_cycled) / 2.0
            d_loss_B_2nd = (d_loss_B_real + d_loss_B_cycled) / 2.0

            # Final Loss for discriminator with the second step adverserial loss
            self.d_loss = (d_loss_A + d_loss_B) / 2.0 + (d_loss_A_2nd + d_loss_B_2nd) / 2.0
            self.discriminator_loss_store.append(self.d_loss.item())

            # Backprop for Discriminator
            self.reset_grad()
            self.d_loss.backward()

            # if num_iterations > self.start_decay:  # Linearly decay learning rate
            #     self.adjust_lr_rate(
            #         self.discriminator_optimizer, name='discriminator')

            self.discriminator_optimizer.step()

            if (i + 1) % 2 == 0:
                #self.save_test(0, 0)
                pbar.set_description(
                    "Iter:{} Generator Loss:{:.4f} Discrimator Loss:{:.4f} GA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                        self.num_iterations,
                        self.generator_loss.item(),
                        # loss['generator_loss'],
                        self.d_loss.item(), generator_loss_A2B, generator_loss_B2A, identiyLoss, cycleLoss, d_loss_A,
                        d_loss_B))

        #                 if num_iterations % 50 == 0:
        #                     store_to_file = "Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
        #                         num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B, generator_loss_B2A,
        #                         identiyLoss, cycleLoss, d_loss_A, d_loss_B)
        #                     print(
        #                         "Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
        #                             num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B,
        #                             generator_loss_B2A, identiyLoss, cycleLoss, d_loss_A, d_loss_B))
        #                     self.store_to_file(store_to_file)

        #             end_time = time.time()
        #             store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
        #                 epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch)
        #             self.store_to_file(store_to_file)
        #             print("Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
        #                 epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch))

    def correct_count(self, x, class_id):
        with torch.no_grad():
            pre_y = self.Disc(x)
        if class_id == 'A':
            y = [0] * len(x)
        if class_id == 'B':
            y = [1] * len(x)
        #res_pre_y = torch.argmax(pre_y, dim=-1)
        res_pre_y = torch.tensor([int(x) for x in pre_y >= 0.5], dtype=torch.float32)
        res_y = y
        res_pre_y_list = res_pre_y.cpu().numpy()
        res_y_list = res_y
        com = np.array(res_pre_y_list - res_y_list)
        cnt_array = np.where(com, 0, 1)
        return np.sum(cnt_array)

    def validation_for_A_dir(self, epoch):
        num_mcep = 36
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_A_dir = self.validation_A_dir
        output_A_dir = self.output_A_dir

        print("Generating Validation Data B from A...")
        file_list = os.listdir(validation_A_dir)
        np.random.shuffle(file_list)
        correct_num_x = 0
        correct_num_y = 0
        for file in tqdm(file_list[:self.validation_num]):
            filePath = os.path.join(validation_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)

            if len(wav) > self.cut_len:
                wav = wav[:self.cut_len]

            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_A,
                                                       std_log_src=self.log_f0s_std_A,
                                                       mean_log_target=self.log_f0s_mean_B,
                                                       std_log_target=self.log_f0s_std_B)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(coded_sp_norm)
            #pdb.set_trace()
            correct_num_x += self.correct_count(coded_sp_norm, 'A')
            correct_num_y += self.correct_count(coded_sp_converted_norm, 'B')

            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                                 self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            #librosa.output.write_wav(path=os.path.join(output_A_dir, os.path.basename(file)),
            #                         y=wav_transformed,
            #                         sr=sampling_rate)
            sf.write(file=os.path.join(output_A_dir, 'output' + '_' + str(epoch) + '_' + os.path.basename(file)),
                     data=wav_transformed,
                     samplerate=sampling_rate
                     )
            copyfile(filePath, os.path.join(output_A_dir, 'input' + '_' + str(epoch) + '_' + os.path.basename(file)))
        return 100.0 * correct_num_x / self.validation_num, 100.0 * correct_num_y / self.validation_num
        #print('Test Acc: {:.4f}'.format(100.0 * correct_num / self.validation_num))

    def validation_for_B_dir(self, epoch):
        num_mcep = 36
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_B_dir = self.validation_B_dir
        output_B_dir = self.output_B_dir

        print("Generating Validation Data A from B...")
        file_list = os.listdir(validation_B_dir)
        np.random.shuffle(file_list)
        correct_num_x = 0
        correct_num_y = 0
        for file in tqdm(file_list[:self.validation_num]):
            filePath = os.path.join(validation_B_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)

            if len(wav) > self.cut_len:
                wav = wav[:self.cut_len]

            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_B,
                                                       std_log_src=self.log_f0s_std_B,
                                                       mean_log_target=self.log_f0s_mean_A,
                                                       std_log_target=self.log_f0s_std_A)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_B_mean) / self.coded_sps_B_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_B2A(coded_sp_norm)
            #pdb.set_trace()
            correct_num_x += self.correct_count(coded_sp_norm, 'B')
            correct_num_y += self.correct_count(coded_sp_converted_norm, 'A')

            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                                 self.coded_sps_A_std + self.coded_sps_A_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            #librosa.output.write_wav(path=os.path.join(output_B_dir, os.path.basename(file)),
            #                         y=wav_transformed,
            #                         sr=sampling_rate)
            sf.write(file=os.path.join(output_B_dir, 'output' + '_' + str(epoch) + '_' + os.path.basename(file)),
                     data=wav_transformed,
                     samplerate=sampling_rate
                     )
            copyfile(filePath, os.path.join(output_B_dir, 'input' + '_' + str(epoch) + '_' + os.path.basename(file)))
        return 100.0 * correct_num_x / self.validation_num, 100.0 * correct_num_y / self.validation_num
        # print('Test Acc: {:.4f}'.format(100.0 * correct_num / self.validation_num))

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'discriminator_loss_store': self.discriminator_loss_store,
            'model_genA2B_state_dict': self.generator_A2B.state_dict(),
            'model_genB2A_state_dict': self.generator_B2A.state_dict(),
            'model_discriminatorA': self.discriminator_A.state_dict(),
            'model_discriminatorB': self.discriminator_B.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A.load_state_dict(
            state_dict=checkPoint['model_discriminatorA'])
        self.discriminator_B.load_state_dict(
            state_dict=checkPoint['model_discriminatorB'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']
        return epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset")
    '''
    logf0s_normalization_default = './cache/logf0s_normalization.npz'
    mcep_normalization_default = './cache/mcep_normalization.npz'
    coded_sps_A_norm = './cache/coded_sps_A_norm.pickle'
    coded_sps_B_norm = './cache/coded_sps_B_norm.pickle'
    model_checkpoint = './model_checkpoint/'
    resume_training_at = './model_checkpoint/_CycleGAN_CheckPoint'
    #     resume_training_at = None

    validation_A_dir_default = './data/S0913/'
    output_A_dir_default = './converted_sound/S0913'

    validation_B_dir_default = './data/gaoxiaosong/'
    output_B_dir_default = './converted_sound/gaoxiaosong/'
    '''

    #logf0s_normalization_default = './cache_mine/logf0s_normalization.npz'
    #mcep_normalization_default = './cache_mine/mcep_normalization.npz'
    #coded_sps_A_norm = './cache_mine/coded_sps_A_norm.pickle'
    #coded_sps_B_norm = './cache_mine/coded_sps_B_norm.pickle'

    cache_folder_default = './cache_disc/'
    model_checkpoint = 'model_checkpoint/'
    resume_training_at = '0_CycleGAN_CheckPoint'
    # resume_training_at = None

    validation_A_dir_default = './engb/data_disc/'
    output_A_dir_default = 'converted_sound/'

    validation_B_dir_default = './enus/data_disc/'
    output_B_dir_default = 'converted_sound/'

    disc_model_default = './disc/'

    tag_default = '_'

    '''
    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_A_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_A_norm)
    parser.add_argument('--coded_sps_B_norm', type=str,
                        help="mcep norm for data B", default=coded_sps_B_norm)
    '''
    parser.add_argument('--model_checkpoint', type=str,
                        help="location where you want to save the model", default=model_checkpoint)
    parser.add_argument('--resume_training_at', type=str,
                        help="Location of the pre-trained model to resume training",
                        default=resume_training_at)
    parser.add_argument('--validation_A_dir', type=str,
                        help="validation set for sound source A", default=validation_A_dir_default)
    parser.add_argument('--output_A_dir', type=str,
                        help="output for converted Sound Source A", default=output_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str,
                        help="Validation set for sound source B", default=validation_B_dir_default)
    parser.add_argument('--output_B_dir', type=str,
                        help="Output for converted sound Source B", default=output_B_dir_default)
    parser.add_argument('--tag', type=str,
                        help="tag for output", default=tag_default)
    parser.add_argument('--disc', type=str,
                        help="disc model", default=disc_model_default)

    argv = parser.parse_args()

    #logf0s_normalization = argv.logf0s_normalization
    #mcep_normalization = argv.mcep_normalization
    #coded_sps_A_norm = argv.coded_sps_A_norm
    #coded_sps_B_norm = argv.coded_sps_B_norm
    argv.tag = os.path.join('./', argv.tag)
    model_checkpoint = os.path.join(argv.tag, argv.model_checkpoint)
    if argv.resume_training_at != None:
        resume_training_at = os.path.join(model_checkpoint, argv.resume_training_at)
    else:
        resume_training_at = argv.resume_training_at

    validation_A_dir = argv.validation_A_dir
    output_A_dir = os.path.join(argv.tag, argv.output_A_dir, validation_A_dir.split('/')[1])
    validation_B_dir = argv.validation_B_dir
    output_B_dir = os.path.join(argv.tag, argv.output_B_dir, validation_B_dir.split('/')[1])

    if argv.disc != None:
        disc_model = os.path.join(argv.disc, 'model_checkpoint')
    else:
        disc_model = argv.disc

    coded_sps_A_norm = list()
    log_f0s_mean_A = 0
    log_f0s_std_A = 0
    coded_sps_A_mean = 0
    coded_sps_A_std = 0

    coded_sps_B_norm = list()
    log_f0s_mean_B = 0
    log_f0s_std_B = 0
    coded_sps_B_mean = 0
    coded_sps_B_std = 0

    global chunk_size_A
    global chunk_size_B
    chunk_size_A = 0
    chunk_size_B = 0
    for file in os.listdir(cache_folder_default):
        if "logf0s_mcep_normalization.npz" in file and "A" in file:
            npz = np.load(os.path.join(cache_folder_default, file))
            log_f0s_mean_A += npz["mean_f0s"]
            log_f0s_std_A += npz["std_f0s"]
            coded_sps_A_mean += npz["mean_sps"]
            coded_sps_A_std += npz["std_sps"]
        if "logf0s_mcep_normalization.npz" in file and "B" in file:
            npz = np.load(os.path.join(cache_folder_default, file))
            log_f0s_mean_B += npz["mean_f0s"]
            log_f0s_std_B += npz["std_f0s"]
            coded_sps_B_mean += npz["mean_sps"]
            coded_sps_B_std += npz["std_sps"]
        if "coded_sps_norm.pickle" in file and "A" in file:
            coded_sps_A_norm.append(os.path.join(cache_folder_default, file))
            chunk_size_A += 1
        if "coded_sps_norm.pickle" in file and "B" in file:
            coded_sps_B_norm.append(os.path.join(cache_folder_default, file))
            chunk_size_B += 1

    if chunk_size_A>chunk_size_B:
        coded_sps_A_norm = np.array_split(coded_sps_A_norm, chunk_size_B)
    if chunk_size_A<chunk_size_B:
        coded_sps_B_norm = np.array_split(coded_sps_B_norm, chunk_size_A)

    log_f0s_mean_A /= chunk_size_A
    log_f0s_std_A /= chunk_size_A
    coded_sps_A_mean /= chunk_size_A
    coded_sps_A_std /= chunk_size_A
    log_f0s_mean_B /= chunk_size_B
    log_f0s_std_B /= chunk_size_B
    coded_sps_B_mean /= chunk_size_B
    coded_sps_B_std /= chunk_size_B

    # Check whether following cached files exists
    if os.listdir(cache_folder_default) == None:
        print(
            "Cached files do not exist, please run the program preprocess_training.py first")

    cycleGAN = CycleGANTraining(log_f0s_mean_A=log_f0s_mean_A,
                                log_f0s_std_A=log_f0s_std_A,
                                coded_sps_A_mean=coded_sps_A_mean,
                                coded_sps_A_std=coded_sps_A_std,
                                log_f0s_mean_B=log_f0s_mean_B,
                                log_f0s_std_B=log_f0s_std_B,
                                coded_sps_B_mean=coded_sps_B_mean,
                                coded_sps_B_std=coded_sps_B_std,
                                coded_sps_A_norm=coded_sps_A_norm,
                                coded_sps_B_norm=coded_sps_B_norm,
                                model_checkpoint=model_checkpoint,
                                validation_A_dir=validation_A_dir,
                                output_A_dir=output_A_dir,
                                validation_B_dir=validation_B_dir,
                                output_B_dir=output_B_dir,
                                disc_model_dir=disc_model,
                                restart_training_at=resume_training_at)
    cycleGAN.train()
