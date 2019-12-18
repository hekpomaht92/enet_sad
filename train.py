import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from parts import Metrics, DCLoss
import config
from enet import generate_model
from utils import ImageGenerator, ImageGeneratorTest
from multiprocessing import set_start_method

np.random.seed(42)
set_start_method('spawn', True)
cfg = config.Config()


class TrainModel:
    def __init__(self, model):
        self.model = model
        self.model.to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
        class_weight = torch.Tensor(cfg.class_weight)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight).to(cfg.device)
        self.dcloss = DCLoss.apply
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [1, 3, 7], gamma=0.5)
        self.writer = SummaryWriter('logs/{}_lr={:06}_bz={}_clsW={}_iou={:02}_dst={}_clipW={}'\
            .format(time.time(), cfg.learning_rate, cfg.batch_size, cfg.class_weight,\
            cfg.lambda_dice, cfg.lambda_distilation, cfg.lambda_clip_weights))
        # self.write_graph()
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.train_dir, num_images=cfg.num_train_images),
            batch_size=cfg.batch_size, shuffle=True),
            'val': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.val_dir, num_images=cfg.num_val_images),
            batch_size=cfg.batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.test_dir, num_images=cfg.num_test_images),
            batch_size=1)
        }
        self.global_train_step = 0

    def write_graph(self):
        data = torch.zeros((cfg.batch_size, cfg.input_image_c,\
                            cfg.input_image_h, cfg.input_image_w)).to(cfg.device)
        self.writer.add_graph(self.model, data)

    def write_tensorboard_logs(self, phase, loss, log_step, label, pred, acc):
        self.writer.add_scalar('{} loss'.format(phase),
                                loss,
                                log_step)
        self.writer.add_scalar('{}_balanced_accuracy'.format(phase),
                                Metrics.balanced_accuracy(label, pred),
                                log_step)
        self.writer.add_scalar('{}_accuracy'.format(phase),
                                acc,
                                log_step)
        self.writer.add_scalar('{}_compute_tpr'.format(phase),
                                Metrics.compute_tpr(label, pred),
                                log_step)
        self.writer.add_scalar('{}_compute_tnr'.format(phase),
                                Metrics.compute_tnr(label, pred),
                                log_step)
        self.writer.add_scalar('{}_compute_ppv'.format(phase),
                                Metrics.compute_ppv(label, pred),
                                log_step)
        self.writer.add_scalar('{}_compute_npv'.format(phase),
                                Metrics.compute_npv(label, pred),
                                log_step)
    
    def write_tensorboard_images(self, pred, label, phase, log_step):
        pred = torch.unsqueeze(pred, 1).to('cpu').numpy()
        label = torch.unsqueeze(label, 1).to('cpu').numpy()
        if label.shape[0] % 8 == 0:
            imgs = np.concatenate((label, pred))
        else:
            imgs = np.concatenate((label, np.zeros((8-label.shape[0]%8, label.shape[1], label.shape[2], label.shape[3])),\
                pred))

        self.writer.add_images('label vs pred, phase: {}'.format(phase), imgs,\
            log_step, dataformats='NCHW')

    def train(self, epoch):
        self.model.train()
        running_loss = 0.
        for i, sample in enumerate(self.dataloaders['train'], 0):
            start_time = time.time()
            inputs = sample['data'].to(cfg.device)
            label = sample['label'].type(torch.LongTensor).to(cfg.device)

            self.optimizer.zero_grad()
            outputs, loss_distilation = self.model(inputs)
            pred = outputs.max(1)[1]

            loss_cros = self.criterion(outputs, label) * cfg.lambda_crossentropy
            loss_dc = self.dcloss(outputs[:,1,:,:], label) * cfg.lambda_dice
            loss_distilation *= cfg.lambda_distilation
            loss = (loss_cros + loss_dc + loss_distilation) / 3
            loss.backward()
            clip_grad_norm_(self.model.parameters(), cfg.lambda_clip_weights)
            self.optimizer.step()
            running_loss += loss.item()

            correct = pred.eq(label).sum().item()
            acc = int(correct) / (cfg.batch_size * cfg.input_image_h * cfg.input_image_w)

            time_train_batch = time.time() - start_time
            print("Phase: {}, Epoch: {:02}, Iter: {:05}, Loss: {:.3f}, Loss_cros: {:.3f}, Loss_dist: {:.3f}, Loss_dice: {:.3f}, Accuracy: {:.3f}, Time: {:.0f}m {:.0f}s"\
                .format('Train', epoch, i, loss.item(), loss_cros.item(), loss_distilation.item(), loss_dc.item(), acc, time_train_batch // 60, time_train_batch % 60))
            if i == 0:
                self.write_tensorboard_logs(phase='Train', loss=running_loss, log_step=self.global_train_step,\
                    label=label, pred=pred, acc=acc)
                self.write_tensorboard_images(pred, label, 'Train', self.global_train_step)
                self.global_train_step += 1
            elif i % 100 == 0:
                running_loss /= 100
                self.write_tensorboard_logs(phase='Train', loss=running_loss, log_step=self.global_train_step,\
                    label=label, pred=pred, acc=acc)
                self.write_tensorboard_images(pred, label, 'Train', self.global_train_step)
                running_loss = 0.
                self.global_train_step += 1

    def val(self, epoch):
        self.model.eval()
        running_loss = 0.
        with torch.no_grad():
            for i, sample in enumerate(self.dataloaders['val'], 0):
                start_time = time.time()
                inputs = sample['data'].to(cfg.device)
                label = sample['label'].type(torch.LongTensor).to(cfg.device)
                
                outputs, loss_distilation = self.model(inputs)
                pred = outputs.max(1)[1]
                loss = (self.criterion(outputs, label) + self.dcloss(outputs.to(cfg.device), label) * cfg.lambda_iou + loss_distilation * cfg.lambda_destilation) / 3

                correct = pred.eq(label).sum().item()
                acc = int(correct) / (cfg.batch_size * cfg.input_image_h * cfg.input_image_w)
                running_loss += loss.item()

                time_validation_batch = time.time() - start_time
                print("Phase: {}, Epoch: {:02}, Iter: {:05}, Loss: {:.5f}, Accuracy: {:.3f}, Time: {:.0f}m {:.0f}s"\
                    .format('Validation', epoch, i, loss.item(), acc, time_validation_batch // 60, time_validation_batch % 60))
            
            self.write_tensorboard_logs(phase='Validation', loss=running_loss / len(self.dataloaders['val']), log_step=epoch,\
                label=label, pred=pred, acc=acc)
            self.write_tensorboard_images(pred, label, 'Validation', epoch)
    
    def test(self):
        self.model.sad = False
        self.model.eval()
        for i, sample in enumerate(self.dataloaders['test'], 0):
            inputs = sample['data'].to(cfg.device)
            start_time = time.time()
            outputs = self.model(inputs)
            time_test_batch = time.time() - start_time
            pred = outputs.max(1)[1]
            
            plt.figure(1)
            plt.imshow(torch.squeeze(pred).to('cpu').numpy())
            plt.figure(2)
            plt.imshow(torch.squeeze(sample['label']).to('cpu').numpy())
            plt.show()
            print("Number: {}, Time: {}".format(i, time_test_batch))
    
    def test_weights(self):
        num_weights = os.listdir('weights')
        for i_weights in range(len(num_weights)):
            path_weight = os.path.join('weights', 'epoch_{:03}.pt'.format(i_weights))
            self.model = generate_model(train_mode=False, pretrained_weights=path_weight)
            self.model.to(cfg.device)
            self.model.eval()
            
            sample = next(iter(self.dataloaders['test']))
            inputs = sample['data'].to(cfg.device)
            outputs = self.model(inputs)
            pred = outputs.max(1)[1]

            img_pred = torch.squeeze(pred).to('cpu').numpy()
            img_label = torch.squeeze(sample['label']).to('cpu').numpy()
            
            fig, axs = plt.subplots(nrows=1, ncols=2)
            ax = axs[0]
            ax.imshow(img_pred)
            ax.set_title('prediction')
            ax = axs[1]
            ax.imshow(img_label)
            ax.set_title('label')
            fig.savefig(os.path.join('weights', 'Figure_{:03}.png'.format(i_weights)))
            plt.close()

    def create_test_videos(self):
        self.model = generate_model(False)
        self.model.to(cfg.device)
        self.model.eval()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_pred = cv2.VideoWriter('pred.avi', fourcc, 20.0, (cfg.input_image_w * 2 + 10, cfg.input_image_h))
        
        with open(os.path.join('lists', 'video_list.txt'), 'r') as f:
            img_list = f.readlines()

        test_dataloader = torch.utils.data.DataLoader(ImageGeneratorTest(\
            input_path=os.path.join('lists','video_list.txt'), num_images=len(img_list)))
        self.model.eval()

        for i, sample in enumerate(test_dataloader, 0):
                    
            inputs = sample['data'].to(cfg.device)

            outputs = self.model(inputs)
            pred = torch.squeeze(outputs.max(1, keepdim=True)[1])

            label_image = np.expand_dims(torch.squeeze(sample['label']).to('cpu').numpy(), 2).astype(np.uint8)
            label_image = np.concatenate((label_image*255, np.zeros_like(label_image, dtype=np.uint8), np.zeros_like(label_image, dtype=np.uint8)), axis=2)

            pred_image = np.expand_dims(pred.to('cpu').numpy(), 2).astype(np.uint8)
            pred_image = np.concatenate((pred_image*255, np.zeros_like(pred_image, dtype=np.uint8), np.zeros_like(pred_image, dtype=np.uint8)), axis=2)

            input_image = np.squeeze(sample['raw_image'].numpy().astype(np.uint8))

            label_image = cv2.add(input_image, label_image)
            pred_image = cv2.add(input_image, pred_image)
            out_image = np.concatenate((pred_image, np.zeros((cfg.input_image_h, 10, 3), dtype=np.uint8), label_image), axis=1)

            # plt.figure(1)
            # plt.imshow(pred_image)
            # plt.figure(2)
            # plt.imshow(label_image)
            # plt.show()

            video_pred.write(out_image)
            print("Number: {}".format(i))
        
        video_pred.release()

    def main(self):
        for epoch in range(cfg.initial_epoch, cfg.num_epochs):
            self.train(epoch)
            # self.val(epoch)
            self.scheduler.step(epoch=epoch)
            torch.save(self.model.state_dict(), os.path.join('weights', 'epoch_{:03}.pt'.format(epoch)))


if __name__ == '__main__':
    model = generate_model()
    trainer = TrainModel(model)
    # trainer.main()
    # trainer.test()
    # trainer.create_test_videos()
    trainer.test_weights()