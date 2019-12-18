import os
import torch


class Config:
    '''Configuration file'''
    def __init__(self):
        super().__init__()
        #image_parameters
        self.input_image_h = 256
        self.input_image_w = 512
        self.input_image_c = 3
        #model_parameters
        self.device = 'cuda:0'
        self.pretrained_weights = None
        self.initial_epoch = 0 if self.pretrained_weights == None\
            else int(self.pretrained_weights[-6:-3]) + 1
        self.class_weight = [0.01, 0.05]
        self.num_workers = 4
        self.learning_rate = 0.01
        self.n_classes = 2
        self.mean = [0.48274088, 0.456816,   0.42435675]
        self.std =  [0.35005408, 0.34594208, 0.32767905]
        self.batch_size = 8
        self.num_epochs = 100
        self.lambda_dice = 2.
        self.lambda_distilation = 50.
        self.lambda_crossentropy = 1.5
        self.lambda_clip_weights = 0.9
        #layers_parameters
        self.drop_rate = 0.4
        #create_folders
        if not os.path.exists('weights'):
            os.makedirs('weights')
        if not os.path.exists('lists'):
            os.makedirs('lists')
        #file_parameters
        if os.path.exists(os.path.join('lists', 'train.txt')) and\
            os.path.exists(os.path.join('lists', 'val.txt')) and\
            os.path.exists(os.path.join('lists', 'test.txt')):
            self.train_dir = os.path.join('lists', 'train.txt')
            self.val_dir = os.path.join('lists', 'val.txt')
            self.test_dir = os.path.join('lists', 'test.txt')
            with open(self.train_dir, "r") as f:
                self.num_train_images = len(f.readlines())
            with open(self.val_dir, "r") as f:
                self.num_val_images = len(f.readlines())
            with open(self.test_dir, "r") as f:
                self.num_test_images = len(f.readlines())
        else:
            print('train.txt or val.txt or test.txt don`t\
                exist, create it using utils.create_train_val_test_lists')


                