import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image as Image
from enet import generate_model


class Configuration:

    def __init__(self):
        super().__init__()
        self.device = 'cuda:0'
        self.mean = [0.48274088, 0.456816,   0.42435675]
        self.std =  [0.35005408, 0.34594208, 0.32767905]
        self.pretrained_weights = 'weights/epoch_004.pt'
        self.input_file = 'lists/test_road.txt'
        # input parameters
        self.input_image_h = 256
        self.input_image_w = 512
        self.input_image_c = 3


cfg = Configuration()


class ImageGenerator(Dataset):

    def __init__(self, input_path):
        with open(input_path, 'r') as f:
            self.img_list = f.readlines()
        self.transforms_color = transforms.Compose([transforms.ToTensor(),\
            transforms.Normalize(cfg.mean, cfg.std)])

    def __len__(self):
        return len(self.img_list)-4

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        img = Image.open(img_path_list.strip())
        img = img.resize((cfg.input_image_w, cfg.input_image_h), Image.ANTIALIAS)
        raw_image = np.array(img)
        img = self.transforms_color(img)
        sample = {'data': img, 'raw_image': raw_image}
        return sample


def main():
    model = generate_model(train_mode=False, pretrained_weights=cfg.pretrained_weights)
    model.to(cfg.device)
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(ImageGenerator(input_path=cfg.input_file))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_pred = cv2.VideoWriter('pred.avi', fourcc, 20.0, (cfg.input_image_w, cfg.input_image_h))

    for i, sample in enumerate(test_dataloader, 0):
        inputs = sample['data'].to(cfg.device)
        
        outputs = model(inputs)
        pred = torch.squeeze(outputs.max(1, keepdim=True)[1])
        
        pred_image = np.expand_dims(pred.to('cpu').numpy(), 2).astype(np.uint8)
        pred_image = np.concatenate((pred_image*255, np.zeros_like(pred_image, dtype=np.uint8),\
            np.zeros_like(pred_image, dtype=np.uint8)), axis=2)
        input_image = np.squeeze(sample['raw_image'].numpy().astype(np.uint8))

        pred_image = cv2.add(input_image, pred_image)
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

        # plt.figure(1)
        # plt.imshow(pred_image)
        # plt.show()

        # cv2.imwrite("{}.png".format(i), pred_image)
        video_pred.write(pred_image)
        print("Number: {}".format(i))
    
    video_pred.release()
    print('Complite')


if __name__ == '__main__':
    main()
