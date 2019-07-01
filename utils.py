import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch


"""-----------label conversion tool------------"""

def labels2cat(label_encoder, _list):
    return label_encoder.transform(_list)


def labels2onehot(OneHotEncoder, label_encoder, _list):
    return OneHotEncoder.transform(label_encoder.transform(_list))


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()



"""----------Conv+LSTM dataloader for UC-101 dataset--------------"""
"""In general, this dataloader can load sepecified frames in certain video folder"""
"""The dataset should be organized in data_path/video_folders/"""

class Dataset_CRNN(data.Dataset):

    def __init__(self, data_path, vid_folders, vid_labels, frames, transform=None):
        #Data loader initializer
        self.data_path = data_path
        self.vid_folders = vid_folders
        self.vid_labels = vid_labels
        self.transform = transform
        self.frames = frames


    def __len__(self):
        #Total number of samples
        return len(self.vid_folders)


    def read_images(self, path, selected_vid, use_transform=True):
        #Read images from the selected video clip folder, self.frames in total
        X = []

        for i in self.frames:
            image = Image.open(os.path.join(path, selected_vid, 'frame{:06d}.jpg'.format(i)))
            if use_transform:
                image = self.transform(image)
            X.append(image)

        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, index):
        #Generate one sample of the data
        #Select sample
        vid = self.vid_folders[index]

        #Load data
        X = self.read_images(self.data_path, vid)
        y = torch.LongTensor([self.vid_labels[index]])

        #print(X.shape)
        #print(y)

        return X, y



