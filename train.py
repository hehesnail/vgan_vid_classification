import os
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import *
from model import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from options import TrainOptions
import pickle

def train(log_interval, model, device, train_loader, optimizer, epoch):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)
        optimizer.zero_grad()

        output = rnn.decoder(cnn.encoder(X)) #shape: (batch_size, num_of_classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        #to compute the training accuracy
        y_pred = torch.max(output, 1)[1]
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        #display the training information
        if (batch_idx + 1) % log_interval



if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_path = opt.dataroot
    action_name_path = './UCF101actions.pkl'
    save_model_path = opt.checkpoints

    #Encoder CNN
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512
    res_size = 224
    dropout_p = 0.0

    #Decoder RNN
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_fc_dim = 256

    #training parameters
    k = opt.classes
    epochs = opt.niter
    batch_size = opt.batch_size
    log_interval = opt.log_interval

    #select which frame to begin & end in videos
    begin_frame, end_frame, skip_frame = 1, 29, 1

    use_cuda = torch.cuda.is_avaiable()
    #device = torch.device("cuda" if use_cuda else "cpu")

    #data loading params
    params = {'batch_size': batch_size, 'shuffle':False, 'num_workers':4, 'pin_memory: True'} if use_cuda else {}


