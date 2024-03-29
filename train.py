import matplotlib
matplotlib.use("Agg")

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
from matplotlib import pyplot as plt

def gan_train(log_interval, model, device, train_loader, g_optimizer, d_optimizer, epoch):
    cnn_encoder, rnn_decoder, net_D = model
    cnn_encoder.train()
    rnn_decoder.train()
    net_D.train()
    criterion = nn.BCELoss()

    real_label = 1
    fake_label = 0

    #G_losses = []
    #D_losses = []
    N_count = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        N_count += X.size(0)

        real_X = cnn_encoder(X)
        _, fake_X = rnn_decoder(real_X)

        #Train with real batch
        net_D.zero_grad()
        b_size = X.size(0)
        label = torch.full((b_size, ), real_label, device=device)
        output = net_D(real_X.detach()).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        #Train with fake batch
        label.fill_(fake_label)
        output = net_D(fake_X.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_fake1 = output.mean().item()

        errD = errD_real + errD_fake
        d_optimizer.step()

        # Update G network
        cnn_encoder.zero_grad()
        rnn_decoder.zero_grad()

        label.fill_(real_label)
        output = net_D(fake_X).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_fake2 = output.mean().item()

        g_optimizer.step()

        print('Train epoch: {} [{}/{} ({:.0f}%)]\tD_Loss:{:.4f}, G_loss:{:.4f}, D(x):{:.4f}, D(G(x)):{:.4f}/{:.4f}'.format(
                epoch+1, N_count, len(train_loader.dataset), 100.*(batch_idx+1)/len(train_loader), errD.item(), errG.item(), D_x,
                D_G_fake1, D_G_fake2))

def train(log_interval, model, device, train_loader, optimizer, epoch):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0
    print(len(train_loader))

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)
        optimizer.zero_grad()

        output = rnn_decoder(cnn_encoder(X)) #shape: (batch_size, num_of_classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        #to compute the training accuracy
        y_pred = torch.max(output, 1)[1]
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        #display the training information
        #if (batch_idx + 1) % log_interval == 0:
        print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f},Accu:{:.2f}%'.format(
                epoch+1, N_count, len(train_loader.dataset), 100.*(batch_idx+1)/len(train_loader), loss.item(), 100*step_score))


    return losses, scores

def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_path = opt.dataroot
    action_name_path = './UCF101actions.pkl'
    save_model_path = os.path.join(opt.checkpoints_dir, opt.name)
    use_gan = opt.use_gan

    #Encoder CNN
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512
    res_size = 224
    dropout_p = 0.0

    #Decoder RNN
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256

    #training parameters
    k = opt.classes
    epochs = opt.niter
    batch_size = opt.batch_size
    log_interval = opt.log_interval

    #select which frame to begin & end in videos
    begin_frame, end_frame, skip_frame = 1, 29, 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    params = {}
    #data loading params
    params = {'batch_size': batch_size, 'shuffle':True, 'num_workers':4, 'pin_memory': True} if use_cuda else {}

    #load UCF101 action names
    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)

    action_names.sort()

    le = LabelEncoder()
    le.fit(action_names)

    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)


    #Load training split dataset
    train_list = []
    train_actions = []

    for line in open('trainlist01.txt', 'r'):
        f_loc1 = line.find('v_')
        f_loc2 = line.find('_g')
        f_loc3 = line.find('.avi')

        suffix = line[f_loc2:f_loc3]

        #train_list.append(line[f_loc1:f_loc2])

        loc1 = line.find('/')
        train_actions.append(line[:loc1])
        train_file = 'v_' + line[:loc1] + suffix
        train_list.append(train_file)

    train_label = labels2cat(le, train_actions)

    #Load test split dataset
    test_list = []
    test_actions = []

    for line in open('testlist01.txt', 'r'):
        f_loc1 = line.find('v_')
        f_loc2 = line.find('_g')
        f_loc3 = line.find('.avi')

        suffix = line[f_loc2:f_loc3]

        #test_list.append(line[f_loc1:f_loc2])

        loc1 = line.find('/')
        test_actions.append(line[:loc1])
        test_file = 'v_' + line[:loc1] + suffix
        test_list.append(test_file)

    test_label = labels2cat(le, test_actions)

    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                           Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    #Create model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p,
                                cnn_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(cnn_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                            h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k, use_gan=use_gan).to(device)
    net_D = None
    if use_gan:
        net_D = Discriminator(input_dim=RNN_hidden_nodes, h_RNN_layers=RNN_hidden_layers, drop_p=dropout_p).to(device)

    #parallel model to multi-GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUS!")
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)
        if use_gan:
            net_D = nn.DataParallel(net_D)

        crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.fc2.parameters()) + \
                      list(cnn_encoder.module.fc3.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                      list(cnn_encoder.module.bn2.parameters()) + list(rnn_decoder.module.parameters())
        d_params = net_D.module.parameters()

    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.fc2.parameters()) +\
                      list(cnn_encoder.fc3.parameters()) + list(cnn_encoder.bn1.parameters()) +\
                      list(cnn_encoder.bn2.parameters()) + list(rnn_decoder.parameters())
        d_params = None
        if use_gan:
            d_params = net_D.parameters()

    optimizer = torch.optim.Adam(crnn_params, lr=opt.lr)
    g_optimizer = torch.optim.Adam(crnn_params, lr=1e-5, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(d_params, lr=2e-5, betas=(0.5, 0.9))

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    #GAN pretraining
    gan_iter = opt.gan_iter
    if use_gan:
        for epoch in range(gan_iter):
            gan_train(log_interval, [cnn_encoder, rnn_decoder, net_D], device, train_loader, g_optimizer, d_optimizer, epoch)

    # start training
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save('./' + opt.name + '_CRNN_epoch_training_losses.npy', A)
        np.save('./' + opt.name + '_CRNN_epoch_training_scores.npy', B)
        np.save('./' + opt.name + '_CRNN_epoch_test_loss.npy', C)
        np.save('./' + opt.name + '_CRNN_epoch_test_score.npy', D)

    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    title ='./' + opt.name +"_fig_UCF101_ResNetCRNN.png"
    plt.savefig(title, dpi=600)
    plt.close(fig)
