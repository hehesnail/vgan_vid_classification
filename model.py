import torch
import torch.nn as nn
import torch.Functional as F
import torchvision.models as models


"""---------The pretrained ResNet-152 encoder---------"""
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, training=True, cnn_embed_dim=300):
        """Load ResNet152 and replace top fc layer"""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc.hidden2
        self.drop_p = drop_p
        self.training = training

        resnet = models.resnet152(pretrained = True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, cnn_embed_dim)

    def forward(self, x):
        cnn_embed_seq = []

        for t in range(x.size(1)):
            with torch.no_grad():
                x = self.resnet(x[:, t, :, :, :])
                x = x.view(x.size(0), -1)

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)
        #after transpose, the shape: (batch_size, time_step, cnn_embed_dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=1).transpose_(0, 1)

        return cnn_embed_seq

"""---------The LSTM decoder for action prediction----------"""
class DecoderRNN(nn.Module):
    def __init__(self, cnn_embed_dim=300, h_RNN_layers=3, h_RNN=512, h_FC_dim=128, drop_p=0.3, training=True, num_classes=101):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = cnn_embed_dim
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.training  = training
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
                    input_size = self.RNN_input_size,
                    hidden_size = self.h_RNN,
                    num_layers = self.h_RNN_layers,
                    batch_first = True)

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, num_classes)

        #TODO: Add scores output, i.e. weighted frame features

    def forward(self, x):

        self.lstm.flatten_parameters()
        RNN_out, (h_n, h_c) = self.lstm(x, None)
        #h_c:(n_layers, batch, hidden_size), h_c:(n_layers, batch, hidden_size)
        #RNN_out:(batch, seq_len, hidden_size), since, batch_first=True

        x = RNN_out[:,-1,:]
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        #TODO: Add weighted frame features

        return x

"""---------The GAN discriminator to discrminate real/fake frame features-----------"""
class Discriminator(nn.Module):
    def __init__(self, input_dim=128, h_RNN_layers=3, h_RNN=512, drop_p=0.3, training=True):
        super(Discriminator, self).__init__()
        #TODO

    def forward(self, x):
        #TODO

