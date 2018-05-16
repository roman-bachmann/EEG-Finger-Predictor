import torch
from torch.autograd import Variable
from torch.nn import functional as F


class LSTM_Model(torch.nn.Module):
    """
    Args:

        input_size (int): Length of input vector for each time step

        hidden_size (int, optional): Size of hidden LSTM state

        num_layers (int, optional): Number of stacked LSTM modules

        dropout (float, optional): Dropout value to use inside LSTM and between
            LSTM layer and fully connected layer.

    """

    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM input dimension is: (batch_size, time_steps, num_features)
        # LSTM output dimension is: (batch_size, time_steps, hidden_size)
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x, hidden):
        self.lstm.flatten_parameters() # For deep copy
        x = self.lstm(x, hidden)[0][:, -1, :] # Take only last output of LSTM (many-to-one RNN)
        x = x.view(x.shape[0], -1) # Flatten to (batch_size, hidden_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def init_hidden(self, batch_size):
        # Initializing the hidden layer.
        # Call every mini-batch, since nn.LSTM does not reset it itself.
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            return (Variable(h_0.cuda()), Variable(c_0.cuda()))
        else:
            return (Variable(h_0), Variable(c_0))

class CNN_Model(torch.nn.Module):
    """
    Args:

        time_steps (int): Length of entire sequence.

        kernel_sizes (List[int], optional): List of parallel convolution kernel sizes.

        conv_channels (List[int], optional): List of in and out dimensions for all
            convolutional layers. First element has to be number of input features.

        dense_size (int, optional): Size of fully connected layer after convolutions.

        dropout (float, optional): Dropout probability to use for conv and fully
            connected layers.
    """

    def __init__(self, time_steps, kernel_sizes=[3,5,7], conv_channels=[28,64,1],
                 dense_size=128, dropout=0):
        super(CNN_Model, self).__init__()
        self.dropout = dropout
        self.conv = [[] for i in range(len(kernel_sizes))]

        for idx, kernel_size in enumerate(kernel_sizes):
            for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):
                conv_i = torch.nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         padding=kernel_size//2)
                self.conv[idx].append(conv_i)
                self.add_module('conv_K{}_O{}'.format(kernel_size, out_channels), conv_i)

        conv_concat_size = time_steps * len(kernel_sizes)

        self.fc1 = torch.nn.Linear(conv_concat_size, dense_size)
        self.fc2 = torch.nn.Linear(dense_size, 2)

    def forward(self, x):
        conv_out = [x.clone().permute(0,2,1) for i in range(len(self.conv))]

        for idx, conv_pipeline in enumerate(self.conv):
            for conv_i in conv_pipeline:
                conv_out[idx] = F.relu(conv_i(conv_out[idx]))
                conv_out[idx] = F.dropout(conv_out[idx], training=self.training)
            conv_out[idx] = conv_out[idx].view(conv_out[idx].shape[0], -1)

        out = torch.cat(conv_out, 1)

        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return out
