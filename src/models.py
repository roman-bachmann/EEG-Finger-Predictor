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


    """

    def __init__(self, time_steps, kernel_sizes=[3,5,7], conv_channels=[28,64,1], dropout=0):
        super(CNN_Model, self).__init__()
        self.conv = [[] for i in range(len(kernel_sizes))]

        for idx, kernel_size in enumerate(kernel_sizes):
            for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):
                conv_i = torch.nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         padding=kernel_size//2)
                self.conv[idx].append(conv_i)

        conv_concat_size = time_steps * len(kernel_sizes)

        self.fc = torch.nn.Linear(conv_concat_size, 2)

    def forward(self, x):
        conv_out = [x.permute(0,2,1) for i in range(len(self.conv))]

        for idx, conv_pipeline in enumerate(self.conv):
            for conv_i in conv_pipeline:
                conv_out[idx] = F.relu(conv_i(conv_out[idx]))
            conv_out[idx] = conv_out[idx].view(conv_out[idx].shape[0], -1)

        out = torch.cat(conv_out, 1)

        out = self.fc(out)
        return out


#seq = train_input.shape[1] # 500
#input_size = train_input.shape[2] #28

class Net(torch.nn.Module):
    def __init__(self, n_time_steps, n_features, batch_size, lstm_hidden_size=32,
                 lstm_layers=1, dropout=0.2, conv3_layers=0, conv5_layers=0,
                 conv7_layers=0, conv_channels=1, fc_layers=0, fc_size=256):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.conv3_layers = conv3_layers
        self.conv5_layers = conv5_layers
        self.conv7_layers = conv7_layers
        self.fc_layers = fc_layers
        self.concat_size = 0

        # LSTM input size is: (batch_size, n_time_steps, n_features)
        # LSTM output size is: (batch_size, n_time_steps, lstm_hidden_size)
        if lstm_layers:
            self.concat_size += lstm_hidden_size
            self.lstm = torch.nn.LSTM(input_size=n_features,
                                      hidden_size=lstm_hidden_size,
                                      num_layers=lstm_layers,
                                      batch_first=True,
                                      dropout=dropout)

        # Conv input size is: (batch_size, n_features, n_time_steps)
        # Conv output size is: (batch_size, conv_channels or 1, n_time_steps)
        if conv3_layers:
            self.concat_size += n_time_steps
            self.conv3a = torch.nn.Conv1d(in_channels=n_features,
                                          out_channels=conv_channels,
                                          kernel_size=3,
                                          padding=1)
        if conv3_layers > 1:
            self.conv3b = torch.nn.Conv1d(in_channels=conv_channels,
                                          out_channels=1,
                                          kernel_size=3,
                                          padding=1)

        if conv5_layers:
            self.concat_size += n_time_steps
            self.conv5a = torch.nn.Conv1d(in_channels=n_features,
                                          out_channels=conv_channels,
                                          kernel_size=5,
                                          padding=2)
        if conv5_layers > 1:
            self.conv5b = torch.nn.Conv1d(in_channels=conv_channels,
                                          out_channels=1,
                                          kernel_size=5,
                                          padding=2)

        if conv7_layers:
            self.concat_size += n_time_steps
            self.conv7a = torch.nn.Conv1d(in_channels=n_features,
                                          out_channels=conv_channels,
                                          kernel_size=7,
                                          padding=3)
        if conv7_layers > 1:
            self.conv7b = torch.nn.Conv1d(in_channels=conv_channels,
                                          out_channels=1,
                                          kernel_size=7,
                                          padding=3)

        if fc_layers:
            self.fca = torch.nn.Linear(self.concat_size, n_hidden)
            self.fcb = torch.nn.Linear(n_hidden, 2)
        else:
            self.fcb = torch.nn.Linear(self.concat_size, 2)

    def forward(self, x):
        concat_list = []

        if self.lstm_layers:
            lstm_out = self.lstm(x)[0][:,-1,:] # take only last output of LSTM (many-to-one RNN)
            lstm_out = lstm_out.view(lstm_out.shape[0], -1) # flatten to (batch, lstm_hidden_size)
            concat_list.append(lstm_out)

        x = x.permute(0,2,1)

        if self.conv3_layers:
            c3 = F.relu(self.conv3a(x))
            for i in range(self.conv3_layers - 1):
                c3 = F.relu(self.conv3b(c3))
            c3 = c3.view(c3.shape[0], -1)
            concat_list.append(c3)

        if self.conv5_layers:
            c5 = F.relu(self.conv5a(x))
            for i in range(self.conv5_layers - 1):
                c5 = F.relu(self.conv5b(c5))
            c5 = c5.view(c5.shape[0], -1)
            concat_list.append(c5)

        if self.conv7_layers:
            c7 = F.relu(self.conv7a(x))
            for i in range(self.conv7_layers - 1):
                c7 = F.relu(self.conv7b(c7))
            c7 = c7.view(c7.shape[0], -1)
            concat_list.append(c7)

        out = torch.cat(concat_list, 1)

        for i in range(self.fc_layers):
            out = self.fca(out)
        out = self.fcb(out)
        return out

    def init_hidden(self, batch_size):
        # Initialising the hidden layer
        return (Variable(torch.zeros(self.batch_size, self.lstm_layers, self.lstm_hidden_size)),
                Variable(torch.zeros(self.batch_size, self.lstm_layers, self.lstm_hidden_size)))
