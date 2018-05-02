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
