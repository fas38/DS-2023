import torch
from torch import nn

# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=2, kernel_size=3, stride=1)
        if num_layers == 1:
            self.lstm = nn.LSTM(2, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        else:
            self.lstm = nn.LSTM(2, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        
        _, (hidden_n, _) = self.lstm(x)
        
        out = self.fc(hidden_n[-1, :, :])

        return out 