import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batchnorm = nn.BatchNorm1d(2)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        # Move model to the specified device
        self.to(self.device)

    def forward(self, x):
        regu_sum = []  # Regularization list (empty for now, can add values if needed)
        x = x.to(self.device)  # Move input to the same device as the model
        x = self.batchnorm(x)
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim)).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out.contiguous().view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out, regu_sum
