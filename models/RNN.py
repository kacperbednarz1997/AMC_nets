import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batchnorm = nn.BatchNorm1d(2)
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.to(self.device)  # Move the model to the appropriate device

    def forward(self, x):
        regu_sum = []  # Lista regularizacji (na razie pusta, w razie potrzeby można dodać odpowiednie wartości)
        x = self.batchnorm(x).to(self.device)  # Ensure x is on the same device
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim)).to(self.device)  # Ensure h0 is on the same device
        out, hn = self.rnn(x, h0)
        out = out.contiguous().view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out, regu_sum

