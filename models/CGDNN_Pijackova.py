import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDropout(nn.Module):
    def __init__(self, p):
        super(GaussianDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:  # Only apply dropout during training
            noise = torch.randn_like(x) * self.p + 1
            return x * noise
        else:
            return x

class CGDNNModel(nn.Module):
    def __init__(self, num_classes, dataset_name):
        super(CGDNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=80, kernel_size=12)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.gru1 = nn.GRU(input_size=80, hidden_size=40, batch_first=True)
        self.gaussian_dropout1 = GaussianDropout(0.4)
        self.gru2 = nn.GRU(input_size=40, hidden_size=40, batch_first=True)
        self.gaussian_dropout2 = GaussianDropout(0.4)
        self.flatten = nn.Flatten()

        # Calculate the correct input dimension for the fully connected layer
        if dataset_name == '2016.10a' or dataset_name == '2016.10b' or dataset_name == 'migou_dataset_19.08':
            self.fc_input_dim = 40 * 58  # 58 comes from the sequence length after pooling
        elif dataset_name == '2018.01a':
            self.fc_input_dim = 40 * 506  # 506 comes from the sequence length after pooling

        self.fc = nn.Linear(self.fc_input_dim, num_classes)  # Fully connected layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        regu_sum = []  # Lista regularizacji (na razie pusta, w razie potrzeby można dodać odpowiednie wartości)
        # x shape: (batch_size, 2, 128)
        x = self.conv1(x)  # After conv1: (batch_size, 80, 117)
        x = self.relu(x)  # Apply ReLU activation
        x = self.maxpool1(x)  # After maxpool: (batch_size, 80, 58)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 58, 80) for GRU
        x, _ = self.gru1(x)
        x = self.gaussian_dropout1(x)
        x, _ = self.gru2(x)
        x = self.gaussian_dropout2(x)
        x = self.flatten(x)  # After flatten: (batch_size, 40*58)
        x = self.fc(x)  # Fully connected layer
        x = self.softmax(x)  # Softmax for output probabilities
        return x, regu_sum
