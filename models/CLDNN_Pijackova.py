import torch
import torch.nn as nn

class CLDNNModel(nn.Module):
    def __init__(self, num_classes, dataset_name):
        super(CLDNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.4)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.4)

        # Calculate the correct input size for the Linear layer
        self.flatten = nn.Flatten()
        if dataset_name == '2016.10a' or dataset_name == '2016.10b' or dataset_name == 'migou_dataset_19.08':
            self.fc_input_dim = 60 * 64  # 60 (sequence length after pooling) * 64 (hidden size of LSTM)
        elif dataset_name == '2018.01a':
            self.fc_input_dim = 508 * 64  # 508 (sequence length after pooling) * 64 (hidden size of LSTM)

        self.fc = nn.Linear(self.fc_input_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        regu_sum = []  # Lista regularizacji (na razie pusta, w razie potrzeby można dodać odpowiednie wartości)
        # x shape: (batch_size, 2, 128)
        x = self.conv1(x)  # After conv1: (batch_size, 64, 121)
        x = self.relu(x)  # Apply ReLU activation
        x = self.maxpool1(x)  # After maxpool: (batch_size, 64, 60)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 60, 64) for LSTM
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.flatten(x)  # After flatten: (batch_size, 60*64)
        x = self.fc(x)  # Fully connected layer
        x = self.softmax(x)  # Softmax for output probabilities
        return x, regu_sum


