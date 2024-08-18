import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes, dataset_name):
        super(CNNModel, self).__init__()
        self.zero_pad = nn.ConstantPad1d(4, 0)  # Padding na wysokość dla Conv1d
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=50, kernel_size=8)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=8)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=4)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.6)
        self.flatten = nn.Flatten()

        # Zamiast twardego kodowania wymiarów wejściowych, obliczymy je dynamicznie
        self._to_linear = None
        self.convs = nn.Sequential(
            self.zero_pad,
            nn.Conv1d(in_channels=2, out_channels=50, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.6)
        )
        if dataset_name == '2016.10a' or dataset_name == '2016.10b' or dataset_name == 'migou_dataset_19.08':
            self._initialize_to_linear_layer((2, 128))
        elif dataset_name == '2018.01a':
            self._initialize_to_linear_layer((2, 1024))

        self.fc1 = nn.Linear(self._to_linear, 70)
        self.fc2 = nn.Linear(70, num_classes)

    def _initialize_to_linear_layer(self, shape):
        # Przekazywanie przykładowego tensora przez warstwy konwolucyjne, aby obliczyć wymiar wejściowy do fc1
        x = torch.randn(1, *shape)
        x = self.convs(x)
        self._to_linear = x.shape[1] * x.shape[2]

    def forward(self, x):
        regu_sum = []  # Lista regularizacji (na razie pusta, w razie potrzeby można dodać odpowiednie wartości)
        x = self.convs(x)
        x = self.flatten(x)
        x = F.selu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x, regu_sum

