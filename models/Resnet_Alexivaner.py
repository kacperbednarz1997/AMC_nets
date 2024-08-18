import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the residual stack
class ResidualStack(nn.Module):
    def __init__(self,filters_in,  filters, seq, max_pool):
        super(ResidualStack, self).__init__()
        self.conv1 = nn.Conv2d(filters_in, filters, kernel_size=(1, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 2), padding='same')
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=(3, 2), padding='same')
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv4 = nn.Conv2d(filters, filters, kernel_size=(3, 2), padding='same')
        self.conv5 = nn.Conv2d(filters, filters, kernel_size=(3, 2), padding='same')
        self.bn3 = nn.BatchNorm2d(filters)
        self.max_pool = max_pool
        if self.max_pool:
            self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        # 1x1 Conv Linear
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Unit 1
        shortcut = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x + shortcut)

        # Residual Unit 2
        shortcut = x
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.bn3(x)
        x = F.relu(x + shortcut)

        # MaxPooling
        if self.max_pool:
            x = self.pool(x)

        return x

# Define the main model
class ResidualModel(nn.Module):
    def __init__(self, num_classes, dataset_name):
        super(ResidualModel, self).__init__()
        self.reshape = nn.Conv2d(1, 32, kernel_size=(1, 1))
        self.res_stack1 = ResidualStack(32,32, "ReStk1", False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1))
        self.res_stack2 = ResidualStack(32, 32, "ReStk2", True)
        self.res_stack3 = ResidualStack(32, 64, "ReStk3", True)
        self.res_stack4 = ResidualStack(64, 64, "ReStk4", True)
        self.flatten = nn.Flatten()

        # Calculate the correct input dimension for the fully connected layer
        if dataset_name == '2016.10a' or dataset_name == '2016.10b' or dataset_name == 'migou_dataset_19.08':
            self.fc_input_dim = 64 * 8
        elif dataset_name == '2018.01a':
            self.fc_input_dim = 64 * 64

        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        regu_sum = []  # Lista regularizacji (na razie pusta, w razie potrzeby można dodać odpowiednie wartości)
        b, iq, l = x.shape
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.reshape(x)
        x = self.res_stack1(x)
        x = self.max_pool1(x)
        x = self.res_stack2(x)
        x = self.res_stack3(x)
        x = self.res_stack4(x)
        x = self.flatten(x)
        x = F.selu(self.fc1(x))
        x = self.dropout1(x)
        x = F.selu(self.fc2(x))
        x = self.dropout2(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x, regu_sum

