import torch

# set config
INPUT_DIM = 2
TEMP_UNIT_NUM = 50
OUTPUT_DIM = 2
DROPOUT = 0.5

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=DROPOUT):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dims[0], batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm1.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm1.weight_hh_l0)

        self.lstm2 = torch.nn.LSTM(input_size=hidden_dims[0], hidden_size=hidden_dims[1], batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm2.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm2.weight_hh_l0)
        
        self.fc1 = torch.nn.Linear(hidden_dims[1], hidden_dims[2])
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout_rate)

        self.fc2 = torch.nn.Linear(hidden_dims[2], hidden_dims[3])
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout_rate)
        
        self.fc_last = torch.nn.Linear(hidden_dims[-1], output_dim)
        torch.nn.init.kaiming_normal_(self.fc_last.weight)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc_last(x)
        x = self.softmax(x)

        return x

# for hyperparameter search
class LSTM_layer4(torch.nn.Module):
    def __init__(self, input_dim=INPUT_DIM, unit_num1=TEMP_UNIT_NUM, unit_num2=TEMP_UNIT_NUM, unit_num3=TEMP_UNIT_NUM, unit_num4=TEMP_UNIT_NUM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size=input_dim, hidden_size=unit_num1, batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm1.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm1.weight_hh_l0)

        self.lstm2 = torch.nn.LSTM(input_size=unit_num1, hidden_size=unit_num2, batch_first=True)
        torch.nn.init.xavier_normal_(self.lstm2.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm2.weight_hh_l0)
        
        self.fc1 = torch.nn.Linear(unit_num2, unit_num3)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout_rate)

        self.fc2 = torch.nn.Linear(unit_num3, unit_num4)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout_rate)
        
        self.fc_last = torch.nn.Linear(unit_num4, output_dim)
        torch.nn.init.kaiming_normal_(self.fc_last.weight)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc_last(x)
        x = self.softmax(x)

        return x 