import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd

train_df = pd.read_csv("data/kddcup99_train.csv",on_bad_lines='skip',header=None)
test_df = pd.read_csv("data/kddcup99_test.csv",on_bad_lines='skip',header=None)
attack_types = pd.read_table("data/trainning_attach_types", header=None,delim_whitespace=True)
attack_dict = {attack_types[0][i]+'.':1 for i in range(len(attack_types))}
attack_dict['normal.'] = 0
# attack_dict
train_df[41] = train_df[41].replace(attack_dict)
test_df[41] = test_df[41].replace(attack_dict)
map_dict = {}
for column in train_df.columns:
    if pd.api.types.is_object_dtype(train_df[column]):
        factorized_column, map = train_df[column].factorize()
        train_df[column] = factorized_column
        map_dict[column] = {map[i]:i for i in range(len(map))}

        _, map_t = test_df[column].factorize()
        # print([i for i in map_t if i not in map])
map_dict[2]['http_2784']=68
map_dict[2]['aol']=69
# map_dict
for column in test_df.columns:
    # print(column)
    if column in map_dict.keys():
        test_df[column] = test_df[column].replace(map_dict[column])
# test_df

class mynet(nn.Module):
    def __init__(self,  act_layer = nn.ReLU(), init_method = nn.init.normal_):
        super(mynet, self).__init__()
        self.act_layer = act_layer
        self.init_method = init_method
        self.net = nn.Sequential(
            # [b, 41] => [b, rank]
            nn.Linear(41, 36),
            self.act_layer,
            nn.Linear(36, 24),
            self.act_layer,
            nn.Linear(24, 12),
            self.act_layer,
            nn.Linear(12, 6),
            self.act_layer,
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.init_method(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        batchsize = x.size(0)
        x = x.view(batchsize, -1)
        x = self.net(x)
        # reshape
        x = x.view(batchsize, -1)
 
        return x

def train(act_layer = nn.ReLU(), init_method = nn.init.normal_, optimizer = optim.Adam):
    print(act_layer, init_method, optimizer)
    epochs = 10
    lr = 1e-3
    model = mynet(act_layer, init_method)
    criterion = nn.MSELoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    # print(model)

    for epoch in range(epochs):
        # print(epoch)
        total_loss = 0
        for batchidx, (x, y) in enumerate(my_dataset_loader):
            pred = model(x)
            loss=criterion(y.view(x.size(0),-1),pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        # if epoch % 10==0:
        #     print(total_loss.data.numpy())

        # print(total_loss.data.numpy())
    
    y_pred = model(torch.from_numpy(test_features.astype(np.float32))).detach().numpy()
    y_pred[y_pred<0.5] = 0
    y_pred[y_pred!=0] = 1
    
    precision, recall, F1_score, _ = precision_recall_fscore_support(test_labels, y_pred, average=None)
    acc = accuracy_score(test_labels, y_pred)
    print("precision: {} \nrecall: {} \nF1 score: {} \naccuracy: {}".format(precision, recall, F1_score, acc))

    return model

for act_layer in (nn.ReLU(), nn.Sigmoid()):
    for init_method in (nn.init.normal_, nn.init.kaiming_normal_):
        for optimizer in (optim.Adam, optim.SGD):
            train(act_layer, init_method, optimizer)