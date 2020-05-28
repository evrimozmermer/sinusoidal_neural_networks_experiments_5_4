from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn import preprocessing
import numpy as np
from skimage.feature import match_template as norm_cross_corr
import json
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self,experiment = 3, formula = 3):
        super(Net, self).__init__()
        
        #experiment and formula
        self.experiment = experiment
        self.formula = formula
        
        #conv layers: (number of filters in, number of filters out, kernel size, stride)
        self.conv1 = nn.Conv2d(1, 8, 5, 2, bias = False) 
        self.conv1.weight = torch.nn.Parameter(self.random_uniform_1(self.conv1.weight,-0.3,0.3))
        if experiment in [2,4]:
            self.conv1.weight.requires_grad = False
        elif experiment in [1,3]:
            self.conv1.weight.requires_grad = True
        
        self.conv2 = nn.Conv2d(8, 64, 3, 2, bias = False)
        self.conv2.weight = torch.nn.Parameter(self.random_uniform_1(self.conv2.weight,-0.3,0.3))
        if experiment in [2,4]:
            self.conv2.weight.requires_grad = False
        elif experiment in [1,3]:
            self.conv2.weight.requires_grad = True
        
        #linear layers (number of in, number of out)
        self.fc1 = nn.Linear(1600, 64, bias=True)
        self.fc1.weight = torch.nn.Parameter(self.random_uniform_1(self.fc1.weight,-0.1,0.1))
        self.fc1.weight.requires_grad = True
        
        self.fc2 = nn.Linear(64, 10, bias=True)
        self.fc2.weight = torch.nn.Parameter(self.random_uniform_1(self.fc2.weight,-0.1,0.1))
        self.fc2.weight.requires_grad = True
        
        #sine layers weight adjustment, normal distribution, weight ranges = [min,max]
        weight_ranges = [-0.1,0.1]
        if self.experiment == 5:
            weight_ranges = [0.0,0.2]
        else:
            weight_ranges = [-0.1,0.1]
        self.weights_in_1 = torch.nn.Parameter(self.random_uniform_2(weight_ranges[0], weight_ranges[1],64,1600))
        self.weights_out_1 = torch.nn.Parameter(self.random_uniform_2(weight_ranges[0], weight_ranges[1],64,1600))
        self.weights_in_2 = torch.nn.Parameter(self.random_uniform_2(weight_ranges[0], weight_ranges[1],10,64))
        self.weights_out_2 = torch.nn.Parameter(self.random_uniform_2(weight_ranges[0], weight_ranges[1],10,64))
        self.pi = torch.tensor(np.pi,dtype = torch.float32)
        model.conv1.weight.permute(0,2,3,1)
        self.norm_filters = []
        self.norm_filters.append(
                self.random_uniform_1(model.conv1.weight.permute(0,2,3,1),0,1).detach().numpy()
                )
        self.norm_filters.append(
                self.random_uniform_1(model.conv2.weight.permute(0,2,3,1),0,1).detach().numpy()
                )
        
    @staticmethod
    def random_uniform_1(layer_weights,highest,lowest):
        random_tensor = torch.rand(layer_weights.shape)*(highest-lowest)+lowest
        return torch.autograd.Variable(random_tensor).requires_grad_(True)

    @staticmethod
    def random_uniform_2(lowest,highest,row,col):
        random_tensor = torch.rand(int(row),int(col))*(highest-lowest)+lowest
        return torch.autograd.Variable(random_tensor).requires_grad_(True)

    def forward(self, x):
        if self.experiment in [1,2]:
            x = self.conv1(x)
            x = self.conv2(x)
            
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.tanh(x)
            
        elif self.experiment in [3,4] and self.formula == 2:
            x = self.conv1(x)
            x = self.conv2(x)
            
            x = torch.flatten(x, 1)
            x = torch.sum(self.weights_out_1*torch.sin(self.pi*2*x*self.weights_in_1),axis = 1)
            x = torch.sum(self.weights_out_2*torch.sin(self.pi*2*x*self.weights_in_2),axis = 1)
            x = x.unsqueeze(0)

        elif self.experiment in [3,4] and self.formula == 3:
            x = self.conv1(x)
            x = self.conv2(x)
            
            x = torch.flatten(x, 1)
            x = torch.sum(self.weights_out_1*x*torch.sin(self.pi*2*x*self.weights_in_1),axis = 1)
            x = torch.sum(self.weights_out_2*x*torch.sin(self.pi*2*x*self.weights_in_2),axis = 1)
            x = x.unsqueeze(0)

        elif self.experiment == 5:
            #proprocess for norm_cross_corr
            x = x.permute(0,2,3,1).numpy()[0]

            #normalized cross correlation
            for cntf in range(len(self.norm_filters[0])):
                if cntf == 0:
                    processed = norm_cross_corr(x, self.norm_filters[0][cntf])
                elif cntf>0:
                    processed = np.dstack((processed,norm_cross_corr(x, self.norm_filters[0][cntf])))
            x = processed
            
            #downsample, works like slide = 2
            x = x[::2,::2,:]
            for cntf in range(len(self.norm_filters[1])):
                if cntf == 0:
                    processed = norm_cross_corr(x, self.norm_filters[1][cntf])
                elif cntf>0:
                    processed = np.dstack((processed,norm_cross_corr(x, self.norm_filters[1][cntf])))
            x = processed
            
            #downsample, works like slide = 2
            x = x[::2,::2,:]
            
            #preprocess for fully connected layers
            x = torch.from_numpy(x).float()
            
            x = torch.flatten(x).unsqueeze(0)
            x = torch.sum(self.weights_out_1*x*torch.sin(self.pi*2*x*self.weights_in_1),axis = 1)
            x = torch.sum(self.weights_out_2*x*torch.sin(self.pi*2*x*self.weights_in_2),axis = 1)
            x = x.unsqueeze(0)
        return x

def test(model, data_test, labels_test):
    model.eval()
    correct = 0
    with torch.no_grad():
        for cnt in range(10000):
            output = model(data_test[cnt].unsqueeze(0))
            if torch.argmax(output) == torch.argmax(labels_test[cnt]):
                correct += 1
    accuracy = 100. * correct / len(data_test)
    print('Test:\n____Accuracy: {}'.format(accuracy))
    
    return accuracy

df_train = pd.read_csv('../digit_dataset/mnist_train.csv').values
train_images = df_train[:,1:]/255+0.01
train_images = np.asarray(train_images)
train_labels = df_train[:,0].reshape((train_images.shape[0],1))
lb = preprocessing.LabelBinarizer()
lb.fit(train_labels)
train_labels_binary = lb.transform(train_labels)
train_labels_binary = np.asarray(train_labels_binary)

df_test = pd.read_csv('../digit_dataset/mnist_test.csv').values
test_images = df_test[:,1:]/255+0.01
test_labels = df_test[:,0]
test_labels_binary = lb.transform(test_labels)

data = torch.from_numpy((train_images).reshape(60000,28,28,1)).permute(0,3,1,2).float()
labels = torch.from_numpy(train_labels_binary).float()

data_test = torch.from_numpy((test_images).reshape(10000,28,28,1)).permute(0,3,1,2).float()
labels_test = torch.from_numpy(test_labels_binary).float()

experiment_dict = {}

for experiment in range(1,6,1):
    model = Net(experiment = experiment)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    experiment_dict["experiment_{}".format(experiment)] = []
    for cnt in range(60000):
        optimizer.zero_grad()
        output = model(data[cnt].unsqueeze(0))
        if experiment in [1,2]:
            loss = F.mse_loss(output,labels[cnt].unsqueeze(0))
            #loss = F.cross_entropy(output,ltorch.argmax(abels[cnt]).unsqueeze(0))
        elif experiment in [3,4,5]:
            loss = F.mse_loss(output,labels[cnt].unsqueeze(0))
        loss.backward()
        optimizer.step()
        
        lossV = output - labels[cnt].unsqueeze(0)
        print("Epoch: {}, Sample: {}, Loss: {}".format(experiment,cnt,torch.max(torch.abs(lossV))))
        print("Experiments:\n",experiment_dict)
        if (cnt+1)%1000 == 0:
            experiment_dict["experiment_{}".format(experiment)].append(test(model, data_test, labels_test))
            
#save dictionary as json
with open('experiment.json', 'w') as outfile:
    json.dump(experiment_dict, outfile)

#create dataframe to save as table in csv
rows = ["{}-{}".format(elm[0],elm[1]) for elm in zip(np.arange(0,60000,1000),np.arange(1000,61000,1000))]
columns = ["experiment_{}".format(elm) for elm in range(1,6,1)]

experiment_data = np.zeros((len(rows),len(columns)))
for cnt in range(1,6,1):
    experiment_data[:,cnt-1] = experiment_dict["experiment_{}".format(cnt)]
    
dataframe = pd.DataFrame(data = experiment_data, index = rows, columns = columns)
dataframe.to_excel("experiment.xlsx")

fig, ax = plt.subplots(figsize=(10,10))
ax.grid()
ax.plot(np.arange(0,60,1),dataframe.values[:,0], '-b', label='E.1')
ax.plot(np.arange(0,60,1),dataframe.values[:,1], '-g', label='E.2')
ax.plot(np.arange(0,60,1),dataframe.values[:,2], '-r', label='E.3')
ax.plot(np.arange(0,60,1),dataframe.values[:,3], '-c', label='E.4')
ax.plot(np.arange(0,60,1),dataframe.values[:,4], '-m', label='E.5')
leg = ax.legend();
