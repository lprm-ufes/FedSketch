import numpy as np
import random
import os
import pandas as pd
from torch.autograd import Variable
from tabnanny import verbose
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
import os.path
import csv
import json
from paho.mqtt import client as mqtt_client
import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import time
import uuid
import sqlite3
import datetime
import gc
from scipy.stats import norm
import math
import traceback
from sketch_utils import compress, decompress, get_params, set_params,differential_garantee_pytorch,delta_weights,get_random_hashfunc

import torch
from torch import nn, optim
import torch.nn.functional as nnf
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision

from torchvision import datasets, transforms


from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import models
import gc
import sys

from tsai.basics import *
from tsai.all import *
from tsai.inference import load_learner
import sklearn
import glob
import os

def set_seeds(seed=0):
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #random.seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'



os.environ['CUDA_VISIBLE_DEVICES'] = "0"
broker = 'localhost'
port = 1883
#port = 8083
topic_introduction = "introduction"
topic_choice= "choice"
topic_agg = "agg"


class FederatedClient(object):
    def __init__(self,ds_train,ds_test, model,criterion, compression, learning_rate, device,number_of_clients_selected,dataset,classes = [],clip_value_min = -1000, clip_value_max = 1000):
    
        self.client_id = str(uuid.uuid4())
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.device = device
        self.compression = compression
        self.length = 20
        self.vector_length = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.index_hash_function = [get_random_hashfunc(_max=int(compression*self.vector_length), seed=repr(j).encode()) for j in range(self.length)]
        
        self.number_of_clients_selected = number_of_clients_selected
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        self.dataset = dataset
        self.classes = classes



    def publish_introduction(self, client):

        msg = {
            'resp_topic' : self.client_id
            }
        result = client.publish(topic_introduction, json.dumps(msg))
        response = subscribe.simple(self.client_id, hostname=broker)
        self.total_round = int(json.loads(response.payload)['total_round'])
        self.global_seed = int(json.loads(response.payload)['global_seed'])
        self.percentile = int(json.loads(response.payload)['percentile'])
        self.desired_episilon = int(json.loads(response.payload)['episilon'])
        self.choice_mode = str(json.loads(response.payload)['choice_mode'])
        previous_seed = torch.initial_seed()
        torch.manual_seed(self.global_seed)
        if self.dataset == "MNIST" or self.dataset == "FAMNIST":
            self.model = models.LeNet5(10)
        elif self.dataset == "CIFAR10":
            self.model = models.LeNet5(10,3)
        torch.manual_seed(previous_seed)
        self.connection_db = sqlite3.connect(str(json.loads(response.payload)['connection_address']), timeout=120)
        
        self.round = 1
        print(self.total_round)

    def publish_choice(self, client):
        print("aqui publish choice")
        random_number = random.uniform(0,1)
        #client.subscribe(self.client_id)
        if self.round == 1 or self.choice_mode == "random":
            msg = {
                'resp_topic' : self.client_id,
                'random_number' : random_number
                }
        elif self.choice_mode == "accuracy":
           msg = {
                'resp_topic' : self.client_id,
                'similarity' : self.acc
                }
        elif self.choice_mode == "loss":
           msg = {
                'resp_topic' : self.client_id,
                'similarity' : self.loss
                }
        else:
           msg = {
                'resp_topic' : self.client_id,
                'similarity' : self.round_similarity                }
        print("publishing choice")
        result = client.publish(topic_choice, json.dumps(msg))
        print(result)
        response = subscribe.simple(self.client_id, hostname=broker)
        print(response.payload)
        self.chosen = int(json.loads(response.payload)['chosen'])
        self.round = int(json.loads(response.payload)['round'])
        print(self.chosen)
        print(self.round)

    def publish_agg(self,client):
        
        #client.subscribe(self.client_id)
        msg = {
                'resp_topic' : self.client_id,
            }
        if self.chosen == 1:
            old_weights = get_params(self.model) 
            self.model,self.loss = train_mlp(2,self.model,self.ds_train,self.criterion, self.learning_rate, self.device)
            self.acc = test_mlp(self.model,self.ds_test, self.device)
            print(f'ACC:{self.acc}')
            weights = get_params(self.model)
            delta = delta_weights(weights,old_weights)
            time0 = time.time()
            self.sketch = compress(delta, self.compression,self.length, self.desired_episilon, self.percentile,self.index_hash_function)
        
            differential_garantee_pytorch(delta,self.sketch,self.desired_episilon,self.percentile)
            print("\nSketching time (in minutes) =",(time.time()-time0)/60)
            sketch_list = [i.tolist() if type(i) != list else i for i in self.sketch]
            agg_data = {
                'Weights' : sketch_list,
            }
            
            ct = datetime.datetime.now()
            print(ct)
            self.cursor_db = self.connection_db.cursor()
            self.cursor_db.execute("""
                            INSERT INTO models (MODEL, ROUND, CLIENT_ID, Timestamp)
                            VALUES (?, ?, ?, ?)
                            """, (json.dumps(agg_data), self.round, self.client_id, ct))
            self.connection_db.commit()
            self.cursor_db.close()
            result = client.publish(topic_agg, json.dumps(msg))
            print(result)
            print("Pesos publicados")
            response = subscribe.simple(self.client_id, hostname=broker)

        else:
            result = client.publish(topic_agg, json.dumps(msg))
            weights = get_params(self.model)
            print("Pesos publicados")
            print(result)
            response = subscribe.simple(self.client_id, hostname=broker)
        
        print("Pesos recebidos")
        self.cursor_db = self.connection_db.cursor()
        self.cursor_db.execute("""
                            SELECT * FROM aggregate_models WHERE ROUND=? ORDER BY Timestamp DESC;
                            """, [self.round])

        linha = self.cursor_db.fetchmany(1)
        model_data = json.loads(linha[0][1])
        self.cursor_db.close()
        sketch_global = [np.asarray(i) for i in model_data['Weights']]
        #print(sketch_list)
        if self.choice_mode == "sketch":
            self.round_similarity = np.trace(cosine_similarity(sketch_global,self.sketch))/len(sketch_global)
        time0 = time.time()
        self.n_weights = decompress(weights,sketch_global, len(sketch_global),self.clip_value_min, self.clip_value_max,self.index_hash_function)
        print("\nDecompressing time (in minutes) =",(time.time()-time0)/60)
        self.global_learning_rate = float(json.loads(response.payload)['global_learning_rate'])
        
        set_params(self.model, self.n_weights,self.global_learning_rate)
        #print(n_weights)
        self.round = int(json.loads(response.payload)['round'])
        gc.collect()
        



    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    print("Connected to MQTT Broker!")
                else:
                    print("Failed to connect, return code %d\n", rc)
        # Set Connecting Client ID
        client = mqtt_client.Client(self.client_id)
        client.on_connect = on_connect
        client.connect(broker, port,keepalive=3600)
        return client

    def run(self):
        self.client = self.connect_mqtt()
        self.publish_introduction(self.client)
        while self.round <= self.total_round :
            print('aqui antes choice')
            self.publish_choice(self.client)
            print('aqui depois choice')
            print(self.round)
            #self.chosen = 1
            self.publish_agg(self.client)
            #self.round+=1
            print('aqui depois agg')
            gc.collect()
            if hasattr(self, 'acc'):
                acc_train = self.acc
            else:
                acc_train = np.nan
            if not hasattr(self, 'loss'):
                self.loss = np.nan
                
            self.acc = test_mlp( self.model,self.ds_test,self.device)
            print(f'ACC:{self.acc}')
            if self.classes != []:
                with open('Resultados/sketch_pytorch_'+ self.dataset  +'_non_iid_' + str(self.choice_mode) + '_' + str(self.number_of_clients_selected) + '.csv','a+') as fd:
                    result_csv_append = csv.writer(fd)
                    result_csv_append.writerow([self.acc,acc_train,self.loss,self.client_id,self.round-1,self.chosen,self.classes[0],self.classes[1]])
            else:
                with open('Resultados/sketch_pytorch_'+ self.dataset  +'_' + str(self.choice_mode) + '_' + str(self.number_of_clients_selected) + '.csv','a+') as fd:
                    result_csv_append = csv.writer(fd)
                    result_csv_append.writerow([self.acc,acc_train,self.loss,self.client_id,self.round-1,self.chosen])
            
                
                
            self.client.loop_start()
        self.client.loop_stop()
        self.client.disconnect()

def define_model():
  input_size = 784
  hidden_sizes = [128, 64]
  output_size = 10

  model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))
  return model

def train(train_args):

  train_loader = train_args[0]
  num_epochs = train_args[1]
  model = train_args[2]
  cost = train_args[3]
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  total_step = len(train_loader)
  for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(train_loader):
          images = images.to(device)
          labels = labels.to(device)

          #Forward pass
          outputs = model(images)
          loss = cost(outputs, labels)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if (i+1) % (total_step/num_epochs) == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
  return model, loss.item()


def train_mlp(num_epochs,model,train_loader,cost,learning_rate, device):

  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  total_step = len(train_loader)
  for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(train_loader):
          images = images.to(device)
          labels = labels.to(device)

          #Forward pass
          outputs = model(images)
          loss = cost(outputs, labels)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if (i+1) % (total_step/num_epochs) == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

  return model, loss.item()
        


def sample_random_dataloader(dataset,num_samples, batch_size):
  indices = torch.randperm(len(dataset))[:num_samples]

  sample = torch.utils.data.Subset(dataset, indices)
  #random_sampler = torch.utils.data.RandomSampler(dataset, num_samples=num_samples)
  dataloader = torch.utils.data.DataLoader(sample, batch_size=batch_size,shuffle=True,num_workers=2)
  return dataloader

def test_mlp(model,test_loader,device):
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
    return 100 * correct / total  


def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    train_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = True)


    test_dataset = torchvision.datasets.MNIST(root = './data',
                                          train = False,
                                          transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                          download=True)
    return train_dataset, test_dataset  
def load_fmnist():
    train_dataset = torchvision.datasets.FashionMNIST("./data", download=True, 
                                                      transform = transforms.Compose(
                                                          [transforms.Resize((32,32)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,))])
                                                  )
    test_dataset = torchvision.datasets.FashionMNIST("./data", download=True, train=False, 
                                                     transform = transforms.Compose([
                                                         transforms.Resize((32,32)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.5,), (0.5,))])
                                                  )
    return train_dataset, test_dataset
def load_cifar10(data_dir,
                batch_size,
                sample_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):

    # define transforms
    transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if test:
        dataset = datasets.CIFAR10(
          root=data_dir, train=False,
          download=True, transform=transform_test,
        )
        data_loader = sample_random_dataloader(dataset,int(sample_size/2), batch_size=batch_size)

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform_train,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform_train,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_loader = sample_random_dataloader(train_dataset,sample_size, batch_size=batch_size)
    val_loader = sample_random_dataloader(valid_dataset ,int(sample_size/2), batch_size=batch_size)

    return train_loader, val_loader

def load_motion_sensor(sub):
    class_map = {
        "dws": 0,
        "ups": 1,
        "wlk": 2,
        "std": 3,
        "sit": 4,
        "jog": 5
    }
    subject = "/sub_" + sub + ".csv"
    path = r'data/A_DeviceMotion_data/' # use your path
    paths = [x[0] for x in os.walk(path)]
    paths.remove(path)
    print(paths)
    X = []
    y = []
    for p in paths:
        df_raw = pd.read_csv(p+subject)
        df_raw["Class"] = p.split('/')[3].split('_')[0]

        X.append(df_raw.drop("Class",axis=1))
        y.append(df_raw["Class"])
    X = np.concatenate(X)
    y = np.concatenate(y)


    X = np.atleast_3d(X).transpose(0,2,1)
    labeler = ReLabeler(class_map)
    y = labeler(y)
    y.astype(int)
    splits = get_splits(y,
                    n_splits=1,
                    valid_size=0.3,
                    test_size=0.1,
                    shuffle=True,
                    balance=False,
                    stratify=True,
                    random_state=42,
                    show_plot=True,
                    verbose=True)
    tfms  = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)

    bs = 256
    dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs*2])
    return dls


if __name__ == "__main__":
    np.random.seed(0)
    #tf.random.set_seed(0)
    #random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cpu")
    
    compression = 0.00066666666#75x
    #compression = 0.0016666363#30x
    num_epochs = 1
    non_iid = False
    sample_size = int(sys.argv[1])
    number_of_clients_selected = int(sys.argv[2])
    dataset = str(sys.argv[3])

    if dataset == "MNIST" or dataset == "FAMNIST":
        batch_size = 64
        num_classes = 10
        learning_rate = 0.01
        if dataset == "MNIST":
            train_dataset,test_dataset = load_mnist()
        else:
            train_dataset,test_dataset = load_fmnist()
        if non_iid:
            learning_rate = 0.01
            batch_size = 10
            possible_classes = [1,2,3,4,5,6,7,8,9]
            num_classes = 2
            classes = random.sample(possible_classes, 2)
            idx = (train_dataset.targets==classes[0]) | (train_dataset.targets==classes[1])
            train_dataset.targets = train_dataset.targets[idx]
            train_dataset.data = train_dataset.data[idx]
        criterion = nn.CrossEntropyLoss()
        train_loader = sample_random_dataloader(train_dataset,sample_size, batch_size=batch_size)
        test_loader = sample_random_dataloader(test_dataset ,int(sample_size/2), batch_size=batch_size)
        model = models.LeNet5(num_classes)#.to(device)
    if dataset == "CIFAR10":

        num_classes = 10
        batch_size = 128
        learning_rate = 0.008
        train_loader,valid_loader = load_cifar10("/home/eduardo/emqx/data/",
                batch_size,
                sample_size,
                test=False)
        test_loader = load_cifar10("/home/eduardo/emqx/data/",
                batch_size,
                sample_size,
                test=True)
        
        print(len(train_loader.dataset))
        model = models.LeNet5(num_classes,3)
        ##model = models.LeNet5(num_classes)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
    if dataset == "MOTION":
        dls = load_motion_sensor()
        learning_rate = 1e-3
        archs = [
         (LSTM,    {'n_layers':1, 'bidirectional': True} ),
        ]
        for i, (arch, k) in enumerate(archs):

            model = create_model(arch, dls=dls, **k)

            learn = Learner(dls, model,  metrics=accuracy)

    if non_iid:
        fc = FederatedClient(train_loader,test_loader,model,criterion,compression,learning_rate, device,number_of_clients_selected,dataset,classes=classes)
    else:
        fc = FederatedClient(train_loader,test_loader,model,criterion,compression,learning_rate, device,number_of_clients_selected,dataset)
    try:
        fc.run()
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    finally:
        fc.connection_db.close()
        fc.client.disconnect()