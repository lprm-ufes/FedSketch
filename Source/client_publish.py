from tabnanny import verbose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import sys
from typing import Dict
import numpy as np
import random
import os.path
from pathlib import Path
import csv
import json,codecs
from functools import reduce
#import psycopg2
from paho.mqtt import client as mqtt_client
import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import time
import uuid
import sqlite3
import datetime
import traceback

from sketch_utils import get_params

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
import json


broker = 'localhost'
port = 1883
#port = 8083
topic_introduction = "introduction"
topic_choice= "choice"
topic_agg = "agg"

class FederatedClient(object):
    def __init__(self,ds_train,ds_test, model,criterion,learning_rate, device,number_of_clients_selected,dataset,classes = [],clip_value_min = -1000, clip_value_max = 1000):
    
        self.client_id = str(uuid.uuid4())
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.device = device
        self.number_of_clients_selected = number_of_clients_selected
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        self.dataset = dataset
        self.classes = classes
        self.train_size = len(self.ds_train.dataset)
        self.round = 1

    def publish_introduction(self, client):
        #client.subscribe(client_id)
        msg = {
            'resp_topic' : self.client_id
            }
        #print(len(json.dumps(msg).encode('utf-8')))
        print('Publicando introduction')
        result = client.publish(topic_introduction, json.dumps(msg))
        response = subscribe.simple(self.client_id, hostname=broker)
        self.total_round = int(json.loads(response.payload)['total_round'])
        self.global_seed = int(json.loads(response.payload)['global_seed'])
        previous_seed = torch.initial_seed()
        torch.manual_seed(self.global_seed)
        if self.dataset == "MNIST" or self.dataset == "FAMNIST":
            self.model = models.LeNet5(10)#.to(self.device)
        elif self.dataset == "CIFAR10":
            self.model = models.LeNet5(10,3)#.to(self.device)
            #self.model = self.model.double()
        torch.manual_seed(previous_seed)
        weights = get_params(self.model)
        with open(self.client_id +'.json', 'w') as fp:
            json.dump({k: v.tolist() for k, v in weights.items()}, fp)
        self.connection_db = sqlite3.connect(str(json.loads(response.payload)['connection_address']), timeout=120)
        
        self.cursor_db = self.connection_db.cursor()
        self.cursor_db.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
        for linha in self.cursor_db.fetchall():
           print(linha)
        self.cursor_db.close()
        print(self.total_round )

    def publish_choice(self, client):

        random_number = random.uniform(0,1)
        msg = {
            'resp_topic' : self.client_id,
            'random_number' : random_number
            }
        result = client.publish(topic_choice, json.dumps(msg))
        print(result)
        response = subscribe.simple(self.client_id, hostname=broker)
        print(response.payload)
        self.chosen = int(json.loads(response.payload)['chosen'])
        self.round = int(json.loads(response.payload)['round'])
        print(self.chosen)
        print(self.round)

    def publish_agg(self,client):
        msg = {
                'resp_topic' : self.client_id,
            }
        if self.chosen == 1:
            if self.dataset == "MNIST" :
                self.model,self.loss = train_mlp(1,self.model,self.ds_train,self.criterion, self.learning_rate, self.device)
            elif self.dataset == "CIFAR10" or self.dataset == "FAMNIST":
                self.model,self.loss = train_mlp(2,self.model,self.ds_train,self.criterion, self.learning_rate, self.device)
            self.acc = test_mlp(self.model,self.ds_test, self.device)
            print(f'ACC:{self.acc}')
            weights = get_params(self.model)
            weights_list = [ v.tolist() for _, v in weights.items()]
            agg_data = {
                'Size' : self.train_size,
                'Weights' : weights_list
            }
            #print(len(json.dumps(msg).encode('utf-8')))
            
            ct = datetime.datetime.now()
            self.cursor_db = self.connection_db.cursor()
            self.cursor_db.execute("""
                            INSERT INTO models (MODEL, ROUND, Timestamp)
                            VALUES (?, ?, ?)
                            """, (json.dumps(agg_data ), self.round, ct))
            self.connection_db.commit()
            self.cursor_db.close()
            result = client.publish(topic_agg, json.dumps(msg))
            print("Pesos publicados")
            response = subscribe.simple(self.client_id, hostname=broker)

        else:
            result = client.publish(topic_agg, json.dumps(msg))
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
        weights = get_params(self.model)
        new_weights = [torch.Tensor(i) for i in model_data['Weights']]
        new_weights_dict = {}
        keys = list(weights.keys())
        #print(keys)
        for k in range(len(keys)):
            new_weights_dict[keys[k]] = new_weights[k]
        set_params(self.model,new_weights_dict)
        #self.model.load_state_dict(new_weights_dict,strict=False)

        #print(get_params(self.model))


    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    print("Connected to MQTT Broker!")
                else:
                    print("Failed to connect, return code %d\n", rc)
        # Set Connecting Client ID
        client = mqtt_client.Client(self.client_id)
        #client = mqtt_client.Client(client_paho_id,transport='websockets')
        #client.username_pw_set(username, password)
        client.on_connect = on_connect
        client.connect(broker, port,keepalive=3600)
        return client

    def run(self):
        self.client = self.connect_mqtt()
        self.publish_introduction(self.client)
        while self.round < self.total_round :
            print('aqui antes choice')
            self.publish_choice(self.client)
            print('aqui depois choice')
            print(self.round)
            #self.chosen = 1
            self.publish_agg(self.client)
            #self.round+=1
            print('aqui depois agg')
            if hasattr(self, 'acc'):
                acc_train = self.acc
            else:
                acc_train = np.nan
            if not hasattr(self, 'loss'):
                self.loss = np.nan
                
            self.acc = test_mlp( self.model,self.ds_test,self.device)
            print(f'ACC:{self.acc}')
            if self.classes != []:
                with open('Resultados/pytorch_'+ self.dataset  +'_non_iid_random_' + str(self.number_of_clients_selected) + '.csv','a+') as fd:
                    result_csv_append = csv.writer(fd)
                    result_csv_append.writerow([self.acc,self.client_id,self.round,self.chosen,self.classes[0],self.classes[1]])
            else:
                with open('Resultados/pytorch_'+ self.dataset  +'_random_' + str(self.number_of_clients_selected) + '.csv','a+') as fd:
                    result_csv_append = csv.writer(fd)
                    result_csv_append.writerow([self.acc,acc_train,self.loss,self.client_id,self.round,self.chosen])
            
            self.client.loop_start()
        self.client.disconnect()

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
        

#def sample_random_dataloader(dataset,num_samples, batch_size):
#  indices = torch.randperm(len(dataset))[:num_samples]
#  d = torch.utils.data.Subset(dataset, indices)
#  dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
#  return dataloader

def set_params(model, data):
  for name, param in model.named_parameters():
    if param.requires_grad:
        #print(param.data)
        #print(data)
        param.data = data[name]

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

if __name__ == "__main__":
    np.random.seed(0)
    #tf.random.set_seed(0)
    #random.seed(0)
    #torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cpu")
    
    num_epochs = 2
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
            possible_classes = [1,2,3,4,5,6,7,8,9]
            num_classes = 2
            classes = random.sample(possible_classes, 2)
            idx = (train_dataset.targets==classes[0]) | (train_dataset.targets==classes[1])
            train_dataset.targets = train_dataset.targets[idx]
            train_dataset.data = train_dataset.data[idx]
            print(np.unique(train_dataset.targets))
            print(train_dataset)
            idx = (test_dataset.targets==classes[0]) | (test_dataset.targets==classes[1])
            test_dataset.targets = test_dataset.targets[idx]
            test_dataset.data = test_dataset.data[idx]
            print(np.unique(test_dataset.targets))
            print(test_dataset)
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

    if non_iid:
        fc = FederatedClient(train_loader,test_loader,model,criterion,learning_rate, device,number_of_clients_selected,dataset,classes=classes)
    else:
        fc = FederatedClient(train_loader,test_loader,model,criterion,learning_rate, device,number_of_clients_selected,dataset)
    try:
        fc.run()
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    finally:
        fc.connection_db.close()
        fc.client.disconnect()