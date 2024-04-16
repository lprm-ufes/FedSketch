import numpy as np
import pandas as pd
import itertools
import mmh3
import random
import statistics
import math
import os
import sklearn
from scipy.spatial.distance import cosine
from scipy.stats import norm
import collections
from tqdm import tqdm, trange
import pickle
from tsai.basics import *
from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import sys
import gc
import tensorflow as tf
import numpy as np
from functools import reduce
import random
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD 
from paho.mqtt import client as mqtt_client
from sketch_utils import compress, get_params
import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
from scipy.spatial.distance import cdist
import time
import json
import logging
import sqlite3
import datetime
import traceback
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import math
import sketch_utils
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from pympler import asizeof

import gc
import sys

def set_seeds(seed=0):
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def step_decay(epoch,initial_lrate):
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
logger = logging.getLogger()
logger.setLevel(logging.INFO) 

broker = 'localhost'
port = 1883
#port = 8083
topic_introduction = "introduction"
topic_choice= "choice"
topic_agg = "agg"
client_paho_id = f'python-mqtt-{random.randint(0, 1000)}'


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def aggregate(weights,num_clientes):
    """Compute weighted average."""
    print(num_clientes)
    with open('output.txt', 'a+') as f:
                        f.write('divisor aggregate: ' + str(num_clientes) + " ")
                        f.write('\n')
    weights_prime = [
        reduce(np.add, layer_updates) / num_clientes
        for layer_updates in zip(*weights)
    ]
    return weights_prime

class FederatedServer(object):

    def __init__(self, min_clients, number_choice_clients, max_rounds, global_leaning_rate, use_number_choice=True ,chance=1,decay_factor = 0.001,choice_mode = "random",weight_decay = False,order_comparison = "increasing"):
        self.round = 0
        self.min_clients = min_clients
        self.number_choice_clients = number_choice_clients
        self.max_rounds = max_rounds
        self.agg_clients = []
        self.choice_clients = []
        self.use_number_choice= use_number_choice
        self.chance = chance
        self.decay_factor = decay_factor
        self.choice_mode = choice_mode
        self.desired_episilon = 1
        self.percentile = 90
        self.length = 20
        self.compression = 0.0066666666
        self.global_learning_rate = global_leaning_rate
        self.global_seed = 0
        self.weight_decay = weight_decay
        self.order_comparison = order_comparison

        self.connection_address = 'agg.db'
        self.connection_db = sqlite3.connect(self.connection_address, timeout=1200)
        self.cursor_db = self.connection_db.cursor()
        self.cursor_db.execute("""
                        CREATE TABLE IF NOT EXISTS models (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        MODEL TEXT NOT NULL,
                        ROUND INTEGER NOT NULL,
                        CLIENT_ID TEXT NOT NULL,
                        Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        );
                        """)
        self.cursor_db.execute("""
                        CREATE TABLE IF NOT EXISTS aggregate_models (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        MODEL TEXT NOT NULL,
                        ROUND INTEGER NOT NULL,
                        Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        );
                        """)
        self.cursor_db.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
        for linha in self.cursor_db.fetchall():
           print(linha)
        self.cursor_db.close()

    def client_selection_random(self):
        if self.use_number_choice:
                    chosen_clients = random.sample(self.choice_clients, self.number_choice_clients)
                    for c in self.choice_clients:
                        
                        topic = c['resp_topic']
                        print(topic)
                        if c in chosen_clients :
                            chosen = 1
                        else:
                            chosen = 0
                        msg = {
                            'chosen' : chosen, 
                            'round' : self.round
                        }
                        publish.single( topic, json.dumps(msg), hostname=broker)

        else:
            for c in self.choice_clients:
                topic = c['resp_topic']
                if float(c['random_number']) <= self.chance:
                    chosen = 1
                else:
                    chosen = 0
                msg = {
                    'chosen' : chosen, 
                    'round' : self.round
                }
                publish.single( topic, json.dumps(msg), hostname=broker)
    def client_selection_sketch(self):
         self.number_clients_non_random = 0
         if self.order_comparison == "decreasing":
            self.similarity_order = sorted(
                                                   self.choice_clients,
                                                   key=lambda x: x['similarity'],
                                                   reverse=True

                                               )
         else:
             self.similarity_order = sorted(
                                                   self.choice_clients,
                                                   key=lambda x: x['similarity']

                                               )
         self.mean_similarity = sum(item['similarity'] for item in self.similarity_order)/len(self.similarity_order)
         self.C = math.ceil(len(self.similarity_order)*((1-self.decay_factor)**self.round))
         for i in range(len(self.similarity_order)):
                 topic = self.similarity_order[i]['resp_topic']
                 if self.order_comparison == "decreasing":
                     order_test = self.similarity_order[i]['similarity'] >= self.mean_similarity
                 else:
                     order_test = self.similarity_order[i]['similarity'] <= self.mean_similarity
                 if i <= self.C and order_test:          
                    print(topic)
                    chosen = 1
                    self.number_clients_non_random+=1
                    msg = {
                        'chosen' : chosen, 
                        'round' : self.round
                    }
                    publish.single(topic, json.dumps(msg), hostname=broker)
                 else:
                    print(topic)
                    chosen = 0
                    msg = {
                        'chosen' : chosen, 
                        'round' : self.round
                    }
                    publish.single(topic, json.dumps(msg), hostname=broker)

                     


    def on_message_choice_client_selection(self, client, userdata, msg):
        self.choice_clients.append(json.loads(msg.payload))
        print("Choice")
        print(len(self.choice_clients) >= self.min_clients)
        if len(self.choice_clients) >= self.min_clients:
            if self.round == 1:
                time.sleep(0.5)
                self.client_selection_random()
                self.last_choice = self.choice_clients
            else:
                time.sleep(0.5)
                self.client_selection_sketch()
            if self.round > self.max_rounds:
                self.round = 1
            self.choice_clients = []


    def on_message_introduction(self, client, userdata, msg):
        client_msg = json.loads(msg.payload)
        time.sleep(0.5)
        
        topic = client_msg['resp_topic']
        msg = {
                'total_round' : self.max_rounds,
                'global_seed' : self.global_seed,
                'percentile': self.percentile,
                'episilon': self.desired_episilon,
                'choice_mode' : self.choice_mode,
                'connection_address' : self.connection_address
            }
        self.round = 1
        publish.single( topic, json.dumps(msg), hostname=broker)
    def on_message_choice_random(self, client, userdata, msg):
            self.choice_clients.append(json.loads(msg.payload))
            print("Choice")
            print(len(self.choice_clients) >= self.min_clients)
            if len(self.choice_clients) >= self.min_clients:
                time.sleep(0.5)
                self.client_selection_random()
                
                if self.round > self.max_rounds:
                    self.round = 1
                self.choice_clients = []
                


    def on_message_agg(self, client, userdata, msg):
            self.agg_clients.append(json.loads(msg.payload))
            print("agg")
            print(len(self.agg_clients) >= self.min_clients)
            print(self.number_choice_clients)
            print(self.min_clients)
            print(len(self.agg_clients))
            if len(self.agg_clients) >= self.min_clients:
                data = []
                client_list = []
                time.sleep(0.5)
                self.cursor_db = self.connection_db.cursor()
                self.cursor_db.execute("""
                                    SELECT * FROM models WHERE ROUND=? ORDER BY Timestamp DESC;
                                   """, [self.round])
                if self.choice_mode != "random" and self.round != 1:
                    print(self.round)
                    print("CLIENT SELECTION")
                    print(self.C)
                    for linha in self.cursor_db.fetchmany(self.number_clients_non_random):
                        model_data = json.loads(linha[1])
                        client_list.append(linha[3])
                        data.append([np.asarray(i) for i in model_data['Weights']])
                    print(len(data))
                    with open('output.txt', 'a+') as f:
                        f.write('Round: ' + str(self.round) + " ")
                        f.write('number_clients: ' + str(self.number_clients_non_random) + " ")
                        f.write('C: ' + str(self.C) + " ")
                        f.write('len data: ' + str(len(data)) + " ")
                    agg_w = aggregate(data,self.number_clients_non_random)
                else:
                    for linha in self.cursor_db.fetchmany(self.number_choice_clients):
                            model_data = json.loads(linha[1])
                            client_list.append(linha[3])
                            data.append([np.asarray(i) for i in model_data['Weights']])
                    agg_w = aggregate(data,self.number_choice_clients)
                self.cursor_db.close()
                #agg_w = aggregate(data,self.number_choice_clients)
                print("Hashmap ocupation")
                print(np.count_nonzero(agg_w)/(len(agg_w)*len(agg_w[0]))*100)
                print("Hashmap max value")
                print(np.max(agg_w))
            
                self.similarity_order = []
                for i in range(len(data)):

                    #print(cdist(data[i],agg_w,metric="cosine"))
                    self.round_similarity = np.trace(cdist(data[i],agg_w,metric="cosine"))/len(agg_w)
                    self.similarity_order.append({'similarity':self.round_similarity,'resp_topic':client_list[i]})
                print("SIMILARITY ORDER SIZE")
                print(len(self.similarity_order))
                
                agg_w = [i.tolist() if type(i) != list else i for i in agg_w] 
                agg_data = {
                        'Weights' : agg_w
                    }
                ct = datetime.datetime.now()
                self.cursor_db = self.connection_db.cursor()
                self.cursor_db.execute("""
                            INSERT INTO aggregate_models (MODEL, ROUND, Timestamp)
                            VALUES (?, ?, ?)
                            """, (json.dumps(agg_data), self.round, ct))
                self.connection_db.commit()
                self.cursor_db.close()
                if self.round > 1 and self.choice_mode != "random":
                    if self.C <= 1:
                        self.round = self.max_rounds   
                self.round += 1
                if self.weight_decay == True:
                    global_learning_rate = step_decay(self.round,self.global_learning_rate)
                else:
                    global_learning_rate = self.global_learning_rate
                for c in self.agg_clients:
                    topic = c['resp_topic']
                    print(topic)
                    
                    msg = {
                        'resp_topic' : topic,
                        'round' : self.round,
                        'global_learning_rate' : global_learning_rate
                    }
                    publish.single(topic, json.dumps(msg), hostname=broker)
                self.agg_clients = []
                
                if self.round > self.max_rounds:
                    self.client.disconnect()
                if self.round > 2 and self.choice_mode != "random":
                    if self.C <= 1:
                       self.client.disconnect() 


    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)
        # Set Connecting Client ID
        client = mqtt_client.Client(client_paho_id)
        #client = mqtt_client.Client(client_paho_id,transport='websockets')
        #client.username_pw_set(username, password)
        client.on_connect = on_connect
        #client.on_log= self.on_log
        client.connect(broker, port, keepalive=3600)
        client.subscribe(topic_introduction)
        client.message_callback_add(topic_introduction, self.on_message_introduction)
        client.subscribe(topic_choice)
        if self.choice_mode == "random": 
            client.message_callback_add(topic_choice, self.on_message_choice_random)
        else:
            client.message_callback_add(topic_choice, self.on_message_choice_client_selection)
        client.subscribe(topic_agg)
        client.message_callback_add(topic_agg, self.on_message_agg)

        return client


    def run(self):
        self.client = self.connect_mqtt()
        try:
            self.client.loop_forever()
        except Exception as e:
            os.remove(fs.connection_address)
            self.client.disconnect()
            print(traceback.format_exc())
            print(e)

        finally:
            self.connection_db.close()

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


if __name__ == "__main__":
    #np.random.seed(0)
    #tf.random.set_seed(0)
    #random.seed(0)

    set_global_determinism(seed=0)

    min_clients = int(sys.argv[1])
    number_choice_clients = int(sys.argv[2])
    rounds = int(sys.argv[3])
    choice_mode = str(sys.argv[4])
    order_comparison = str(sys.argv[5])
    learning_rate = 0.1

    #choice_mode = "random" selecao aleatoria
    #choice_mode = "sketch" selecao por comparacao de sketchs
    #choice_mode = "accuracy" selecao por acuracia
    #choice_mode = "loss" selecao por loss
    fs = FederatedServer(min_clients, number_choice_clients, rounds,learning_rate ,choice_mode=choice_mode,weight_decay=False,order_comparison = order_comparison)

    fs.run()
    try:
        os.remove(fs.connection_address)
    except OSError:
        pass
