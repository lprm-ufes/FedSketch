import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import sys
import tensorflow as tf
import numpy as np
from functools import reduce
import random
from tensorflow.keras.models import load_model
from paho.mqtt import client as mqtt_client
import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import time
import json
import logging
import sqlite3
import datetime
import traceback
logger = logging.getLogger()
logger.setLevel(logging.INFO) 

broker = 'localhost'
port = 1883
#port = 8083
topic_introduction = "introduction"
topic_choice= "choice"
topic_agg = "agg"
client_paho_id = f'python-mqtt-{random.randint(0, 1000)}'

    # username = 'emqx'
    # password = 'public'


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

def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = 0
    for i in results:
        num_examples_total += i[1]

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
class FederatedServer(object):

    def __init__(self, min_clients, number_choice_clients, max_rounds, use_number_choice=True ,chance=1):
        self.round = 1
        self.min_clients = min_clients
        self.number_choice_clients = number_choice_clients
        self.max_rounds = max_rounds
        self.agg_clients = []
        self.choice_clients = []
        self.use_number_choice= use_number_choice
        self.chance = chance
        self.global_seed = 0
        self.connection_address = 'agg.db'
        self.connection_db = sqlite3.connect(self.connection_address, timeout=1200)
        self.cursor_db = self.connection_db.cursor()
        self.cursor_db.execute("""
                        CREATE TABLE IF NOT EXISTS models (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        MODEL TEXT NOT NULL,
                        ROUND INTEGER NOT NULL,
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
           
        
    #def on_log(self, client, userdata, level, buf):
    #    print("log: ",buf)
    def on_message_introduction(self, client, userdata, msg):
        client_msg = json.loads(msg.payload)
        time.sleep(0.5)
        topic = client_msg['resp_topic']
        msg = {
                'total_round' : self.max_rounds,
                'global_seed' : self.global_seed,
                'connection_address' : self.connection_address
            }
        
        publish.single( topic, json.dumps(msg), hostname=broker)
    def on_message_choice(self, client, userdata, msg):
            self.choice_clients.append(json.loads(msg.payload))
            print("Choice")
            print(len(self.choice_clients) >= self.min_clients)
            if len(self.choice_clients) >= self.min_clients:
                time.sleep(0.5)
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
                
                if self.round > self.max_rounds:
                    self.round = 1
                self.choice_clients = []
                


    def on_message_agg(self, client, userdata, msg):
            self.agg_clients.append(json.loads(msg.payload))
            print("agg")
            print(len(self.agg_clients) >= self.min_clients)
            #print(len(self.agg_clients))
            if len(self.agg_clients) >= self.min_clients:
                data = []
                time.sleep(0.5)
                self.cursor_db = self.connection_db.cursor()
                self.cursor_db.execute("""
                                    SELECT * FROM models WHERE ROUND=? ORDER BY Timestamp DESC;
                                   """, [self.round])

                for linha in self.cursor_db.fetchmany(self.number_choice_clients):
                    model_data = json.loads(linha[1])
                    data.append([[np.asarray(i) for i in model_data['Weights']],int(model_data['Size'])])
                self.cursor_db.close()
                agg_w = aggregate(data)
                agg_w = [i.tolist() if type(i) != list else i for i in agg_w] 
                #print(agg_w)
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
                for c in self.agg_clients:
                    topic = c['resp_topic']
                    print(topic)
                    
                    msg = {
                        'resp_topic' : topic
                    }
                    publish.single( topic, json.dumps(msg), hostname=broker)
                self.agg_clients = []
                self.round += 1
                if self.round > self.max_rounds:
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
        client.message_callback_add(topic_choice, self.on_message_choice)
        client.subscribe(topic_agg)
        client.message_callback_add(topic_agg, self.on_message_agg)

        return client


    def run(self):
        self.client = self.connect_mqtt()
        try:
            self.client.loop_forever()
        except Exception as e:
            #os.remove(fs.connection_address)
            self.client.disconnect()
            print(traceback.format_exc())
            print(e)

        finally:
            self.connection_db.close()


if __name__ == "__main__":
    #np.random.seed(0)
    #tf.random.set_seed(0)
    #random.seed(0)
    min_clients = int(sys.argv[1])
    number_choice_clients = int(sys.argv[2])
    rounds = int(sys.argv[3])
    fs = FederatedServer(min_clients, number_choice_clients, rounds)

    fs.run()
    #try:
    #    #os.remove(fs.connection_address)
    #except OSError:
    #    pass
