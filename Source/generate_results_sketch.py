from pydoc import describe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from scipy import stats

sketch_client_selection_mnist = pd.read_csv("../Resultados/sketch_pytorch_MNIST_sketch_50_avg_correct.csv",header=None)
sketch_client_selection_famnist = pd.read_csv("../Resultados/sketch_pytorch_FAMNIST_sketch_50.csv",header=None)
sketch_client_selection_cifar = pd.read_csv("../Resultados/sketch_pytorch_CIFAR10_sketch_50.csv",header=None)

acc_client_selection_mnist = pd.read_csv("../Resultados/sketch_pytorch_MNIST_accuracy_avg_correct.csv",header=None)
acc_client_selection_famnist = pd.read_csv("../Resultados/sketch_pytorch_FAMNIST_accuracy_50.csv",header=None)
acc_client_selection_cifar = pd.read_csv("../Resultados/sketch_pytorch_CIFAR10_accuracy_50.csv",header=None)

sketch_random_mnist_12 = pd.read_csv("../Resultados/sketch_pytorch_MNIST_random_12.csv",header=None)
sketch_random_famnist_12 = pd.read_csv("../Resultados/sketch_pytorch_FAMNIST_random_12.csv",header=None)
sketch_random_cifar_12 = pd.read_csv("../Resultados/sketch_pytorch_CIFAR10_random_12.csv",header=None)

sketch_random_mnist_25 = pd.read_csv("../Resultados/sketch_pytorch_MNIST_random_25.csv",header=None)
sketch_random_famnist_25 = pd.read_csv("../Resultados/sketch_pytorch_FAMNIST_random_25.csv",header=None)
sketch_random_cifar_25 = pd.read_csv("../Resultados/sketch_pytorch_CIFAR10_random_25.csv",header=None)

sketch_random_mnist_37 = pd.read_csv("../Resultados/sketch_pytorch_MNIST_random_37.csv",header=None)
sketch_random_famnist_37 = pd.read_csv("../Resultados/sketch_pytorch_FAMNIST_random_37.csv",header=None)
sketch_random_cifar_37 = pd.read_csv("../Resultados/sketch_pytorch_CIFAR10_random_37.csv",header=None)

sketch_random_cifar_50 = pd.read_csv("../Resultados/sketch_pytorch_CIFAR10_random_50.csv",header=None)
sketch_random_mnist_50 = pd.read_csv("../Resultados/sketch_pytorch_MNIST_random_50.csv",header=None)
sketch_random_famnist_50 = pd.read_csv("../Resultados/sketch_pytorch_FAMNIST_random_50.csv",header=None)

random_mnist_50 = pd.read_csv("../Resultados/pytorch_MNIST_random_50.csv",header=None)
random_cifar_50 = pd.read_csv("../Resultados/pytorch_CIFAR10_random_50.csv",header=None)
random_famnist_50 = pd.read_csv("../Resultados/pytorch_FAMNIST_random_50.csv",header=None)


columns = ['Accuracy','Client ID', 'Round', 'Chosen'] 
sketch_client_selection_mnist.columns = columns 
acc_client_selection_mnist.columns = columns 

sketch_random_mnist_12.columns = columns 
sketch_random_mnist_12 = sketch_random_mnist_12.loc[sketch_random_mnist_12['Round'] <= 50]

sketch_random_cifar_12.columns = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen'] 
sketch_random_famnist_12.columns = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen'] 
sketch_random_mnist_25.columns = columns
sketch_random_mnist_25 = sketch_random_mnist_25.loc[sketch_random_mnist_25['Round'] <= 50]
sketch_random_cifar_25.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen'] 
sketch_random_famnist_25.columns = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen'] 

sketch_random_mnist_37.columns = columns
sketch_random_cifar_37.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
sketch_random_famnist_37.columns = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen'] 
sketch_random_mnist_50.columns = columns
sketch_random_cifar_50.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
sketch_random_famnist_50.columns = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen'] 
sketch_client_selection_cifar.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
sketch_client_selection_famnist.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
acc_client_selection_cifar.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
acc_client_selection_famnist.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
random_mnist_50.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
random_cifar_50.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']
random_famnist_50.columns  = ['Accuracy','Train_Accuracy','Loss','Client ID', 'Round', 'Chosen']

random_mnist_50['Round'] = random_mnist_50['Round']+1
#sketch_random_cifar_50.columns = columns
##print("Sketch comparison")
##print(sketch_client_selection.groupby('Round')['Chosen'].mean())
##print(np.mean(sketch_client_selection.groupby('Round')['Chosen'].mean()))
#print("Acc comparison")
#print(sketch_client_selection_decreasing_correct.groupby('Round')['Chosen'].mean())
#print(np.mean(sketch_client_selection_decreasing_correct.groupby('Round')['Chosen'].mean()))
##print("Loss comparison")
##print(loss_client_selection.groupby('Round')['Chosen'].mean())
##print(np.mean(loss_client_selection.groupby('Round')['Chosen'].mean()))
sketch_client_selection_mnist['Client Selection Algorithm'] = 'sketch comparison'
sketch_client_selection_famnist['Client Selection Algorithm'] = 'sketch comparison'
sketch_client_selection_cifar['Client Selection Algorithm'] = 'sketch comparison'
acc_client_selection_mnist['Client Selection Algorithm']= 'accuracy comparison'
acc_client_selection_famnist['Client Selection Algorithm']= 'accuracy comparison'
acc_client_selection_cifar['Client Selection Algorithm']= 'accuracy comparison'
##loss_client_selection_decreasing['Client Selection Algorithm'] = 'loss comparison decreasing'
#sketch_client_selection_increasing['Client Selection Algorithm'] = 'sketch comparison increasing'
#acc_client_selection_increasing['Client Selection Algorithm']= 'accuracy comparison increasing'
##loss_client_selection_increasing['Client Selection Algorithm'] = 'loss comparison increasing'
sketch_random_mnist_12['Client Selection Algorithm'] = 'random 25%'
sketch_random_famnist_12['Client Selection Algorithm'] = 'random 25%'
sketch_random_cifar_12['Client Selection Algorithm'] = 'random 25%'
sketch_random_mnist_25['Client Selection Algorithm'] = 'random 50%'
sketch_random_famnist_25['Client Selection Algorithm'] = 'random 50%'
sketch_random_cifar_25['Client Selection Algorithm'] = 'random 50%'

#sketch_random_mnist_25_non_iid['Client Selection Algorithm'] = 'random 50% non iid'
#sketch_random_50_global['Client Selection Algorithm'] = 'random 50% with global learning rate'
#sketch_random_50_global_with_decay['Client Selection Algorithm'] = 'random 50% with global learning rate with decay'
sketch_random_mnist_37['Client Selection Algorithm'] = 'random 75%'
sketch_random_famnist_37['Client Selection Algorithm'] = 'random 75%'
sketch_random_cifar_37['Client Selection Algorithm'] = 'random 75%'
sketch_random_mnist_50['Client Selection Algorithm'] = 'all FedSketch'
sketch_random_famnist_50['Client Selection Algorithm'] = 'all FedSketch'
sketch_random_cifar_50['Client Selection Algorithm'] = 'all FedSketch'
random_mnist_50['Client Selection Algorithm'] = 'all FedAvg'
random_famnist_50['Client Selection Algorithm'] = 'all FedAvg'
random_cifar_50['Client Selection Algorithm'] = 'all FedAvg'
#sketch_random_cifar_50['Client Selection Algorithm'] = 'all'
##df_plot = pd.concat([sketch_client_selection,sketch_random],axis=0)
##df_plot = pd.concat([sketch_random_100,acc_client_selection_increasing,sketch_random_75
##                     ,sketch_random_50,sketch_client_selection_decreasing,sketch_random_25,sketch_random_50_global,sketch_random_50_global_with_decay],axis=0)
df_plot_clients_25 = pd.DataFrame.from_dict({
                                          'Round': np.arange(50)+1,
                                          'Number Clients': np.ones(50)*25
                                        })
df_plot_clients_sketch_mnist = pd.DataFrame.from_dict({
                                          'Round': np.arange(50)+1,
                                          'Number Clients': sketch_client_selection_mnist.groupby('Round')['Chosen'].mean()*50
                                        })
print(sketch_random_mnist_50.groupby('Round')['Accuracy'].mean())
print(sketch_random_famnist_50.groupby('Round')['Accuracy'].mean())
print(sketch_random_cifar_50.groupby('Round')['Accuracy'].mean())
print(random_mnist_50.groupby('Round')['Accuracy'].mean())
print(random_famnist_50.groupby('Round')['Accuracy'].mean())
print("Selection")
print(np.mean(acc_client_selection_mnist.groupby('Round')['Chosen'].mean())*50)
print(np.mean(sketch_client_selection_famnist.groupby('Round')['Chosen'].mean())*50)
print(np.mean(acc_client_selection_cifar.groupby('Round')['Chosen'].mean())*50)
print("Sketch comparison")
#print(sketch_client_selection_mnist.groupby('Round')['Accuracy'].mean()*50)
#print(sketch_client_selection_mnist.groupby('Round')['Accuracy'].mean())

#print("Sketch comparison")
#print(sketch_client_selection_cifar.groupby('Round')['Accuracy'].mean()*50)
#print(sketch_client_selection_cifar.groupby('Round')['Accuracy'].mean())
#df_plot_clients_sketch_inc = pd.DataFrame.from_dict({
#                                          'Round': np.arange(100)+1,
#                                          'Number Clients': sketch_client_selection_increasing.groupby('Round')['Chosen'].mean()*100
#                                        })
#
df_plot_clients_acc_mnist = pd.DataFrame.from_dict({
                                          'Round': np.arange(50)+1,
                                          'Number Clients': acc_client_selection_mnist.groupby('Round')['Chosen'].mean()*50
                                        })
print("ACC comparison")
#print(acc_client_selection_famnist.groupby('Round')['Accuracy'].mean()*50)
print(acc_client_selection_famnist.groupby('Round')['Accuracy'].mean())
#df_plot_clients_acc_dec = pd.DataFrame.from_dict({
#                                          'Round': np.arange(100)+1,
#                                          'Number Clients': acc_client_selection_decreasing.groupby('Round')['Chosen'].mean()*100
#                                        })
#
df_plot_clients_25['Client Selection Algorithm'] = 'random 50%'
df_plot_clients_sketch_mnist['Client Selection Algorithm'] = 'sketch comparison'
#df_plot_clients_sketch_inc['Client Selection Algorithm'] = 'sketch comparison increasing'
df_plot_clients_acc_mnist['Client Selection Algorithm']= 'accuracy comparison'
#df_plot_clients_acc_dec['Client Selection Algorithm']= 'accuracy comparison decreasing'
df_plot_clients = pd.concat([df_plot_clients_25,df_plot_clients_sketch_mnist,df_plot_clients_acc_mnist],axis=0)
#print(df_plot_clients.head())
df_plot_mnist_all = pd.concat([sketch_random_mnist_50,random_mnist_50],axis=0)
df_plot_famnist_all = pd.concat([sketch_random_famnist_50,random_famnist_50],axis=0)
df_plot_cifar_all = pd.concat([sketch_random_cifar_50,random_cifar_50],axis=0)


df_plot_mnist = pd.concat([sketch_random_mnist_50,sketch_random_mnist_37,sketch_random_mnist_25,acc_client_selection_mnist,
                     ],axis=0)#,sketch_random_mnist_25_non_iid],axis=0)
                     #sketch_client_selection_mnist_non_iid, acc_client_selection_mnist_non_iid],axis=0)
#                     ,sketch_client_selection_increasing,acc_client_selection_increasing,
#                     sketch_random_50_global,sketch_random_50_global_with_decay],axis=0)
#df_plot_mnist = df_plot_mnist.reset_index()
df_plot_famnist = pd.concat([sketch_client_selection_famnist,sketch_random_famnist_25,sketch_random_famnist_37,sketch_random_famnist_50
                     ],axis=0)
df_plot_cifar = pd.concat([sketch_random_cifar_25,acc_client_selection_cifar,sketch_random_cifar_37,sketch_random_cifar_50,
                           ],axis=0)
df_plot_cifar = df_plot_cifar.reset_index()

plt.figure()
b = sns.lineplot(x='Round', y='Accuracy', hue='Client Selection Algorithm', 
             data=df_plot_mnist_all)
b.set_title("(a): MNIST",fontsize=15)
b.set_xlabel("Round",fontsize=15)
b.set_ylabel("Acurácia",fontsize=15)
handles, labels = b.get_legend_handles_labels()
b.legend(handles=handles[0:], labels=labels[0:])
plt.savefig('Imagens/sketch/Mean accuracy per round sketch non sketch comparison MNIST.png',dpi=300, bbox_inches = "tight")


#plt.show()
plt.clf()

plt.figure()
b = sns.lineplot(x='Round', y='Accuracy', hue='Client Selection Algorithm', 
             data=df_plot_famnist_all)
b.set_title("(b): FAMNIST",fontsize=15)
b.set_xlabel("Round",fontsize=15)
b.set_ylabel("Acurácia",fontsize=15)
handles, labels = b.get_legend_handles_labels()
b.legend(handles=handles[0:], labels=labels[0:])
plt.savefig('Imagens/sketch/Mean accuracy per round sketch non sketch comparison FAMNIST.png',dpi=300, bbox_inches = "tight")

#plt.show()
plt.clf()

plt.figure()
b = sns.lineplot(x='Round', y='Accuracy', hue = "Client Selection Algorithm", 
             data=df_plot_cifar_all)
b.set_title("(c): CIFAR10",fontsize=15)
b.set_xlabel("Round",fontsize=15)
b.set_ylabel("Acurácia",fontsize=15)
handles, labels = b.get_legend_handles_labels()
b.legend(handles=handles[0:], labels=labels[0:])
#sns.lineplot(x='Round', y='Train_Accuracy', color = "blue", 
#             data=df_plot_cifar)
plt.savefig('Imagens/sketch/Mean accuracy per round sketch non sketch comparison CIFAR10.png',dpi=300, bbox_inches = "tight")
plt.title("Comparação entre o uso e a ausência do uso de sketchs CIFAR10.")
#plt.show()
plt.clf()

modes = ['all FedSketch', 'all FEDAVG', 'random 50%', 'random 75%', 'accuracy comparison', 'sketch comparison']
colors = sns.color_palette('tab10', len(modes))

# create a dictionary of modes and colors
palette = dict(zip(modes, colors))

plt.figure()
b = sns.lineplot(x='Round', y='Accuracy', hue='Client Selection Algorithm', palette=palette,
             data=df_plot_mnist)
b.set_title("(a): MNIST",fontsize=15)
b.set_xlabel("Round",fontsize=15)
b.set_ylabel("Acurácia",fontsize=15)
plt.savefig('Imagens/sketch/Mean accuracy per round for all selection Algorithms MNIST.png',dpi=300, bbox_inches = "tight")

#plt.show()
plt.clf()

plt.figure()
b = sns.lineplot(x='Round', y='Accuracy', hue='Client Selection Algorithm', palette=palette, 
             data=df_plot_famnist)
b.set_title("(b): FAMNIST",fontsize=15)
b.set_xlabel("Round",fontsize=15)
b.set_ylabel("Acurácia",fontsize=15)
plt.savefig('Imagens/sketch/Mean accuracy per round for all selection Algorithms FAMNIST.png',dpi=300, bbox_inches = "tight")

#plt.show()
plt.clf()

plt.figure()
b = sns.lineplot(x='Round', y='Accuracy', hue = "Client Selection Algorithm", palette=palette, 
             data=df_plot_cifar)
#sns.lineplot(x='Round', y='Train_Accuracy', color = "blue", 
#             data=df_plot_cifar)
b.set_title("(c): CIFAR10",fontsize=15)
b.set_xlabel("Round",fontsize=15)
b.set_ylabel("Acurácia",fontsize=15)
plt.savefig('Imagens/sketch/Mean accuracy per round for all selection Algorithms CIFAR10.png',dpi=300, bbox_inches = "tight")
#plt.show()
plt.clf()

##
plt.figure()
b = sns.lineplot(x='Round', y='Number Clients', hue='Client Selection Algorithm', 
             data=df_plot_clients)
b.set_xlabel("Round",fontsize=15)
b.set_ylabel("Acurácia",fontsize=15)
#plt.savefig('Imagens/sketch/Mean accuracy per round for both selection Algorithms random 50.png',dpi=300, bbox_inches = "tight")
plt.title("Número de Clientes escolhidos por round para cada Algorithmo de seleção.")
#plt.show()
plt.clf()