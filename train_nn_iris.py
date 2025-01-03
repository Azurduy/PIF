"""
This module is mostly identical (with a few modifications) 
to the originals from 
https://github.com/covit2019/analysis_codes
corresponding to 
Kubo, Yusuke et al. (2020). 'A copula-based visualization technique for a neural network'.

secondary author: Florian Heitmann
flowbiker@hotmail.de
"""


import sys
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

def load_data():
  iris = datasets.load_iris()
  features = iris.data
  features = normalize_features(features)
  targets = iris.target
  targets = tf.keras.utils.to_categorical(targets)
  (x_train, x_test, y_train, y_test) = train_test_split(features, targets, test_size=0.2, random_state=0,)
  return (features, targets), (x_train, y_train), (x_test, y_test)


def normalize_features(features):
  scaler = MinMaxScaler((0.,1.))
  scaler.fit( features )
  return scaler.transform(features)


def define_model():
  nodes = []
  model = tf.keras.models.Sequential([
    # flo
    # tf.keras.layers.Input(4),
    tf.keras.layers.Input((4,)),

    tf.keras.layers.Dense(6, activation=tf.nn.relu, use_bias=True),
    tf.keras.layers.Dense(6, activation=tf.nn.relu, use_bias=True),
    #tf.keras.layers.Dense(4, activation=tf.nn.relu, use_bias=True),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax, use_bias=True)
  ])
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

  model.summary()
  nodes.append(["x_0", "x_1", "x_2", "x_3"])
  nodes.append(["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5"])
  nodes.append(["h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"])
  nodes.append(["pred_0", "pred_1", "pred_2"])
  return model, nodes


def get_intermediate_layer_models(model):
  models = []
  for iHiddenLayer in range( len(model.layers)-1 ):
    models.append( tf.keras.models.Model(inputs = model.input, outputs = model.layers[iHiddenLayer].output) )
  return models


def draw_learning_plot(log_file, fig_name):
  df = pd.read_csv(log_file)
  fig = plt.figure(figsize=(16,6))
  # accuracy
  plt.subplot(1, 2, 1)
  plt.ylim(0.,1.)
  plt.grid()
  plt.plot(df["accuracy"], label = "acc")
  plt.plot(df["val_accuracy"], label = "val_acc")
  #plt.plot(df["accuracy"], label = "acc", marker = "o")
  #plt.plot(df["val_accuracy"], label = "val_acc", marker = "o")
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.legend(loc = "best")
  plt.grid(color = 'gray', alpha = 0.2)

  # loss
  plt.subplot(1, 2, 2)
  #plt.ylim(0.1,10.)
  #plt.yscale('log')
  plt.grid()
  plt.plot(df["loss"], label = "loss")
  plt.plot(df["val_loss"], label = "val_loss")
  #plt.plot(df["loss"], label = "loss", marker = "o")
  #plt.plot(df["val_loss"], label = "val_loss", marker = "o")

  plt.ylabel("loss")
  plt.xlabel("epoch")
  plt.legend(loc = "best")
  plt.grid(color = 'gray', alpha = 0.2)
  plt.savefig(fig_name)


def get_output_of_hidden_nodes(features, models):
  results = []
  for model in models:
     results.append( model.predict(features) )
  return results


# def dump_results( input_layer, hidden_layers, predictions, category, labels, filename):
def dump_results( input_layer, hidden_layers, predictions, category, labels, filename=None):
  output = pd.DataFrame(input_layer)
  for layer in hidden_layers:
    output = pd.DataFrame( np.concatenate([output.values, layer], 1) )
  output = pd.DataFrame( np.concatenate([output.values, predictions, category], 1) )
  labels_1d =[]
  for layer in labels:
    for label in layer:
      labels_1d.append(label)
  labels_1d.append("true_y")
  labels_1d.append("pred_y")

  output.columns = labels_1d
  # print("first rows of model values: \n", output.head() )
  if filename != None:
    output.to_csv(filename, index=None)
  return output


def get_y(targets, predictions):
  res = []
  print('len(targets) ->', len(targets))
  for i in range( len(targets) ):
    res.append([np.argmax(targets[i]), np.argmax(predictions[i])])
  return np.array(res)

# new function
def get_weights(model):
  weights = []
  biases = []
  for i in range(len(model.layers)):
    weights.append( model.layers[i].get_weights()[0] )
    biases.append( model.layers[i].get_weights()[1] )
  return weights, biases

if __name__ == '__main__':
  np.random.seed(42)
  tf.random.set_seed(42)

  (features, targets), (x_train, y_train), (x_test, y_test) = load_data()

  #model_name = './nn_model.h5'
  model, nodes = define_model()
  # Train it self
  log_file = './inout/train.log'
  csv_logger = tf.keras.callbacks.CSVLogger(log_file)
  #model.fit(x_train, y_train, epochs=10000, verbose=0, validation_data=(x_test, y_test), callbacks=[csv_logger])
  model.fit(x_train, y_train, epochs=1000, verbose=0, validation_data=(x_test, y_test), callbacks=[csv_logger])
  #model.save('./nn_model.h5')
  tf.keras.utils.plot_model(model, to_file='./inout/iris_nn_model.png', show_shapes=True, show_layer_names=False)
  draw_learning_plot(log_file, "./inout/learning_curve.png")

  hidden_layer_models = get_intermediate_layer_models(model)
  print('number of hidden layers ->', len(hidden_layer_models))
  intermediate_output = get_output_of_hidden_nodes(features, hidden_layer_models)

  predictions = model.predict(features)
  category = get_y(targets, predictions)
  dump_results( features, intermediate_output, predictions, category, nodes, filename='./inout/nn_outputs.csv' )

  weights, biases = get_weights(model)
  print("weights: ", weights)
  print("biases: ", biases)
  with open('./inout/weights', 'wb') as f:
    pickle.dump(weights, f)
  with open('./inout/biases', 'wb') as f:
    pickle.dump(biases, f)
  # with open('./inout/biases', 'rb') as f:
  #   loaded_biases = pickle.load(f)
  #   print("loaded_biases: ", loaded_biases)
    
  # save model
  model.save("./inout/model.keras")
