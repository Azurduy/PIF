"""
This module provides tools to
prune specific nodes of a nural network,
i) by setting corresponding weights to zero
ii) by deep pruning

includes performance measures (accuracy & distance)

author: Florian Heitmann
flowbiker@hotmail.de
"""




import copy
import pickle
import numpy as np
import tensorflow as tf
from itertools import combinations
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


def prune(model, nodes_to_be_pruned):
    for node in nodes_to_be_pruned:
        split = node.split("_")
        node_layer = int(split[0][1:])+1
        node_idx = int(split[1])
        weights = model.layers[node_layer].get_weights()[0]
        biases = model.layers[node_layer].get_weights()[1]
        # print("weights: \n", weights)
        # print("biases: \n", biases)

        # set weights of node to be "pruned" to zero
        weights[node_idx] = 0.0
        model.layers[node_layer].set_weights([weights, biases])
        # weights = model.layers[node_layer].get_weights()[0]
        # print("weights: \n", weights)
        # print("model.layers: ", model.layers)

        # exit()
    return model


def prune_deep(model, nodes_to_be_pruned, train_layers_last=3):
    """
    The given model is pruned, but instead of setting the corresponding
    node weights to zero, the nodes are totally deleted.
    """
    # to be deleted
    rows_per_layer = {} # rows of weight matrix after node
    cols_per_layer = {} # columns of weight matrix before node
    for node in nodes_to_be_pruned:
        split = node.split("_")
        node_layer_prev = int(split[0][1:])
        node_layer = node_layer_prev+1
        node_idx = int(split[1])

        if node_layer not in [*rows_per_layer]: # keys/layers
            rows_per_layer[node_layer] = [node_idx]
        else:
            rows_per_layer[node_layer] += [node_idx]

        if node_layer_prev not in [*cols_per_layer]: # keys/layers
            cols_per_layer[node_layer_prev] = [node_idx]
        else:
            cols_per_layer[node_layer_prev] += [node_idx]

    weights = [layer.get_weights()[0] for layer in model.layers]
    biases = [layer.get_weights()[1] for layer in model.layers]

    for layer in [*rows_per_layer]:
        weights[layer] = np.delete(weights[layer], rows_per_layer[layer], 0)

    for layer in [*cols_per_layer]:
        weights[layer] = np.delete(weights[layer], cols_per_layer[layer], 1)
        biases[layer] = np.delete(biases[layer], cols_per_layer[layer], 0)
      
    node_counts = [w.shape[1] for w in weights] # per layer (from input to output)

    # print("")
    # print("prune deep")
    # print("new node counts per layer: ", node_counts)
    # print("new weights: \n", weights)
    # print("new biases: \n", biases)
    # print("")


    # creating sequential model
    # initialize layer accumulator
    new_layers = [tf.keras.layers.Input((4,))]
    for idx, nc in enumerate(node_counts):
        if nc != 0:
            new_layers += [tf.keras.layers.Dense(nc, activation=tf.nn.relu, use_bias=True)]
        else: pass
    # output layer
    new_layers += [tf.keras.layers.Dense(3, activation=tf.nn.softmax, use_bias=True)]
    new_model = tf.keras.models.Sequential(new_layers)

    # weights_before = [layer.get_weights()[0] for layer in new_model.layers]
    # biases_before = [layer.get_weights()[1] for layer in new_model.layers]
    # print("weights before weight setting: \n", weights_before)
    # print("biases before weight setting: \n", biases_before)
    # print("")

    # set weights
    zero_count = 0
    for idx, nc in enumerate(node_counts):
        if nc != 0:
            try:
                new_model.layers[idx-zero_count].set_weights([weights[idx], biases[idx]])
            except: pass # omit setting weights of fully deleted layers
        else: zero_count += 1

    # define trainable layers
    layers_count = len(new_model.layers)
    for i in range(layers_count):
        new_model.layers[i].trainable = False
    for i in range(train_layers_last):
        try:
            new_model.layers[-i].trainable = True
        except: pass


            
    new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],)


    # weights = new_model.layers[node_layer].get_weights()[0]
    # biases = new_model.layers[node_layer].get_weights()[1]
    # # print("weights: \n", weights)
    # # print("biases: \n", biases)

    return new_model



# evaluate pruned model
def normalize_features(features):
  scaler = MinMaxScaler((0.,1.))
  scaler.fit( features )
  return scaler.transform(features)

def get_iris():
    iris = datasets.load_iris()
    features = iris.data
    features = normalize_features(features)
    targets = iris.target
    targets = tf.keras.utils.to_categorical(targets)
    return features, targets

def get_accuracy(model, features, targets, predictions, threshold=0.5):
    # predictions = model.predict(features)
    predictions_bin = (predictions > threshold).astype(float)
    accuracy = 1 - 0.5*np.sum((predictions_bin-targets)**2)/np.sum(targets)

    # print("")
    # print("in get_accuracy")
    # print("---------------")
    # print("predictions: \n", predictions)
    # # print("argmax(predictions): \n", np.argmax(predictions, axis=1))
    # print("predictions_bin: \n", predictions_bin)
    # print("targets: \n", targets)
    # print("accuracy: \n", accuracy)
    # print("NEW pre accuracy: \n", np.argmax(predictions, axis=1)==np.argmax(targets, axis=1))
    # print("")
    # exit()

    return accuracy

def get_distance(model, features, targets, predictions):
    # predictions = model.predict(features)
    average_dist = np.mean(np.sqrt((predictions-targets)**2))
    return average_dist



if __name__ == "__main__":

    #########
    # INPUT #
    #########
    hidden_nodes = ["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5", "h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"]
    nodes_to_be_pruned = {}
    # monotone correlation indicators
    nodes_to_be_pruned["tau_path"] = ["h1_2", "h0_2", "h0_1", "h1_4"]
    nodes_to_be_pruned["rho_path"] = ["h0_3", "h0_0", "h1_0", "h1_4"]
    # monotone correlation indicators with seq. pruning (same!)
    nodes_to_be_pruned["tau_path_seq"]= ["h0_0", "h0_3", "h1_0", "h1_5"]
    nodes_to_be_pruned["rho_path_seq"]= ["h0_0", "h0_3", "h1_0", "h1_5"]
    # non-monotone correlation indicators
    nodes_to_be_pruned["chatterjee_path"] = ["h0_3", "h1_0", "h1_5", "h1_3"]
    nodes_to_be_pruned["MIC_path"] = ["h1_5", "h1_2", "h1_1", "h0_3"]
    nodes_to_be_pruned["absolute_distance_numerical"] = ["h0_0", "h0_3", "h1_3", "h1_4"]
    # non-monotone correlation indicators with seq. pruning
    nodes_to_be_pruned["chatterjee_path_seq"] = ["h0_3", "h1_0", "h1_5", "h1_3"]
    nodes_to_be_pruned["MIC_path_seq"] = ["h1_0", "h1_1", "h1_2", "h1_5"]

    # relevance propagation
    nodes_to_be_pruned["relprop_rcf_alpha1"] = ["h1_4", "h0_3", "h0_0", "h1_0"]
    nodes_to_be_pruned["relprop_rcf_alpha0.5"] = ["h0_3", "h1_4", "h0_0", "h1_3"]

    # subset_size = 4
    # subsets = list(combinations(hidden_nodes, subset_size))
    # for subset in subsets:
    #     nodes_to_be_pruned[str(list(subset))] = list(subset)
    # print(nodes_to_be_pruned)

    model_original = tf.keras.models.load_model("./inout/model.keras")

    # for prediction
    threshold = 0.5


    features, targets = get_iris()
    orig_predictions = model_original.predict(features)
    accuracies = {}
    accuracies["original"] = get_accuracy(model_original, features, targets, orig_predictions, threshold=threshold)
    distances = {}
    distances["original"] = get_distance(model_original, features, targets, orig_predictions,)

    for method, nodes_2bp in nodes_to_be_pruned.items():
        model = prune(copy.deepcopy(model_original), nodes_2bp)
        predictions = model.predict(features)
        accuracies[method] = get_accuracy(model, features, targets, predictions, threshold=threshold)
        distances[method] = get_distance(model, features, targets, predictions)


    sorted_accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
    sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=False))
        
    print("accuracies")
    for keys,values in sorted_accuracies.items():
        print(keys + ": ", values)
    print("distances")
    for keys,values in sorted_distances.items():
        print(keys + ": ", values)

    

    # save dictionary
    # with open('./inout/pruned_model_accuracies', 'wb') as f:
    #     pickle.dump(accuracies, f)

    # open saved dictionary
    # with open('./inout/pruned_model_accuracies', 'rb') as f:
    #     accuracies = pickle.load(f)
    #     # print(accuracies)
    #     sorted_accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
    #     for keys,values in sorted_accuracies.items():
    #         print(keys + ": ", values)
        
