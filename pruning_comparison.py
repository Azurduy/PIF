"""
This module compares different pruning methods
based on performance measures (accuracy & distance).

author: Florian Heitmann
flowbiker@hotmail.de


##############
# pseudocode #
##############
# i)
# define random seeds & build model
# split data
# ii) 
# train model with training data
# iii) 
# get node activation values for training (?) data 
# iv)
# for each method to be compared:
# calculate least important nodes (for different pruning percentages)
# prune NN & calculate performance metrics for test (?) data
# got to i) until finalization condition
# v)
# save performance metrics and generate boxplot
# alternatively: read performance metrics and generate boxplot
"""


from train_nn_iris import *
from cm import *
from NNCorrelationMatrix import *
from prune import *
from seq_pruning import *
from relprop import *

import os
import random
import matplotlib.pyplot as plt
import numpy as np


def build_model(optimizer_config):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((4,)),
        tf.keras.layers.Dense(6, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(6, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax, use_bias=True)])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=optimizer_config["learning_rate"]),
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss=optimizer_config["loss"],
        metrics=optimizer_config["metrics"])
    return model


def calculate_metrics(metrics, model, features, targets, threshold):
    vals = pd.DataFrame(metrics)
    predictions = model.predict(features)
    vals[metrics[0]] = get_accuracy(model, features, targets, predictions, threshold=threshold)
    vals[metrics[1]] = get_distance(model, features, targets, predictions)
    return vals


def average_sort_print(dfs, metric, methods, reverse=True):
    metric_averages = {}
    for method in methods:
        for i in node_count_2bp:
            vals = dfs[method][str(i)][metric]
            metric_averages[method+"_"+str(i)] = np.mean(vals)

    sorted_avgs = dict(sorted(metric_averages.items(), key=lambda item: item[1], reverse=reverse))
    for keys,values in sorted_avgs.items():
        print(keys + ": ", values)


if __name__ == "__main__":

    #########
    # INPUT #
    #########

    repetitions = 100
    # for data splitting
    test_size = 0.2

    # numbers of nodes to be pruned
    # node_count_2bp =  [2, 4, 6, 8]
    # reduce runtime for finetuning
    node_count_2bp =  [4, 6, 8]
    # how many layers are trainable for finetuning? (last ones...)
    layers2train = 2
    

    methods = [
               "relprop_pos_weights_only",
               "relprop_rcf_alpha1.0", # alpha is read from string
               "relprop_rcf_alpha0.5", # alpha is read from string
               "kendall_path",
               "spearman_path",
               "chatterjee_path",
               "kendall_path_seq",
               "spearman_path_seq",
               "chatterjee_path_seq",  
               ]
    metrics = ["accuracy", "distance"]

    nodes = []
    nodes.append(["x_0", "x_1", "x_2", "x_3"])
    nodes.append(["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5"])
    nodes.append(["h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"])
    nodes.append(["pred_0", "pred_1", "pred_2"])

    hidden_nodes = ["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5", "h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"]

    optimizer_config = {}
    optimizer_config["learning_rate"] = 0.001
    optimizer_config["loss"] = "categorical_crossentropy"
    optimizer_config["metrics"] = ["accuracy",]

    # for prediction
    threshold = 0.5

    # 42
    answer = 42


    # 0)
    features, targets = get_iris()
    dfs = {}
    for method in methods:
        dfs[method] = {}
        for i in node_count_2bp:
            dfs[method][str(i)] = pd.DataFrame(metrics)

    # i)
    # define random seeds & build model & split data
    
    random_seeds = [answer+i for i in range(repetitions)]
    for rs in random_seeds:
        print("")
        print("current random seed: ", rs, " final rs: ", str(answer+len(random_seeds)))
        print("")


        os.environ['PYTHONHASHSEED']=str(rs)
        rs += 1
        random.seed(rs)
        rs += 1
        np.random.seed(rs)
        rs += 1
        tf.random.set_seed(rs)
        rs += 1
        # # new global `tensorflow` session?!
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # tf.keras.backend.set_session(session)

        # split data
        (x_train, x_test, y_train, y_test) = train_test_split(features, targets, test_size=test_size, random_state=rs)


    # ii) 
    # train model with training data
        model = build_model(optimizer_config)
        model.fit(x_train, y_train, epochs=1000, verbose=0, validation_data=(x_test, y_test),)
        model.save("./inout/models/model_rs"+str(rs-4)+".keras")

    # # OR load trained model
    #     model = tf.keras.models.load_model("./inout/models/model_rs"+str(rs-4)+".keras")



    # iii) 
    # get node activation values for training (??) data
        # x, y = x_train, y_train
        x, y = features, targets # using all data
        activals = get_activation_values(model, nodes, x, y, filename=None) # eventually @./inout/nn_outputs_temp.csv


    # iv)
    # for each method to be compared:
    # calculate least important nodes (for different pruning percentages)
    # prune NN & calculate performance metrics for test (??) data
    # got to i) until finalization condition
        for method in methods:
            if "path" in method and "seq" not in method:
                correlation_indicator = method[:-5]
                print("")
                print("path method with correlation_indicator: ", correlation_indicator)
                cm = get_correlation_matrix(correlation_indicator, activals)
                hidden_nodes_all, active_hidden_nodes = hidden_nodes, hidden_nodes
                CM = NNCorrelationMatrix("CM", cm, './inout/nodes_4_6_6_3.txt')
                sorted_scores = get_node_scores(hidden_nodes_all, active_hidden_nodes, CM, show=False)

            elif "relprop" in method:
                # nn_outputs = pd.read_csv("./inout/nn_outputs_temp.csv")
                model_temp = copy.deepcopy(model)
                if "pos_weights_only" in method:
                    relevance_values = relprop(model, activals, pos_weights_only=True)
                else: # with biases and relevance conservation factor (rcf)
                    alpha = float(method[-3:])
                    relevance_values = relprop(model, activals, pos_weights_only=False, alpha=alpha)

                sorted_scores = sort_node_scores(relevance_values, hidden_nodes)

            elif "path_seq" in method:
                # these methods eventually modify "./inout/nn_outputs_temp.csv"
                correlation_indicator = method[:-9]
                current_model = copy.deepcopy(model)
                to_be_pruned = seq_prune(max(node_count_2bp), correlation_indicator, current_model,
                                         nodes, hidden_nodes, features, targets)
                sorted_scores = to_be_pruned[::-1]
                print("")
                print("path_seq method with correlation_indicator: ", correlation_indicator)
                 

            for i in node_count_2bp:
                    # least important nodes
                    nodes_2b_pruned = list(sorted_scores)[-i:]
                    print("nodes_2b_pruned: ", nodes_2b_pruned)
                    model_temp = copy.deepcopy(model)

                    # # SELECT
                    # # shallow pruning without fine tuning
                    # model_pruned = prune(model_temp, nodes_2b_pruned)
                    # # OR deep pruning and fine tuning
                    # model_pruned = prune_deep(model_temp, nodes_2b_pruned) 
                    model_pruned = prune_deep(model_temp, nodes_2b_pruned, train_layers_last=layers2train)
                    model_pruned.fit(x_train, y_train, epochs=1000, verbose=0, validation_data=(x_test, y_test),)
                    model_pruned.save("./inout/models/model_rs"+str(rs-4)+"_"+method+"_prunednodes"+str(i)+"_finetuned"+str(layers2train)+".keras")
                    # model_pruned = tf.keras.models.load_model("./inout/models/model_rs"+str(rs-4)+"_"+method+"_prunednodes"+str(i)+"_finetuned"+str(layers2train)+".keras")

                    metric_values = calculate_metrics(metrics, model_pruned, features, targets, threshold)
                    # dfs[method][str(i)].append(metric_values, ignore_index=True)
                    dfs[method][str(i)] = pd.concat( [dfs[method][str(i)], metric_values] )

    # v)
    # save/open performance metrics and generate boxplot
                    
    # save dataframe-dictionary OR (below)
    with open('./inout/pruning_methods_metrics', 'wb') as f:
        pickle.dump(dfs, f)


    # # open saved dictionary
    # # with open('./inout/pruning_methods_metrics', 'rb') as f:
    # # with open('./inout/pruning_methods_metrics_30s', 'rb') as f:
    # # with open('./inout/pruning_methods_metrics_100s', 'rb') as f:
    # with open('./inout/pruning_methods_metrics_100s_finetuning_last2layers', 'rb') as f:
    #     dfs = pickle.load(f)

    
    # # metric = metrics[0] # accuracy
    # metric = metrics[1] # distance
    # # average_sort_print(dfs, metric, methods, reverse=True)


    # fig, axs = plt.subplots(nrows=len(node_count_2bp), ncols=1)
    # for idx, i in enumerate(node_count_2bp):
    #     data = []
    #     for method in methods:
    #         data_temp = dfs[method][str(i)][metric].to_numpy()
    #         data += [data_temp[~np.isnan(data_temp)]]
    #     axs[idx].boxplot(data)
    #     axs[idx].yaxis.grid(True)
    #     axs[idx].set_xticks([y + 1 for y in range(len(data))], labels=["" for i in range(len(methods))])
    #     axs[idx].set_ylabel(str(i) + ' nodes pruned \n' + metric, fontsize=30)
    #     axs[idx].tick_params(labelsize=18)
      
    # pretty_methods = ["\n".join(m.split("_")) for m in methods]
    # axs[-1].set_xticks([y + 1 for y in range(len(data))], labels=pretty_methods, fontsize=30)
    # axs[-1].set_xlabel('methods', fontsize=40)
    # fig.set_size_inches(32,18)
    # # fig.savefig(metric+"_comparison.svg", format="svg", dpi=1200)
    # fig.savefig("./inout/"+metric+"_comparison.eps", format="eps", bbox_inches="tight", dpi=1200)
    # plt.show()

    

