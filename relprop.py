"""
This module is an implementation of the relevance propagation Ansatz
applied for pruning.
based on:
i) 
Bach, Sebastian et al. (July 2015). 'On Pixel-Wise Explanations for Non-Linear Classifier
Decisions by Layer-Wise Relevance Propagation'
ii)
Yeom, Seul-Ki et al. (2021). 'Pruning by explaining: A novel criterion for deep neural network
pruning'


author: Florian Heitmann
flowbiker@hotmail.de


##############
# pseudocode #
##############
# INPUT: data, model, hidden_layer_output, output
# i)
# copy nn model (store relevance values here)
# ii) 
# relevance_values = pd.DataFrame()
# for point in data:
#     one hot encoding of output nodes (store as bias in model_copy)
#     for layer in layers:
#         apply LRP (store relevance values as biases in model_copy)
#     relevance_values.append(model_copy.biases)
# iii)
# for node on nodes:
#     score = sum up relevance values
# sort nodes with scores

"""

import copy
import numpy as np
import pandas as pd
import tensorflow as tf







def relprop(model, nn_outputs, pos_weights_only=True, alpha=1.0):
    predictions = nn_outputs["pred_y"].to_numpy(copy=True)
    # data should only be the node values
    data = nn_outputs.drop(columns=["true_y", "pred_y"]).copy(deep=True)
    data_len = len(data.index)
    headers=[*data]
    relevance_values = pd.DataFrame(columns=headers)

    # iterate over NN input data
    for k in range(data_len):
        row = data.iloc[k]
        # set last layer (ll) relevance values
        # ll_weights = model_cp.layers[-1].get_weights()[0]
        ll_biases = model.layers[-1].get_weights()[1]
        # rel_vals_row = np.zeros(len(ll_biases))

        rel_vals_row = [0.0] * len(ll_biases)
        # # test propagating all information in result
        # rel_vals_row = [1.0] * len(ll_biases)

        # predicted class gets relevance 1.0, others 0.0
        rel_vals_row[int(predictions[k])] = 1.0

        # iterate reversely over hidden layers
        # for idx_rev, layer in enumerate(model_cp.layers[::-1]):
        for idx_rev, layer in enumerate(model.layers[::-1]):    
            idx = (len(model.layers)-1)-idx_rev
            # weight matrix
            W = layer.get_weights()[0]
            # biases
            b = layer.get_weights()[1]
            # activation values of current layer (l)
            a = row[-len(rel_vals_row)-W.shape[0]:-len(rel_vals_row)].to_numpy()
            

            rel_vals_current_layer = np.zeros(W.shape[0])
            for j in range(W.shape[1]):
                # nodes that already have relevance values (layer l+1)
                R_j = rel_vals_row[j]
                col_j = W[:,j]
                aiwij_pos = np.array([a_i*w_ij if a_i*w_ij>0 else 0.0 for a_i,w_ij in zip(a, col_j)])
                sum_i_pos = sum(aiwij_pos) # over i
                if pos_weights_only == True:
                    Rij_pos= aiwij_pos*R_j/sum_i_pos
                    rel_vals_current_layer = np.nansum(np.stack((Rij_pos, rel_vals_current_layer)), axis=0)
                else: # with biases and relevance conservation factor (rcf)
                    bj_pos = 0.0
                    if b[j]>0:
                        bj_pos = b[j]
                    sum_i_pos += bj_pos
                    if sum_i_pos == 0.0: sum_i_pos = 1.0

                    aiwij_neg = np.array([a_i*w_ij if a_i*w_ij<0 else 0.0 for a_i,w_ij in zip(a, col_j)])
                    sum_i_neg = sum(aiwij_neg) # over i
                    bj_neg = 0.0
                    if b[j]<0:
                        bj_neg = b[j]
                    sum_i_neg += bj_neg
                    if sum_i_neg == 0.0: sum_i_neg = 1.0

                    # relevance conservation factor
                    rcf = 1.0 - alpha*bj_pos/sum_i_pos - (1.0-alpha)*bj_neg/sum_i_neg
                    Rij_pos, Rij_neg = aiwij_pos*R_j/(sum_i_pos*rcf), aiwij_neg*R_j/(sum_i_neg*rcf)
                    rel_vals_current_layer = np.nansum(np.stack((alpha*Rij_pos, (1.0-alpha)*Rij_neg, rel_vals_current_layer)), axis=0)


            rel_vals_row = rel_vals_current_layer.tolist() + rel_vals_row

        relevance_values.loc[len(relevance_values)] = rel_vals_row

    return relevance_values


def sort_node_scores(values, hidden_nodes):
    scores = {}
    for node in hidden_nodes:
        # scores[node] = sum(relevance_values[node].to_numpy())
        scores[node] = np.mean(values[node].to_numpy())
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_scores



# ########
# # TEST #
# ########
# # propagating all result classes

# # iterate over NN input data
# for k in range(data_len):
#     row = data.iloc[k]
#     # set last layer (ll) relevance values
#     # ll_weights = model_cp.layers[-1].get_weights()[0]
#     ll_biases = model.layers[-1].get_weights()[1]
#     ll_node_count = len(ll_biases)
#     # rel_vals_row = [0.0] * ll_node_count
   

#     for node_idx in range(ll_node_count): 
#         rel_vals_row = [0.0] * ll_node_count
#         # predicted class gets relevance 1.0, others 0.0
#         rel_vals_row[node_idx] = 1.0
#         print("rel_vals_row: ", rel_vals_row)

#         # iterate reversely over hidden layers
#         # for idx_rev, layer in enumerate(model_cp.layers[::-1]):
#         for idx_rev, layer in enumerate(model.layers[::-1]):    
#             idx = (len(model.layers)-1)-idx_rev
#             # weight matrix
#             W = layer.get_weights()[0]
#             # biases
#             b = layer.get_weights()[1]
#             # activation values of current layer (l)
#             a = row[-len(rel_vals_row)-W.shape[0]:-len(rel_vals_row)].to_numpy()
            
#             # print("")
#             # print("layer: ", idx)
#             # print("#############")
#             # print("weight matrix: \n", W)
#             # print("biases: \n", b)
#             # print("activation values: \n", a)
#             # exit()

#             rel_vals_current_layer = np.zeros(W.shape[0])
#             for j in range(W.shape[1]):
#                 # nodes that already have relevance values (layer l+1)
#                 R_j = rel_vals_row[j]
#                 col_j = W[:,j]
#                 aiwij_pos = np.array([a_i*w_ij if a_i*w_ij>0 else 0.0 for a_i,w_ij in zip(a, col_j)])
#                 sum_i_pos = sum(aiwij_pos) # over i
#                 bj_pos = 0.0
#                 if b[j]>0:
#                     bj_pos = b[j]
#                 sum_i_pos += bj_pos
#                 if sum_i_pos == 0.0: sum_i_pos = 1.0

#                 aiwij_neg = np.array([a_i*w_ij if a_i*w_ij<0 else 0.0 for a_i,w_ij in zip(a, col_j)])
#                 sum_i_neg = sum(aiwij_neg) # over i
#                 bj_neg = 0.0
#                 if b[j]<0:
#                     bj_neg = b[j]
#                 sum_i_neg += bj_neg
#                 if sum_i_neg == 0.0: sum_i_neg = 1.0

#                 # relevance conservation factor
#                 rcf = 1.0 - alpha*bj_pos/sum_i_pos - (1.0-alpha)*bj_neg/sum_i_neg
#                 Rij_pos, Rij_neg = aiwij_pos*R_j/(sum_i_pos*rcf), aiwij_neg*R_j/(sum_i_neg*rcf)
#                 rel_vals_current_layer = np.nansum(np.stack((alpha*Rij_pos, (1.0-alpha)*Rij_neg, rel_vals_current_layer)), axis=0)

#                 # print("R_j: ", R_j)
#                 # print("col_j of weight matrix: ", col_j)
#                 # print("activation values: \n", a)
#                 # print("aiwij_pos: ", aiwij_pos)
#                 # print("sum_i_pos: ", sum_i_pos, "at node", j)
#                 # print("aiwij_neg: ", aiwij_neg)
#                 # print("sum_i_neg: ", sum_i_neg, "at node", j)
#                 # print(" Rij_pos, Rij_neg: ",  Rij_pos, Rij_neg)
#                 # print("np.nansum(np.stack((0.5*Rij_pos, 0.5*Rij_neg)), axis=0): \n", np.nansum(np.stack((0.5*Rij_pos, 0.5*Rij_neg)), axis=0))
#                 # print("rel_vals_current_layer: \n", rel_vals_current_layer)

#             rel_vals_row = rel_vals_current_layer.tolist() + rel_vals_row
#             # print("rel_vals_row: \n", rel_vals_row)
#             # exit()

#         relevance_values.loc[len(relevance_values)] = rel_vals_row
#         # print("")
#         # print("relevance_values: \n", relevance_values)
#         # exit()

# ############
# # END TEST #
# ############


if __name__ == "__main__":

    #########
    # INPUT #
    #########
    alpha = 0.5 # alpha + beta = 1.0


    model = tf.keras.models.load_model("./inout/model.keras")
    nn_outputs = pd.read_csv("./inout/nn_outputs.csv")
    relevance_values = relprop(model, nn_outputs, alpha=alpha)

    # sort nodes with respect to relevance score
    hidden_nodes = ["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5", "h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"]
    # scores = {}
    # for node in hidden_nodes:
    #     # scores[node] = sum(relevance_values[node].to_numpy())
    #     scores[node] = np.mean(relevance_values[node].to_numpy())
    sorted_scores = sort_node_scores(relevance_values, hidden_nodes)

    for keys,values in sorted_scores.items():
        print(keys + ": ", values)

