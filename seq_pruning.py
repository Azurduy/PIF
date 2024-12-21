# sequential pruning specific nodes of the NN 
# by setting corresponding weights to zero

##############
# pseudocode #
##############
# i)
# calculate least important node of original model
# with node activation values
# ii)
# set corresponding weights to zero ("pruning")
# iii)
# get node activation values of pruned model
# iv)
# repeat until finalization condition


from train_nn_iris import *
from cm import *
from NNCorrelationMatrix import *
from prune import *


def get_activation_values(model, nodes, features, targets, filename=None):
    # features, targets = get_iris()
    hidden_layer_models = get_intermediate_layer_models(model)
    intermediate_output = get_output_of_hidden_nodes(features, hidden_layer_models)
    predictions = model.predict(features)
    category = get_y(targets, predictions)
    # activals = dump_results(features, intermediate_output, predictions, category, nodes, filename='./inout/nn_outputs_temp.csv' )
    activals = dump_results(features, intermediate_output, predictions, category, nodes, filename=filename)
    return activals

def get_correlation_matrix(correlation_indicator, activals, save_as=None):
    # df_pseudo_obs, headers = get_pseudo_observations('./inout/nn_outputs_temp.csv')
    df_pseudo_obs, headers = get_pseudo_observations(activals)

    # # test of correlation values 
    # # select test-data of 2 variables
    # pseudo_obs1, pseudo_obs2 = df_pseudo_obs[headers[0]].to_numpy(), df_pseudo_obs[headers[1]].to_numpy()
    # test_correlation_indicators(pseudo_obs1, pseudo_obs2, headers[:2])

    cm = get_cm(df_pseudo_obs, headers, correlation_indicator)                
    # replace weired values with np.nan
    cm.where(cm<=1.0, inplace=True)
    cm.where(cm>=-1.0, inplace=True)
    if save_as != None:
        cm.to_csv("./inout/"+save_as+".csv", header=False, index=None)
    return cm


def get_node_scores(hidden_nodes_all, active_hidden_nodes, CM, show=False):
    # CM = NNCorrelationMatrix('CM', './inout/iris_CM_new_temp.csv', './inout/nodes_4_6_6_3.txt')
    products_pred0 = pd.DataFrame(CM.products_to_node("pred_0"))
    products_pred1 = pd.DataFrame(CM.products_to_node("pred_1"))
    products_pred2 = pd.DataFrame(CM.products_to_node("pred_2"))
    products_pred0.columns = ["reversed_path", "pred_0"]
    products_pred1.columns = ["reversed_path", "pred_1"]
    products_pred2.columns = ["reversed_path", "pred_2"]
    products = deepcopy(products_pred0)
    products["pred_1"] = deepcopy(products_pred1["pred_1"])
    products["pred_2"] = deepcopy(products_pred2["pred_2"])
     # new path importance score
    products["score"] = np.abs(products["pred_0"]) + np.abs(products["pred_1"]) + np.abs(products["pred_2"])
    # very similar results:
    # products["score"] = products.apply(lambda row: np.max([row["pred_0"], row["pred_1"], row["pred_2"]]), axis=1)
    
    # sort nodes with respect to sum of path importance scores
    path_col = "reversed_path"
    score_col = "score"
    # hidden_nodes = ["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5", "h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"]
    scores = {}
    # initialize dictionary
    for node in hidden_nodes_all:
        scores[node] = []
    # get data series from dataframe
    path_series = products[path_col]
    score_series = products[score_col]

    for i, path in enumerate(path_series):
        for node in path[:-1]:
            scores[node].append(score_series[i])
    # calulate sum over all scores at each node
    for node in hidden_nodes_all:
        scores[node] = np.nansum(scores[node])

    active_scores = {}
    for node in active_hidden_nodes:
        active_scores[node] = scores[node]
    sorted_scores = dict(sorted(active_scores.items(), key=lambda item: item[1], reverse=True))
    if show==True:
        for keys,values in sorted_scores.items():
            print(keys + ": ", values)
    return sorted_scores


def seq_prune(node_count_2b_pruned, correlation_indicator, current_model,
              nodes, hidden_nodes_all, features, targets):
    to_be_pruned = []
    active_hidden_nodes = copy.deepcopy(hidden_nodes_all)
    for i in range(node_count_2b_pruned):
        # features, targets = get_iris()
        activals = get_activation_values(current_model, nodes, features, targets)
        cm = get_correlation_matrix(correlation_indicator, activals)
        CM = NNCorrelationMatrix("CM", cm, './inout/nodes_4_6_6_3.txt')
        sorted_scores = get_node_scores(hidden_nodes_all, active_hidden_nodes, CM)
        # least important node
        node_2b_pruned = list(sorted_scores)[-1]
        # print("")
        # print("pruning stage:", i)
        # # print("current_nodes: \n", current_nodes)
        # print("node_2b_pruned: ", node_2b_pruned)

        active_hidden_nodes.remove(node_2b_pruned)
        # print("active_hidden_nodes: \n", active_hidden_nodes)
        
        nodes_to_be_pruned = [node_2b_pruned]
        current_model = copy.deepcopy(prune(current_model, nodes_to_be_pruned))
        to_be_pruned += [node_2b_pruned]
    return to_be_pruned



if __name__ == "__main__":

    #########
    # INPUT #
    #########
    # path = "C:/Users/flowb/OneDrive/TWVL/tesis_grothe/ai_cbv/new/inout/nn_outputs_temp.csv"

    # choose one
    # correlation_indicator = "kendall"
    # correlation_indicator = "spearman"
    # correlation_indicator = "chatterjee"
    # correlation_indicator = "MIC"
    # correlation_indicator = "TIC" # many NaNs
    # correlation_indicator = "spearman_numerical"
    correlation_indicator = "absolute_distance_numerical"

    node_count_2b_pruned = 4


    nodes = []
    nodes.append(["x_0", "x_1", "x_2", "x_3"])
    nodes.append(["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5"])
    nodes.append(["h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"])
    nodes.append(["pred_0", "pred_1", "pred_2"])

    hidden_nodes = ["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5", "h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"]

    current_model = tf.keras.models.load_model("./inout/model.keras")
    # current_nodes = nodes
    hidden_nodes_all = copy.deepcopy(hidden_nodes)
    
    # print("hidden_active_nodes: \n", active_hidden_nodes)
    features, targets = get_iris()
    
    to_be_pruned = seq_prune(node_count_2b_pruned, correlation_indicator, current_model,
                             nodes, hidden_nodes_all, features, targets)
    print(to_be_pruned)

