import NodeStructure as ns

import pandas as pd
import numpy as np
#from operator import itemgetter
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

import pprint

class NNCorrelationMatrix():
  # def __init__(self, name, cm_file, node_file):
  def __init__(self, name, cm, node_file):
    self.__name = name
    # self.__full_CM = pd.read_csv(cm_file, header=None)
    self.__full_CM = cm
    self.__network = ns.NodeStructure(node_file)

    if self.__full_CM.shape[0] != self.__network.n_params():
      print('!! number of parameters are inconsistent between two csv files !!')

    tmp_columns = []
    for layer in self.__network.layers():
      for node in layer:
        
        tmp_columns.append(node)
    self.__full_CM.columns = tmp_columns
    self.__full_CM.index = tmp_columns


  def show_cm(self):
    print(self.__full_CM)


  def draw_cm(self):
    plt.figure(figsize = (10,7))

    ax = sns.heatmap(self.__full_CM, square=True, annot=True, vmin=-1, vmax=1, cmap='jet')#, xticklabels=label, yticklabels=label)
    new_yticks = [t - 0.5 for t in ax.get_yticks()]
    new_yticks.append(ax.get_yticks()[-1] + 0.5)
    ax.set_yticks(new_yticks, minor=True)
    plt.title("Correlation Matrix")

    plt.savefig("./inout/" + self.__name + ".png")  


  def show_layers(self):
    self.__network.show()


  def products_to_node(self, node):
    res = []
    routes = self.__network.route_to_node(node)
    #print('routes ->\n', routes)
    # for route in routes:
    # test:
    for j, route in enumerate(routes):
      # if j==0:
        # print("node: ", node)
        # print("route: ", route)
        # print("len(route): ", len(route))
      product = 1.
      for i in range(1, len(route)):
        product *= self.__full_CM[route[i-1]][route[i]]
        # if j==0:
          # print("route[i-1], route[i]: ", (route[i-1], route[i]))
          # print("self.__full_CM[route[i-1]][route[i]]: ", self.__full_CM[route[i-1]][route[i]])
          # print("product: ", product)
      #res.append( (route, product) )
      res.append( (route[1:], product) )
    return res


if __name__ == '__main__':
  # CM = NNCorrelationMatrix('CM', './iris_all_CM.csv', './nodes.txt')
  # CM = NNCorrelationMatrix('CM', './inout/iris_CM.csv', './inout/nodes_4_6_6_3.txt')
  CM = NNCorrelationMatrix('CM', './inout/iris_CM_new.csv', './inout/nodes_4_6_6_3.txt')
  
  CM.show_cm()
  print("")
  CM.draw_cm()

  #CM.show_layers()
  products_pred0 = pd.DataFrame(CM.products_to_node("pred_0"))
  products_pred1 = pd.DataFrame(CM.products_to_node("pred_1"))
  products_pred2 = pd.DataFrame(CM.products_to_node("pred_2"))

  # print( products_pred0 )
  # print( products_pred1 )
  # print( products_pred2 )
  # print( products_pred0.head(10) )
  # print( products_pred1.head(10) )
  # print( products_pred2.head(10) )
  # exit()


  products_pred0.columns = ["reversed_path", "pred_0"]
  products_pred1.columns = ["reversed_path", "pred_1"]
  products_pred2.columns = ["reversed_path", "pred_2"]

  n_labels = 3
  products = deepcopy(products_pred0)
  products["pred_1"] = deepcopy(products_pred1["pred_1"])
  products["pred_2"] = deepcopy(products_pred2["pred_2"])
  products["mean"] = (products["pred_0"] + products["pred_1"] + products["pred_2"]) / n_labels
  products["deviation"] = np.sqrt( ((products["pred_0"]-products["mean"])**2 + (products["pred_1"]-products["mean"])**2 + (products["pred_2"]-products["mean"])**2) / (n_labels-1))
  # new path importance score
  products["score"] = np.abs(products["pred_0"]) + np.abs(products["pred_1"]) + np.abs(products["pred_2"])
  # # very similar results:
  # products["score"] = products.apply(lambda row: np.max([row["pred_0"], row["pred_1"], row["pred_2"]]), axis=1)
  # test with excluded pred_0 OR pred_2, because of monotone dependence
  # products["score"] = np.abs(products["pred_1"]) + np.abs(products["pred_2"])
  # products["score"] = np.abs(products["pred_0"]) + np.abs(products["pred_1"])

  # sort nodes with respect to sum of path importance scores
  path_col = "reversed_path"
  score_col = "score"
  hidden_nodes = ["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5", "h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"]
  scores = {}
  # initialize dictionary
  for node in hidden_nodes:
    scores[node] = []
  # get data series from dataframe
  path_series = products[path_col]
  score_series = products[score_col]

  for i, path in enumerate(path_series):
    for node in path[:-1]:
      scores[node].append(score_series[i])
  # calulate sum over all scores at each node
  for node in hidden_nodes:
    scores[node] = np.nansum(scores[node])
  sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
  for keys,values in sorted_scores.items():
    print(keys + ": ", values)
  # exit()

  products = products.sort_values("deviation", ascending=False)
  products.to_csv("./inout/products_of_correlation_sorted_by_deviation.csv", index=None)
  print( products )

  products = products.sort_values("score", ascending=False)
  products.to_csv("./inout/products_of_correlation_sorted_by_score.csv", index=None)
  print( products )


