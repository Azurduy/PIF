"""
This module is mostly identical (with a few modifications) 
to the originals from 
https://github.com/covit2019/analysis_codes
corresponding to 
Kubo, Yusuke et al. (2020). 'A copula-based visualization technique for a neural network'.

secondary author: Florian Heitmann
flowbiker@hotmail.de
"""


from copy import deepcopy
import numpy as np

class NodeStructure():
  def __init__(self, node_file):
    self.__layers = []
    self. __n_params = 0
    with open(node_file) as f:
      for line in f.readlines():
        layer =line.split()
        self.__layers.append(layer)
        self.__n_params += len( layer )


  def show(self):
    print(self.__layers)


  def layers(self):
    return self.__layers


  def n_params(self):
    return self.__n_params


  def find(self, node):
    for i, layer in enumerate(self.__layers):
      if node in layer:
        return i, layer.index(node)
    print(node, 'was not found in this NodeStructure')
    return None


  def route_to_node(self, node):
    routes = [[node]]
    last_layer = self.find(node)[0]
    for iLayer in range(last_layer, 0,-1):
      children = self.__layers[iLayer-1]
      routes = deepcopy(self.__add_branches(routes, children))
    return routes
    #return np.array(routes)


  def __add_branches(self, routes, branches):
    res = []
    for route in routes:
      for branch in branches:
        tmp = deepcopy(route)
        tmp.append(branch)
        res.append( tmp )
    return res


  def interpret_route(self, route):
    res = []
    for i in range(1, len(route)):
       res.append( (self.find(route[i-1]), self.find(route[i])) )
    return res


  def position_to_index(self, pos):
    offset = 0
    for iLayer in range(pos[0]):
      offset += len(self.__layers[iLayer])
    return offset + pos[1]


if __name__ == '__main__':
  nodes = NodeStructure('./nodes.txt')
  #nodes.show()
  #print(nodes.n_params())
  #print(nodes.find('pred_0') , nodes.layers()[nodes.find('pred_0')[0]][nodes.find('pred_0')[1]])

  routes = nodes.route_to_node('pred_0')
  print(nodes.interpret_route(routes[0]))
