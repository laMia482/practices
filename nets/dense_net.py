from nets.base_net import BaseNet
from utils import dataloader
from activate_fn import activate as act_fn
import tensorflow as tf
import tensorlayer as tl

class DenseNet(BaseNet):
  '''
  '''
  def __init__(self, in_shape = [784], out_shape = []):
    '''
    '''
    super().__init__(in_shape = in_shape, out_shape = out_shape)
    return

  def _build_network(self):
    '''
    '''
    network = tl.layers.InputLayer(self._inputs, name = 'input_layer')
    network = tl.layers.FlattenLayer(network, name = 'flatten1')
    network = tl.layers.DenseLayer(network, n_units = 800, act = tf.nn.relu, name = 'dense1')
    network = tl.layers.DropoutLayer(network, keep = 0.5, is_fix = True, name = 'drop1')
    network = tl.layers.DenseLayer(network, n_units = 800, act = tf.nn.relu, name = 'dense2')
    network = tl.layers.DenseLayer(network, n_units = 10, act = tf.identity, name = 'output')
    return network

if __name__ == '__main__':
  raise Exception('Not permitted to launch dense_net.py directely')
