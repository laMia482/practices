from nets.base_net import BaseNet
from utils import dataloader
from activate_fn import activate as act_fn
import tensorflow as tf
import tensorlayer as tl

class LinearNet(BaseNet):
  '''
  '''
  def __init__(self, in_shape = [1], out_dtype = tf.float32):
    '''
    '''
    super().__init__(in_shape = in_shape, out_dtype = out_dtype)
    return

  def _build_network(self):
    '''
    '''
    network = tl.layers.InputLayer(self._inputs, name = 'input_layer')
    network = tl.layers.FlattenLayer(network, name = 'flatten')
    network = tl.layers.DenseLayer(network, n_units = 1, act = tf.nn.relu, name = 'output')
    return network

  def get_loss(self, predictions, target):
    '''
    '''
    cost = tl.cost.mean_squared_error(predictions, target, is_mean = True, name = 'mean_squared_error')
    return cost

  def get_precious(self, predictions, target):
      '''
      '''
      return self.get_loss(predictions, target)

if __name__ == '__main__':
  raise Exception('Not permitted to launch linear_net.py directely')
