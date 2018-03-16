from nets.base_net import BaseNet
from activate_fn import activate as act_fn
import tensorflow as tf
import tensorlayer as tl

class RnnNet(BaseNet):
  '''
  '''
  def __init__(self, inshape = [784], outshape = []):
    '''
    '''
    super().__init__(inshape = inshape, outshape = outshape)
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

