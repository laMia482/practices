from abc import ABCMeta, abstractmethod
from utils import dataloader
import tensorflow as tf
import tensorlayer as tl

class BaseNet:
  __metaclass__ = ABCMeta
  '''
  '''
  def __init__(self, in_shape = [28, 28, 1], in_dtype = tf.float32, out_shape = [], out_dtype = tf.int64):
    '''
    '''
    self._inputs = tf.placeholder(in_dtype, shape = [None] + in_shape, name = 'inputs')
    self._ground_truth = tf.placeholder(out_dtype, shape = [None] + out_shape, name = 'groundtruth')
    self._inner_feed_dict = {}
    self._network = self._build_network()
    self._predictions = self._network.outputs
    return

  @abstractmethod
  def _build_network(self):
    '''
    '''
    network = tl.layers.InputLayer(self._inputs, name = 'input_layer')
    network = tl.layers.Conv2d(network, 32, (5, 5), (1, 1), act = tf.nn.relu, padding = 'VALID', name = 'cnn1')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding = 'VALID', name = 'pool1')
    network = tl.layers.Conv2d(network, 128, (3, 3), (2, 2), act = tf.nn.relu, padding = 'VALID', name = 'cnn2')
    network = tl.layers.MaxPool2d(network, (2, 2), (1, 1), padding = 'VALID', name = 'pool2')
    network = tl.layers.FlattenLayer(network, name = 'flatten1')
    network = tl.layers.DropoutLayer(network, keep = 0.5, is_fix = True, name = 'drop1')
    network = tl.layers.DenseLayer(network, n_units = 800, act = tf.nn.relu, name = 'dense1')
    network = tl.layers.DenseLayer(network, n_units = 10, act = tf.identity, name = 'output')
    return network

  def push_inner_feed_dict(self, key, value):
    '''
    '''
    self._inner_feed_dict[key] = value
    return

  def get_inner_feed_dict(self):
    '''
    @brief: return the dict of some inner params such as dropout_prob 
            because all tl.layers.DropoutLayer has this issue: < take a placeholder to be filled >
    '''
    return self._inner_feed_dict

  def load_dataset(self, filename = None):
    '''
    @brief: load dataset and return X_train, Y_train, x_val, y_val, x_test, y_test
    '''
    return dataloader.load_mnist_dataset(filename)

  def inputs(self):
    '''
    '''
    return self._inputs

  def inputs_shape(self):
    '''
    '''
    sp = list(self._inputs.get_shape())
    sp[0] = -1
    return sp

  def outputs(self):
    '''
    '''
    return self._ground_truth

  def outputs_shape(self):
    '''
    '''
    sp = list(self._ground_truth.get_shape())
    sp[0] = -1
    return sp

  def network(self):
    '''
    '''
    return self._network
  
  def predictions(self):
    '''
    '''
    return self._predictions

  def get_loss(self, predictions, target):
    '''
    '''
    cost = tl.cost.cross_entropy(predictions, target, name = 'cost')
    return cost

  def get_precious(self, predictions, target):
    '''
    '''
    correct_prediction = tf.equal(tf.argmax(predictions, 1), target)
    precious = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return precious


if __name__ == '__main__':
  raise Exception('Not permitted to launch base_net.py directely')
