from nets.base_net import BaseNet
from utils import dataloader
from activate_fn import activate as act_fn
import tensorflow as tf
import tensorlayer as tl

class CifarConvNet(BaseNet):
  '''
  '''
  def __init__(self, in_shape = [32, 32, 3]):
    '''
    '''
    super().__init__(in_shape = in_shape)
    return

  def _build_network(self):
    '''
    '''
    network = tl.layers.InputLayer(self._inputs, name = 'input_layer')
    network = tl.layers.Conv2d(network, 32, (5, 5), (1, 1), act = act_fn.swish, padding = 'VALID', name = 'cnn1')
    network = tl.layers.Conv2d(network, 128, (5, 5), (1, 1), act = act_fn.swish, padding = 'VALID', name = 'cnn2')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding = 'VALID', name = 'pool2')
    network = tl.layers.Conv2d(network, 256, (3, 3), (2, 2), act = act_fn.swish, padding = 'VALID', name = 'cnn3')
    network = tl.layers.MaxPool2d(network, (2, 2), (1, 1), padding = 'VALID', name = 'pool3')
    network = tl.layers.FlattenLayer(network, name = 'flatten3')
    network = tl.layers.DropoutLayer(network, keep = 0.8, is_fix = True, name = 'drop4')
    network = tl.layers.DenseLayer(network, n_units = 800, act = act_fn.swish, name = 'dense5')
    network = tl.layers.DenseLayer(network, n_units = 10, act = tf.identity, name = 'output')
    return network

  def load_dataset(self, filename = None):
    '''
    @brief: load dataset and return X_train, Y_train, x_val, y_val, x_test, y_test
    '''
    return dataloader.load_cifar10_dataset(filename)

if __name__ == '__main__':
  raise Exception('Not permitted to launch cifar_conv_net.py directely')