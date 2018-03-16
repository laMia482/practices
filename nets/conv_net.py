from nets.base_net import BaseNet
from utils import dataloader
from activate_fn import activate as act_fn
import tensorflow as tf
import tensorlayer as tl

max_predictions_num = 50

class ConvNet(BaseNet):
  '''
  '''
  def __init__(self, in_shape = [224, 224, 3], out_shape = [max_predictions_num, 5], out_dtype = tf.float32):
    '''
    '''
    super().__init__(in_shape = in_shape, out_shape = out_shape, out_dtype = out_dtype)
    return

  def _build_network(self):
    '''
    '''
    network = tl.layers.InputLayer(self._inputs, name = 'input_layer')
    network = tl.layers.Conv2d(network, 32, (5, 5), (2, 2), act = tf.nn.relu, padding = 'VALID', name = 'cnn1')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding = 'VALID', name = 'pool1')
    network = tl.layers.Conv2d(network, 128, (5, 5), (3, 3), act = tf.nn.relu, padding = 'VALID', name = 'cnn2')
    network = tl.layers.MaxPool2d(network, (2, 2), (1, 1), padding = 'VALID', name = 'pool2')
    network = tl.layers.Conv2d(network, 256, (1, 1), (1, 1), act = tf.nn.relu, padding = 'VALID', name = 'cnn3')
    network = tl.layers.Conv2d(network, 512, (5, 5), (2, 2), act = tf.nn.relu, padding = 'VALID', name = 'cnn4')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding = 'VALID', name = 'pool4')
    network = tl.layers.FlattenLayer(network, name = 'flatten4')
    network = tl.layers.DenseLayer(network, n_units = max_predictions_num * 5, act = tf.nn.relu, name = 'dense4')
    network = tl.layers.ReshapeLayer(network, shape = (-1, max_predictions_num, 5), name = 'output')
    return network

  def load_dataset(self, filename = None):
    '''
    @brief: load dataset and return X_train, Y_train, x_val, y_val, x_test, y_test
    '''
    return dataloader.load_voc_dataset(filename)

  def get_loss(self, predictions, target):
    '''
    '''
    cost = tl.cost.mean_squared_error(predictions, target, is_mean = True, name = 'mean_squared_error')
    return cost

  def get_precious(self, predictions, target):
    '''
    '''
    predictions_labels = tf.cast((20 * predictions[:, :1]), tf.int64)
    ground_truth_labels = tf.cast(target[:, :1], tf.int64)
    correct_prediction = tf.equal(predictions_labels, ground_truth_labels)
    precious = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return precious

if __name__ == '__main__':
  raise Exception('Not permitted to launch conv_net.py directely')