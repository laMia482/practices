import tensorflow as tf

def swish(x):
  '''
  '''
  y = tf.nn.relu(x) * tf.nn.sigmoid(x)
  return y


def relu1(x, vmin = 0., vmax = 1.):
  '''
  '''
  y = tf.maximum(tf.minimum(x, vmax), vmin)
  return y

def l2norm(x, dim = 1):
  '''
  '''
  y = tf.nn.l2_normalize(x, dim = dim)
  return y