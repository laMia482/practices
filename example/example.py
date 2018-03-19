import numpy as np
import tensorflow as tf
import tensorlayer as tl

def classify():
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession(config = sess_config)

  X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape = (-1, 784))

  x = tf.placeholder(tf.float32, shape = [None, 784], name = 'x')
  y_ = tf.placeholder(tf.int64, shape = [None, ], name = 'y_')

  network = tl.layers.InputLayer(x, name = 'input_layer')
  network = tl.layers.DropoutLayer(network, keep = 0.8, name = 'drop1')
  network = tl.layers.DenseLayer(network, n_units = 800, act = tf.nn.relu, name = 'relu1')
  network = tl.layers.DropoutLayer(network, keep = 0.5, name = 'drop2')
  network = tl.layers.DenseLayer(network, n_units = 800, act = tf.nn.relu, name = 'relu2')
  network = tl.layers.DropoutLayer(network, keep = 0.5, name = 'drop3')
  network = tl.layers.DenseLayer(network, n_units = 10, act =tf.identity, name = 'output_layer')

  y = network.outputs
  cost = tl.cost.cross_entropy(y, y_, name = 'cost')
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  y_op = tf.argmax(tf.nn.softmax(y))

  train_params = network.all_params
  train_op = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1 = 0.9, beta2 = 0.999, 
                        epsilon = 1e-08, use_locking = False).minimize(cost, var_list = train_params)

  tl.layers.initialize_global_variables(sess)

  network.print_params()
  network.print_layers()

  tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_, acc = acc, batch_size = 500, 
            n_epoch = 10, print_freq = 1, X_val = X_val, y_val = y_val, eval_train = False)

  tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size = None, cost = cost)

  tl.files.save_npz(network.all_params, name = 'ckpts/model.npz')
  sess.close()

def linear():
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession(config = sess_config)
    
  trX = np.linspace(-1, 1, 101)  
  # trY = 2 * trX + np.ones(*trX.shape) * 4 + np.random.randn(*trX.shape) * 0.03
  trY = 2 * trX + np.random.randn(*trX.shape) * 0.03
  trX = trX.reshape([-1, 1])
  trY = trY.reshape([-1])
  X = tf.placeholder(tf.float32, shape = [None, 1])
  Y = tf.placeholder(tf.float32, shape = [None])
    
  def model(X, w, b):  
    # return X * w + b
    return tf.matmul(X, w) + b
    
  w = tf.Variable(0.0, name="weights")  
  b = tf.Variable(0.0, name="biases")  
  y_model = model(X, [[w]], b)  
    
  cost = tf.square(Y - y_model)
  train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
  init = tf.initialize_all_variables()  
  sess.run(init)  

  for i in range(10):
    for (x, y) in zip(trX, trY):
      _, w_val, b_val = sess.run([train_op, w, b], feed_dict={X: [x], Y: [y]})
    print('[*] epoch: {:2d}, w: {:.6f}, b: {:.6f}'.format(i, w_val, b_val))

  print('[*] test begins')
  x = np.array([-0.4, -0.8, 1.0, 0.5, 0.4, -0.8, 0.0, 0.7]).reshape([-1, 1])
  y_val = sess.run(y_model, feed_dict={X: x})
  print('x: \n{}'.format(x))
  print('y: \n{}'.format(y_val))

def main(_):
  linear()


if __name__ == '__main__':
  tf.app.run()