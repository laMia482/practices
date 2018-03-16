import os
from utils import dataloader
import tensorflow as tf
import tensorlayer as tl


class Model:
  '''
  '''
  def __init__(self, cfg, input_network):
    '''
    '''
    self._cfg = cfg
    self._sess = self._build_session()
    self._model = input_network()
    self._net = self._model.network()
    self._inputs = self._model.inputs()
    self._ground_truth = self._model.outputs()
    self._predictions = self._model.predictions()
    return

  def __del__(self):
    '''
    '''
    self._sess.close()
    return

  def _build_session_config(self, is_gpu_grow = False):
    sess_cfg = tf.ConfigProto()
    sess_cfg.gpu_options.allow_growth = is_gpu_grow
    return sess_cfg

  def _build_session(self, sess_config = None):
    sess_config = self._build_session_config(self._cfg.gpu_grow)
    sess = tf.InteractiveSession(config = sess_config)
    return sess

  def load_npz(self, filename = None):
    '''
    '''
    tl.files.load_and_assign_npz_dict(sess = self._sess, name = filename)
    return self._sess

  def _load_data(self, filename = None):
    '''
    '''
    return self._model.load_dataset(filename)

  def train(self, x = None, y = None):
    '''
    '''
    if None in [x, y]:      
      x1, y1= self._load_data(self._cfg.train_file)
    else:
      x1, y1 = x, y
    x1 = x1.reshape(self._model.inputs_shape())
    precious = self._model.get_precious(self._predictions, self._ground_truth)
    cost = self._model.get_loss(self._predictions, self._ground_truth)

    train_params = self._net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.9, 
                beta2 = 0.999, epsilon = 1e-08, use_locking = False).minimize(
                  cost, var_list = train_params)
    tl.layers.initialize_global_variables(self._sess)
    self._net.print_params()
    self._net.print_layers()
    iters, epoch, cur_epoch = 0, 0, 0
    print('[*] start train')
    while True:
      batch_x, batch_y = dataloader.fetch_batch_size(x1, y1, self._cfg.batch_size)
      _, loss, acc = self._sess.run([train_op, cost, precious], 
                                    feed_dict = {self._inputs: batch_x, self._ground_truth: batch_y})
      iters += 1
      cur_epoch = int(iters * self._cfg.batch_size / x1.shape[0])
      if cur_epoch >= self._cfg.max_epoch:
        break
      if cur_epoch % self._cfg.show_every_n_epoch == 0 and cur_epoch != epoch:
        epoch = cur_epoch
        print('[*] reach {:3d}-th epoch, loss: {:.6f}, acc: {:.6f}'.format(cur_epoch, loss, acc))
      else:
        if iters % self._cfg.show_every_n_iter == 0:
          print('[*] iters: {:6d}, loss: {:.6f}, acc: {:.6f}'.format(iters, loss, acc))
    
    # tl.files.save_ckpt(self._sess, save_dir = self._cfg.save_path, mode_name = 'model.ckpt')
    # tl.files.save_npz(sess = self._sess, save_list = self._net.all_params, name = os.path.join(self._cfg.save_path, 'model.npz'))
    tl.files.save_npz_dict(sess = self._sess, save_list = self._net.all_params, name = os.path.join(self._cfg.save_path, 'model.npz'))
    return

  def eval(self, x = None, y = None):
    '''
    '''
    if None in [x, y]:      
      x3, y3= self._load_data(self._cfg.eval_file)
    else:
      x3, y3 = x, y
    x3 = x3.reshape(self._model.inputs_shape())
    # tl.files.load_ckpt(sess = self._sess, save_dir = self._cfg.save_path, mode_name = 'model.ckpt')
    # tl.files.load_and_assign_npz(sess = self._sess, name = os.path.join(self._cfg.save_path, 'model.npz'), network = self._net)
    self.load_npz(filename = os.path.join(self._cfg.save_path, 'model.npz'))
    precious = self._model.get_precious(self._predictions, self._ground_truth)
    cost = self._model.get_loss(self._predictions, self._ground_truth)
    current_batch_size = 1024
    print('[*] start eval')
    while True:
      try:
        tl.utils.test(self._sess, self._net, precious, x3, y3, self._inputs, self._ground_truth, batch_size = current_batch_size, cost = cost)
      except Exception:
        current_batch_size = int(current_batch_size / 2)
        print('Exception: not enough memory for using, trying smaller batch size as {}'.format(current_batch_size))
      else:
        break
    return

  def predict(self, inputs):
    '''
    @brief: get predictions from inputs
    '''
    x = inputs.reshape(self._model.inputs_shape())
    outputs = self._sess.run([self._predictions], feed_dict = {self._inputs: x})
    return outputs

if __name__ == '__main__':
  raise Exception('Not permitted to launch model.py directely')
