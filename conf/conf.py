import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_bool('test_example', False, 'switch to use an easy example')
flags.DEFINE_bool('is_train', True, 'switch for train or not')
flags.DEFINE_bool('is_eval', True, 'switch for eval or not')
flags.DEFINE_bool('is_predict', True, 'switch for use sample prediction or not')
flags.DEFINE_bool('is_fine_tune', True, 'fine-tune on the base of existing model')
flags.DEFINE_bool('gpu_grow', True, 'switch for limit gpu memory and allow it to grow if neccessary')
flags.DEFINE_integer('batch_size', 256, 'how many samples to feed in each training iteration')
flags.DEFINE_integer('max_epoch', 20, 'train for max epoch')
flags.DEFINE_integer('show_every_n_epoch', 50, 'show info every this epoch')
flags.DEFINE_integer('show_every_n_iter', 100, 'show info every this iter')
flags.DEFINE_string('train_file', 'data/cifar/train', 'train file')
flags.DEFINE_string('eval_file', 'data/cifar/eval', 'eval file')
flags.DEFINE_string('save_path', 'ckpts', 'path to save trained model')

config = flags.FLAGS

if __name__ == '__main__':
  raise Exception('Not permitted to launch cong.py directely')