from conf.conf import config as cfg
from nets.cifar_conv_net import CifarConvNet as WorkingNet
from models.model import Model


def main():
  '''
  '''
  if cfg.test_example is True:
    import example.example
    example.example.classify()
  else:
    model = Model(cfg = cfg, input_network = WorkingNet)
    if cfg.is_train is True:
      model.train()
    if cfg.is_eval is True:
      model.eval()
    if cfg.is_predict is True:
      import os
      import numpy as np
      inputs = np.random.random([4, 32, 32, 3])
      model.load_npz(filename = os.path.join('ckpts', 'model.npz'))
      outputs = model.predict(inputs = inputs)
      print('outputs: \n{}'.format(outputs))


if __name__ == '__main__':
  '''
  '''
  main()
