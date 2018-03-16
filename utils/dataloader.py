import numpy as np
import cv2
import tensorlayer as tl


def fetch_batch_size(x, y, batch_size = 32, is_shuffle = True):
  '''
  @brief: fetch batch_size pices from inputs with random order or not
          assume x, y is in numpy.array
  '''
  xs, ys = x.shape[0], y.shape[0]
  batch_indexes = np.random.choice(xs, batch_size)
  X, Y = [], []
  for index in batch_indexes:
    X.append(x[index])
    Y.append(y[index])
  return np.asarray(X), np.asarray(Y)


def load_mnist_dataset(filename):
  '''
  '''
  X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape = (-1, 784))
  return X_train, y_train


def load_cifar10_dataset(filename):
  '''
  '''
  X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape = (-1, 32, 32, 3))
  return X_train, y_train


def load_voc_dataset(filename):
  '''
  @brief: [[label, x_center, y_center, width, height], [...]...] where
          labe:int64, others:float32 in ratio format
  '''
  imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list, \
    classes, classes_in_person, classes_dict, n_objs_list, objs_info_list, \
    objs_info_dicts = tl.files.load_voc_dataset(path = '/data/e0024/workspace/DATASET_VOC', dataset = '2012')
  X, Y, X_train, Y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], [], [], []
  for i in range(len(imgs_file_list)):
    img = cv2.imread(imgs_file_list[i])
    if img is None:
      print('image file: {} not found, skip...'.format(imgs_file_list[i]))
      continue
    img = (cv2.resize(img, (224, 224)).astype(np.float32) - 128.) / 255.
    X.append(img)
    ground_truth = tl.prepro.parse_darknet_ann_str_to_list(objs_info_list[i])
    while np.shape(ground_truth)[0] < 50:
      ground_truth.append([0, 0, 0, 0, 0])
    ground_truth = ground_truth[:50]
    Y.append(ground_truth)
  X = np.array(X)
  Y = np.array(Y)
  step = int(1. / 3 * len(imgs_file_list))
  X_train, Y_train = X[:step], Y[:step]
  x_val, y_val = X[step:-step], Y[step:-step]
  x_test, y_test = X[:-step], Y[:-step]
  return X, Y


def load_miy_dataset(filename):
  '''load miy dataset
  @brief: each line is formated as [[label, top, left, right, bottom], [...], ...] where
          label:int64, others:float32 in ratio format
  '''
  return

if __name__ == '__main__':
  raise Exception('Not permitted to launch dataloader.py directely')