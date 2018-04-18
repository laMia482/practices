import cv2
import copy
import numpy as np
from utils.logger import logger
logger.setLevel('debug')


class DataAugmentor(object):
  '''
  '''
  def __init__(self):
    '''
    '''
    self._funcs = []
    self._add_funcs()
    return
    
  def augmented(self, image):
    '''
    '''
    X = []
    for func in self._funcs:
      x = copy.deepcopy(func(image))
      X.append(x)
    return X
    
  def _add_funcs(self):
    '''
    '''
    self._funcs.append(self._fliplr)
    self._funcs.append(self._flipud)
    self._funcs.append(self._translate)
    self._funcs.append(self._rotate)
    self._funcs.append(self._set_zero)
    self._funcs.append(self._fix_brightness)
    self._funcs.append(self._add_salt_noise)
    self._funcs.append(self._add_gaussian_noise)
    self._funcs.append(self._transform2RGB)
    self._funcs.append(self._transform2HSV)
    return
    
  def _flip(self, image, code):
    '''flip image, 0 for up and down, 1 for left and right
    '''
    return cv2.flip(image, code)
    
  def _flipud(self, image):
    '''flip image up and down
    '''
    return self._flip(image, 0)
  
  def _fliplr(self, image):
    '''flip image left and right
    '''
    return self._flip(image, 1)
    
  def _translate(self, image, x = 0, y = 0):
    '''translate image with x(> 0 for right) and y(> 0 for down)
    '''
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted
    
  def _rotate(self, image, angle = 45, center = None, scale = 1.0):
    '''rotate image
    '''
    h, w = image.shape[:2]
    if center is None:
      center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
    
  def _set_zero(self, image, xbegin = 0, xend = 0, ybegin = 0, yend = 0):
    '''set image[xbegin: xend, ybegin: yend] = 0
    '''
    out = copy.deepcopy(image)
    out[xbegin: xend, ybegin: yend] = 0
    return out
    
  def _fix_brightness(self, image, pix = 25, scale = 1.0):
    '''multiple by scale and add pix to each pixel of image
    '''
    out = copy.deepcopy(image)
    out = (out * scale).astype(image.dtype) + pix
    return out
    
  def _add_salt_noise(self, image, coe = 0.5):
    '''add salt noise
    '''
    pl = int(image.shape[0] * coe * image.shape[1] * coe)
    px = np.random.choice(image.shape[0], pl)
    py = np.random.choice(image.shape[1], pl)
    v = np.random.choice(256, [pl, image.shape[2]])
    out = copy.deepcopy(image)
    out[px, py] = v
    return out
    
  def _add_gaussian_noise(self, image, coe = 0.5, u = 128, sigma = 42):
    '''add Gaussian noise
    '''
    pl = int(image.shape[0] * coe * image.shape[1] * coe)
    px = np.random.choice(image.shape[0], pl)
    py = np.random.choice(image.shape[1], pl)
    v = np.random.normal(u, sigma, [pl, image.shape[2]])
    out = copy.deepcopy(image)
    out[px, py] = v
    return out
    
  def _transform2RGB(self, image):
    '''transform image to BGR or RGB
    '''
    out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return out
    
  def _transform2HSV(self, image):
    '''transform image to BGR or HSV
    '''
    out = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return out
    