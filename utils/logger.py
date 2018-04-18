import sys
import logging
from colorlog import ColoredFormatter

_LEVEL_BRIEF_NAME = {
  'DEBUG'    : 'D',
  'INFO'     : 'I',
  'WARNING'  : 'W',
  'ERROR'    : 'E',
  'FATAL'    : 'F',
  'CRITICAL' : 'C'
}

formatter_str = '%(asctime)s %(levelname)-8s [*] %(message)s'
uncolor_formatter = ColoredFormatter(formatter_str)
color_formatter = ColoredFormatter('%(log_color)s' + formatter_str)


_LEVEL_MAP = {
  'DEBUG'          : logging.DEBUG,
  'INFO'           : logging.INFO,
  'WARN'           : logging.WARN,
  'WARNING'        : logging.WARNING,
  'ERROR'          : logging.ERROR,
  'FATAL'          : logging.FATAL,
  'CRITICAL'       : logging.CRITICAL,
  'debug'          : logging.DEBUG,
  'info'           : logging.INFO,
  'warn'           : logging.WARN,
  'warning'        : logging.WARNING,
  'error'          : logging.ERROR,
  'fatal'          : logging.FATAL,
  'critical'       : logging.CRITICAL,
  '0'              : logging.DEBUG,
  '1'              : logging.INFO,
  '2'              : logging.WARN,
  '3'              : logging.ERROR,
  '4'              : logging.FATAL,
  '5'              : logging.CRITICAL,
  logging.DEBUG    : logging.DEBUG,
  logging.INFO     : logging.INFO,
  logging.WARN     : logging.WARN,
  logging.WARNING  : logging.WARNING,
  logging.ERROR    : logging.ERROR,
  logging.FATAL    : logging.FATAL,
  logging.CRITICAL : logging.CRITICAL,
}


class Logger(object):
  '''
  '''
  def __init__(self, logfile = None, colored = True, level = 'INFO'):
    '''
    '''
    self._logfile = logfile
    self._colored = colored
    self._level = _LEVEL_MAP[level]
    self._logger = logging.getLogger()
    self._setAll()
    return
    
  def _setAll(self):
    self._console_handler = logging.StreamHandler(sys.stderr)
    self.setColor(self._colored)
    self._logger.addHandler(self._console_handler)
    if self._logfile is not None:
      self.setLogFile(logfile)
      self._logger.addHandler(self._file_handler)
    self.setLevel(self._level)
    return
    
  def setColor(self, colored = True):
    self._colored = colored
    self._formatter = color_formatter if self._colored is True else uncolor_formatter
    self._console_handler.formatter = self._formatter
    return
    
  def setLogFile(self, logfile):
    if self._logfile is None:
      self._file_handler = logging.FileHandler(logfile)
      self._logger.addHandler(self._file_handler)
    self._file_handler.setFormatter(self._formatter)
    self._logfile = logfile
    return
    
  def setLevel(self, level = 'INFO'):
    self._level = _LEVEL_MAP[level]
    return self._logger.setLevel(self._level)
    
  def debug(self, x):
    return self._logger.debug(x)
    
  def info(self, x):
    return self._logger.info(x)
    
  def warn(self, x):
    return self._logger.warn(x)
    
  def warning(self, x):
    return self.warn(x)
    
  def error(self, x):
    return self._logger.error(x)
    
  def fatal(self, x):
    return self._logger.fatal(x)
    
  def critical(self, x):
    return self._logger.critical(x)


logger = Logger()
