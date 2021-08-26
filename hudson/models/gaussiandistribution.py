"""
Classes for design space models based on Gaussian distribution

Created on Feb 14, 2020
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

from logging import getLogger
from numpy import full, mean, std
from search.learningtechniques import Model

log = getLogger(__name__)
                                              

class GaussianDistribution(Model):
  """Gaussian distribution model
  
  """
  def __init__(self, *pargs, **kwargs):
    super(GaussianDistribution, self).__init__(*pargs, **kwargs)
    self.results = []
    self.mean = 0.0
    self.std_dev = 1.0
  
  def clear(self):
    """Remove all measurements in the dataset."""
    self.results = []

  def add_value(self, cfg, result):
    """Add a result to the dataset.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    """
    self.results.append(result)
  
  def train(self):
    """Train the model with the dataset."""
    if self.results:
      self.mean = mean(self.results)
      self.std_dev = std(self.results)
    else:
      self.mean = 0.0
      self.std_dev = 1.0

  def predict(self, cfg_vec):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configurations
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions
    """
    sample_cnt = len(cfg_vec)
    return full(sample_cnt, self.mean), full(sample_cnt, self.std_dev)
