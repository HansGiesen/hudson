"""
Classes for random search with error avoidance

Created on Jan 26, 2020
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

from .learningtechniques import LearningTechnique
from numpy.random import normal, uniform
from opentuner.search.technique import SearchTechnique
import logging
from random import choice

log = logging.getLogger(__name__)

# This factor affects the number of random points considered.  We want to have
# a reasonable chance at finding at least one good configuration for our list
# of candidates, such that bad candidates are more likely to be ignored.
FACTOR = 10.0


class RandomSearch(SearchTechnique):
  """Random search technique
  """
  def desired_configuration(self):
    """Suggest a new configuration to evaluate.
    
    Returns
    -------
    Configuration
      Suggested configuration
    """
    interface = self.driver.tuning_run_main.measurement_driver.interface
    if not getattr(interface, "downsample", False):
      cfg = self.manipulator.random()
    else:
      # Use only samples that are in the downsampled design space.
      if interface.cfgs is None:
        interface.create_design_space()
      cfg = choice(interface.cfgs)
    return self.driver.get_configuration(cfg)


class RandomSearchAvoidErrors(LearningTechnique):
  """Random search technique with error avoidance

  Attributes
  ----------
  result_cnt : int
    Number of results obtained so far
  """
  def __init__(self, model, *pargs, **kwargs):
    super(RandomSearchAvoidErrors, self).__init__([model], *pargs, **kwargs)
    self.result_cnt = 0
  
  def handle_requested_result(self, result):
    """This callback is invoked by the search driver to report new results.
    
    Parameters
    ----------
    result : Result
      Result
    """
    success_prob = 1.0 if result.state == 'OK' else 0.0
    self.models[0].add_value(result.configuration, success_prob)
    self.result_cnt += 1
    self.retrain = True

  def select_configuration(self):
    """Suggest a new configuration to evaluate.
    
    Returns
    -------
    Configuration
      Suggested configuration
    """
    total = 0.0
    cfgs = []
    values = []
    for i in range(max(int(FACTOR * self.result_cnt), 1)):
      cfg = self.manipulator.random()
      cfg_vec = self.manipulator.get_vec(cfg)
      pred, error = self.models[0].predict(cfg_vec)
      value = normal(pred, error)
      # Clipping is necessary for Gaussian process.  Should probably replace
      # it with random forest.
      if value < 0.0:
        value = 0.0
      elif value > 1.0:
        value = 1.0
      total += value
      cfgs.append(cfg)
      values.append(value)

    sample = uniform(0.0, total)

    prob = 0.0
    for value, cfg in zip(values, cfgs):
      prob += value
      if prob >= sample:
        break

    return self.driver.get_configuration(cfg)

