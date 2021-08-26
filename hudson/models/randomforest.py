"""
Random Forest Model

Author: Hans Giesen (giesen@seas.upenn.edu)
"""

# Amount of time that we would like to spend on training in seconds
TRAINING_TIME = 60.0
# Factor for slowing down updates of number of trees used by random forest
UPDATE_FACTOR = 0.2

import logging
import numpy as np
import sys
import time

from search.learningtechniques import Model
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger(__name__)


class RandomForest(Model):
  """Random Forest model

  Parameters
  ----------
  forest_cnt : int
    Number of forests
  tree_cnt : int
    Number of trees in a forest
  regressors : list of RandomForestRegressor
    Random forests
  invalid_cfg_vecs : list of list of float
    Configurations for which we have measurements
  valid_cfg_vecs : list of list of float
    Configurations for which we do not have measurements
  results : list of float
    Measurements for each configuration
  """

  def __init__(self, *pargs, **kwargs):
    super(RandomForest, self).__init__(*pargs, **kwargs)
    self.forest_cnt = 8
    self.tree_cnt = 100
    self.regressors = []
    self.invalid_cfg_vecs = []
    self.valid_cfg_vecs = []
    self.results = []

  def clear(self):
    """Remove all measurements in the dataset."""
    self.regressors = []
    self.invalid_cfg_vecs = []
    self.valid_cfg_vecs = []
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
    cfg_vec = self.get_vec(cfg.data)
    if np.isinf(result) or np.isnan(result):
      self.invalid_cfg_vecs.append(cfg_vec)
    else:
      self.valid_cfg_vecs.append(cfg_vec)
      self.results.append(result)

  def train(self):
    """Train the model with the dataset."""
    if len(self.results) == 0:
      return
    
    log.info("Retraining the random forest with %d samples and %d trees.",
             len(self.results), self.tree_cnt)
    
    # Use the average for configurations that have invalid results.  We need
    # these fake points to assure that the uncertainty is reduced.  Otherwise,
    # random forest regression would keep sampling the same points.
    fake_results = [np.mean(self.results)] * len(self.invalid_cfg_vecs)
    cfg_vecs = self.valid_cfg_vecs + self.invalid_cfg_vecs
    results = self.results + fake_results

    start = time.time()
    for forest in range(self.forest_cnt):
      regressor = RandomForestRegressor(n_estimators=self.tree_cnt)
      regressor.fit(cfg_vecs, results)
      self.regressors.append(regressor)
    training_time = time.time() - start

    # Compute the number of trees needed to reach the desired training time.
    tree_cnt = self.tree_cnt * TRAINING_TIME / training_time

    # Update the number of trees slowly to avoid instabilities due to
    # measurement inaccuracies.
    self.tree_cnt = int((1 - UPDATE_FACTOR) * self.tree_cnt +
                        UPDATE_FACTOR * tree_cnt)
    
  def predict(self, cfg_vecs, return_all=False):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configurations
    return_all : bool
      True if predictions and standard deviations from all fidelities must be
      returned
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions
    list of dict
      All predictions and errors.  Each dictionary has the following members:

        mean : float
          Predictions
        std_dev : float
          Standard deviation of predictions
    """
    if len(self.regressors) == 0:
      dims = cfg_vecs.shape[0]
      means = np.zeros(dims)
      std_devs = np.ones(dims)
    else:
      preds = [regressor.predict(cfg_vecs) for regressor in self.regressors]
      means = np.mean(preds, axis=0)
      std_devs = np.std(preds, axis=0)

    if return_all:
      return means, std_devs, [{'mean': means, 'std_dev': std_devs}]
    else:
      return means, std_devs

