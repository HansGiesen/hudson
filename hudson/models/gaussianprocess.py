"""
Classes for design space models based on Gaussian processes

Created on Jul 8, 2019
@author: Hans Giesen (giesen@seas.upenn.edu)

Copyright 2019 Xilinx, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from search.learningtechniques import Model
from logging import getLogger
import math
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
import warnings

log = getLogger(__name__)
                                              

class GaussianProcess(Model):
  """Gaussian process model

  Attributes
  ----------
  gp : GaussianProcessRegressor
    Underlying Gaussian process regression implementation
  invalid_cfg_vecs : list of list of float
    Configurations for which we have measurements
  valid_cfg_vecs : list of list of float
    Configurations for which we do not have measurements
  results : list of float
    Measurements for each configuration
  output_offset : float
    Offset added to results as part of normalization
  output_scaling : float
    Factor multiplied with results as part of normalization
  input_offset : array of shape (no of params, )
    Offset deducted from input as part of normalization
  input_scaling : array of shape (no of params, )
    Factor removed input as part of normalization
  binary : bool
    Whether measurements have a binary distribution
  random_state : RandomState
    Random state
  debug : bool
    Print intermediate results useful for comparing with Jupyter model.
  ready_thres : bool
    Number of samples that the model must have to be considered reliable
  """

  debug = False
  ready_thres = 8

  def __init__(self, *pargs, **kwargs):
    self.relax_bounds = kwargs.pop('relax_bounds', False)
    self.restarts = kwargs.pop('restarts', 2)
    self.binary = kwargs.pop('binary', False)
    self.avoid_errors = kwargs.pop('avoid_errors', True)
    self.random_state = kwargs.pop('random_state', None)
    super(GaussianProcess, self).__init__(*pargs, **kwargs)
    self.gp = None
    self.valid_cfg_vecs = []
    self.invalid_cfg_vecs = []
    self.results = []
    self.input_offset = None
    self.input_scaling = None
    self.output_offset = None
    self.output_scaling = None

  def clear(self):
    """Remove all measurements in the dataset."""
    self.gp = None
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
    cfg_vec = np.array(self.get_vec(cfg.data))
    scaled_vec = self.scale_vecs(cfg_vec).tolist()
    if math.isinf(result) or math.isnan(result):
      self.invalid_cfg_vecs.append(scaled_vec)
    else:
      self.valid_cfg_vecs.append(scaled_vec)
      self.results.append(result)
 
  def is_ready(self):
    """Return True if the model has enough data to be reliable.

    Returns
    -------
    bool
      True if the model is reliable
    """
    return len(self.valid_cfg_vecs) >= self.ready_thres

  def train(self):
    """Train the model with the dataset."""
    if len(self.results) == 0:
      return

    if not self.gp:
      # This optimizer is equal to the default one except for a limitation on
      # the number of iterations.
      def optimizer(obj_func, initial_theta, bounds):
        result = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True,
                          bounds=bounds, options={'maxiter': 100})
        return result.x, result.fun

      dims = self.input_offset.shape[0]
      if self.relax_bounds:
        const_kernel = ConstantKernel(1.0, (1e-10, 1e20))
        rbf_kernel = Matern(np.ones(dims), [(1e-10, 1e10)] * dims, 2.5)
      else:
        const_kernel = ConstantKernel(1.0, (1e-2, 1e3))
        rbf_kernel = Matern(np.ones(dims), [(1e-2, 1e2)] * dims, 2.5)
      noise_kernel = WhiteKernel(noise_level=1e-10,
                                 noise_level_bounds=(1e-20, 1e10))
      kernel = const_kernel * rbf_kernel + noise_kernel
  
      self.gp = GaussianProcessRegressor(kernel=kernel,
                                         n_restarts_optimizer=self.restarts,
                                         optimizer=optimizer, alpha=0.0,
                                         random_state=self.random_state)

    if not self.binary:
      self.output_offset = np.mean(self.results)
      scaling = np.std(self.results)
      self.output_scaling = 1.0 if scaling == 0.0 else scaling
    else:
      self.output_offset = 1.0 if np.mean(self.results) > 0.5 else 0.0
      self.output_scaling = 1.0
    results = (np.array(self.results) - self.output_offset) / self.output_scaling

    if self.avoid_errors:
      # Use the average for configurations that failed to build.  We need these
      # fake points to assure that the uncertainty is reduced.  Otherwise,
      # Bayesian optimization would keep sampling the same points.
      fake_result = 0.0
    else:
      # We want to discourage the tuner from selecting configurations that
      # failed to build, but the Gaussian Process cannot handle extreme results,
      # so let's settle for a result that is 2 standard deviations above the
      # average.
      fake_result = 2.0
    fake_results = [fake_result] * len(self.invalid_cfg_vecs)
    cfg_vecs = self.valid_cfg_vecs + self.invalid_cfg_vecs
    results = results.tolist() + fake_results

    if self.debug:
      print("Output scaling: mean = {}, std_dev = {}"
            .format(self.output_offset, self.output_scaling))
      print("Train: X = {}, Y = {}".format(cfg_vecs, results))

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.gp.fit(cfg_vecs, results)

    if self.debug:
      print("Theta: {}".format(self.gp.kernel_.theta.tolist()))

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
    if self.gp is None:
      sample_cnt = cfg_vecs.shape[0]
      pred_means = np.zeros(sample_cnt)
      if self.binary:
        pred_std_devs = np.zeros(sample_cnt)
      else:
        pred_std_devs = np.ones(sample_cnt)

    else:
      scaled_vecs = self.scale_vecs(cfg_vecs)

      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        means, std_devs = self.gp.predict(scaled_vecs, return_std = True)
      
      pred_means = self.output_offset + self.output_scaling * means

      if self.binary:
        pred_means = np.clip(pred_means, 0.0, 1.0)
        pred_std_devs = np.sqrt(pred_means * (1.0 - pred_means))
      else:
        pred_std_devs = self.output_scaling * std_devs

    if return_all:
      all_preds = [{'mean': pred_means, 'std_dev': pred_std_devs}]
      return pred_means, pred_std_devs, all_preds
    else:
      return pred_means, pred_std_devs

  def scale_vecs(self, cfg_vecs):
    """Scale a configuration vector to the unit scale.
   
    We need to scale the configuration vectors supplied to the Gaussian process
    because we are using fixed bounds on the kernel parameters.

    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configurations

    Returns
    -------
    array of shape (no of samples, no of features)
      Scaled configuration vectors
    """
    if self.input_offset is None:
      bounds = np.array(self.manipulator.get_bounds())
      self.input_offset = bounds[:, 0]
      upper_bounds = bounds[:, 1]
      self.input_scaling = upper_bounds - self.input_offset
    return (cfg_vecs - self.input_offset) / self.input_scaling


class BinaryGaussianProcess(Model):
  """Gaussian process model for binary classification

  Attributes
  ----------
  gp : GaussianProcessRegressor
    Underlying Gaussian process regression implementation
  cfg_vecs : list of dict
    Configurations for which we have measurements
  results : list of float
    Measurements for each configuration
  input_offset : array of shape (no of params, )
    Offset deducted from input as part of normalization
  input_scaling : array of shape (no of params, )
    Factor removed input as part of normalization
  classes : set of float
    Each of the distinct measurement values that has been encountered so far
  seed : int
    Random seed.  None if random seed should not be set.
  debug : bool
    Print intermediate results useful for comparing with Jupyter model.
  """

  seed = None
  debug = False

  def __init__(self, *pargs, **kwargs):
    self.relax_bounds = kwargs.pop('relax_bounds', False)
    self.restarts = kwargs.pop('restarts', 2)
    super(BinaryGaussianProcess, self).__init__(*pargs, **kwargs)
    self.gp = None
    self.cfg_vecs = []
    self.results = []
    self.input_offset = None
    self.input_scaling = None
    self.classes = set()

  def clear(self):
    """Remove all measurements in the dataset."""
    self.gp = None
    self.cfg_vecs = []
    self.results = []
    self.classes = set()

  def add_value(self, cfg, result):
    """Add a result to the dataset.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result.  This can be a binary value or a probability.  Probabilities are
      rounded to the nearest binary value.
    """
    if math.isinf(result) or math.isnan(result):
      return
    cfg_vec = np.array(self.get_vec(cfg.data))
    self.cfg_vecs.append(self.scale_vecs(cfg_vec).tolist())
    result = 1.0 if result >= 0.5 else 0.0
    self.results.append(result)
    self.classes.add(result)
  
  def train(self):
    """Train the model with the dataset."""
    if len(self.results) == 0:
      return

    if len(self.classes) < 2:
      return

    if not self.gp:
      # This optimizer is equal to the default one except for a limitation on
      # the number of iterations.
      def optimizer(obj_func, initial_theta, bounds):
        result = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True,
                          bounds=bounds, options={'maxiter': 100})
        return result.x, result.fun

      dims = self.input_offset.shape[0]
      if self.relax_bounds:
        const_kernel = ConstantKernel(1.0, (1e-10, 1e20))
        rbf_kernel = Matern(np.ones(dims), [(1e-10, 1e10)] * dims, 2.5)
      else:
        const_kernel = ConstantKernel(1.0, (1e-2, 1e3))
        rbf_kernel = Matern(np.ones(dims), [(1e-2, 1e2)] * dims, 2.5)
      kernel = const_kernel * rbf_kernel
  
      self.gp = GaussianProcessClassifier(kernel=kernel,
                                          n_restarts_optimizer=self.restarts,
                                          optimizer=optimizer,
                                          random_state=self.seed)

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      self.gp.fit(self.cfg_vecs, self.results)

    if self.debug:
      print("Theta: {}".format(self.gp.kernel_.theta.tolist()))

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
    sample_cnt = cfg_vecs.shape[0]
    if len(self.classes) < 2:
      pred = list(self.classes)[0] if self.results else 0.5
      std_dev = np.sqrt((1.0 - pred) * pred)
      preds = np.full(sample_cnt, pred)
      std_devs = np.full(sample_cnt, std_dev)
      if return_all:
        return preds, std_devs, [{'mean': preds, 'std_dev': std_devs}]
      else:
        return preds, std_devs

    scaled_vecs = self.scale_vecs(cfg_vecs)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      pred_means = self.gp.predict_proba(scaled_vecs)[:, 1]

    pred_std_devs = np.sqrt(pred_means * (1.0 - pred_means))
    if return_all:
      all_preds = [{'mean': pred_means, 'std_dev': pred_std_devs}]
      return pred_means, pred_std_devs, all_preds
    else:
      return pred_means, pred_std_devs

  def scale_vecs(self, cfg_vecs):
    """Scale a configuration vector to the unit scale.
   
    We need to scale the configuration vectors supplied to the Gaussian process
    because we are using fixed bounds on the kernel parameters.

    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configurations

    Returns
    -------
    array of shape (no of samples, no of features)
      Scaled configuration vectors
    """
    if self.input_offset is None:
      bounds = np.array(self.manipulator.get_bounds())
      self.input_offset = bounds[:, 0]
      upper_bounds = bounds[:, 1]
      self.input_scaling = upper_bounds - self.input_offset
    return (cfg_vecs - self.input_offset) / self.input_scaling
