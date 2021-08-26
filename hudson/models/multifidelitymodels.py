"""
Classes for models that support multiple fidelities

Created on Jul 18, 2019
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

from logging import getLogger
from numpy import array, clip, isinf, sqrt
from scipy.optimize import minimize
from search.learningtechniques import Model
from sklearn.linear_model import LinearRegression, RANSACRegressor

log = getLogger(__name__)


class MultiFidelityModel(Model):
  """Class that supports multiple fidelities

  Parameters
  ----------
  models : list of Model
    Model for each fidelity
  results : list of list of (Configuration, float)
    This data structure consists of a list with a list for every fidelity.
    Each element in the list corresponds with a measurement.
  update_fidelity : int
    The lowest fidelity for which new samples were added since last training

  """
  def __init__(self, metric, models, *pargs, **kwargs):
    super(MultiFidelityModel, self).__init__(metric, *pargs, **kwargs)
    self.models = models
    self.max_fidelity = len(models)
    self.results = [[] for fidelity in range(self.max_fidelity)]
    self.update_fidelity = 0

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
      Search driver
    """
    super(MultiFidelityModel, self).set_driver(driver)
    for model in self.models:
      model.set_driver(driver)

  def add_value(self, cfg, result):
    """Add a result to the model.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    """
    self.results[cfg.fidelity - 1].append((cfg, result))
    self.update_fidelity = min(self.update_fidelity, cfg.fidelity - 1)

  def is_ready(self):
    """Return True if the model has enough data to be reliable.

    Returns
    -------
    bool
      True if the model is reliable
    """
    return self.models[0].is_ready()

  def train(self):
    """Train the model with the dataset."""
    for fidelity in range(self.update_fidelity, self.max_fidelity):
      model = self.models[fidelity]
      model.clear()
      if fidelity > 0:
        results = self.results[fidelity]
        if results:
          cfg_vecs = array([self.get_vec(result[0].data) for result in results])
          preds = self.predict(cfg_vecs, fidelity=fidelity)[0]
          for (cfg, result), pred in zip(results, preds):
            model.add_value(cfg, result, pred=pred)
      else:
        for cfg, result in self.results[fidelity]:
          model.add_value(cfg, result)
      model.train()
    self.update_fidelity = len(self.models)

#    for fidelity, model in enumerate(self.models):
#      if isinstance(model, ScaledSumModel):
#        log.info('The scaling coefficient of %s at fidelity %i is %f',
#                 self.metric, fidelity + 1, model.coef)

  def predict(self, cfg_vecs, fidelity=0, return_all=False):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration
    fidelity : int
      Desired fidelity level of prediction
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

    Notes
    -----
    The model must be trained before this function is called.
    """
    if fidelity is 0:
      fidelity = len(self.models)

    all_preds = []
    for level, model in enumerate(self.models[: fidelity]):
      if level == 0:
        means, std_devs = model.predict(cfg_vecs)
      else:
        means, std_devs = model.predict(cfg_vecs, preds = means,
                                        errors = std_devs)
      all_preds.append({'mean': means, 'std_dev': std_devs})

    if return_all:
      return means, std_devs, all_preds
    else:
      return means, std_devs
    
  def select_configuration(self):
    """Suggest a new configuration to evaluate to initialize the model.
    
    Returns
    -------
    dict or None
      Configuration.  None is returned if initialization has completed.
    """
    for model in self.models:
      cfg = model.select_configuration()
      if cfg is not None:
        break
    return cfg

  def sample_models(self, cfg_vecs):
    """Samples each of the submodels for debug purposes.

    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration

    Returns
    -------
    list of tuple of arrays
      A list with expected values and standard deviations for each fidelity
    """
    return [model.predict(cfg_vecs) for model in self.models]


class CLoModel(Model):
  """Model to support multi-fidelity model of C. Lo et al.

  This model computes predictions by multiplying a given prediction with a
  constant and adding a prediction provided by an encapsulated model.  This
  model is supposed to be equal to a stage in the original multi-fidelity model
  by C. Lo et al.

  Parameters
  ----------
  coef : float
    Scaling coefficient
  model : Model
    Model
  cfgs : list of dict
    Configurations for which we have measurements
  preds : list of float
    Predictions from previous fidelity for each configuration
  results : list of float
    Measurements for each configuration

  """
  def __init__(self, model, *pargs, **kwargs):
    super(CLoModel, self).__init__(*pargs, **kwargs)
    self.coef = 1.0
    self.model = model
    self.cfgs = []
    self.preds = []
    self.results = []

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
    """
    super(CLoModel, self).set_driver(driver)
    self.model.set_driver(driver)

  def clear(self):
    """Remove all measurements in the dataset."""
    self.cfgs = []
    self.preds = []
    self.results = []

  def add_value(self, cfg, result, pred = 0):
    """Add a result to the model.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    pred : float
      Prediction of result
    """
    self.cfgs.append(cfg)
    self.preds.append(pred)
    self.results.append(result)
  
  def train(self):
    """Train the model with the dataset."""

    def compute_error(params, self):
      """Compute mean squared error of model for a given coefficient.
      
      Parameters
      ----------
      params : list of float
        Scaling coefficient
      self : CLoModel
        Model to be optimized
      """
      self.train_submodel(params[0])
      self.coef = params[0]
      error = 0.0
      for cfg, pred, result in zip(self.cfgs, self.preds, self.results):
        cfg_vec = self.get_vec(cfg.data)
        error += (self.predict(cfg_vec, pred = pred)[0] - result) ** 2
      return error

    self.coef = minimize(compute_error, 1.0, (self, ),
                         bounds = ((0.0, None), ), tol = 1e-2).x[0]
    self.train_submodel(self.coef)

  def predict(self, cfg_vecs, preds = 0, errors = 0):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration
    preds : array of shape (no of samples, no of features)
      Predictions of result
    errors : array of shape (no of samples, no of features)
      Errors in predictions of result
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions

    Notes
    -----
    The model must be trained before this function is called.
    """
    model_preds = self.model.predict(cfg_vecs)
    final_preds = self.coef * preds + model_preds[0]
    final_errors = sqrt((self.coef * errors) ** 2.0 + model_preds[1] ** 2.0)
    return final_preds, final_errors

  def select_configuration(self):
    """Suggest a new configuration to evaluate to initialize the model.
       
    Returns
   -------
    dict or None
      Configuration.  None is returned if initialization has completed.
    """
    return self.model.select_configuration()

  def train_submodel(self, coef):
    """Train the encapsulated model with the current dataset.
    
    Parameters
    ----------
    coef : float
      Scaling coefficient
    """
    self.model.clear()
    for cfg, pred, result in zip(self.cfgs, self.preds, self.results):
      diff = result - coef * pred
      self.model.add_value(cfg, diff)
    self.model.train()


class ScaledSumModel(Model):
  """Scaled sum model
  
  This model computes predictions by adding a given prediction to a scaled
  prediction from the encapsulated model.  This model is identical to a stage
  in the multi-fidelity model by C. Lo et al. except that they perform a joint
  optimization to obtain good values for the scaling coefficient.

  Parameters
  ----------
  coef : float
    Scaling coefficient
  model : Model
    Model
  cfgs : list of dict
    Configurations for which we have measurements
  preds : list of float
    Predictions from previous fidelity for each configuration
  results : list of float
    Measurements for each configuration

  """
  def __init__(self, model, *pargs, **kwargs):
    super(ScaledSumModel, self).__init__(*pargs, **kwargs)
    self.coef = 1.0
    self.model = model
    self.cfgs = []
    self.preds = []
    self.results = []

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
    """
    super(ScaledSumModel, self).set_driver(driver)
    self.model.set_driver(driver)

  def clear(self):
    """Remove all measurements in the dataset."""
    self.cfgs = []
    self.preds = []
    self.results = []

  def add_value(self, cfg, result, pred = 0):
    """Add a result to the model.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    pred : float
      Prediction of result
    """
    self.cfgs.append(cfg)
    self.preds.append(pred)
    self.results.append(result)

  def train(self):
    """Train the model with the dataset.
    """
    results = array(self.results)
    indices = ~isinf(results)
    if indices.sum() > 0:
      preds = array(self.preds).reshape(-1, 1)
      try:
        regression = RANSACRegressor()
        regression.fit(preds[indices, :], results[indices])
        self.coef = regression.estimator_.coef_[0]
      except:
        regression = LinearRegression()
        regression.fit(preds[indices, :], results[indices])
        self.coef = regression.coef_[0]
    else:
      self.coef = 1.0

    self.model.clear()
    for cfg, pred, result in zip(self.cfgs, self.preds, self.results):
      diff = result - self.coef * pred
      self.model.add_value(cfg, diff)
    self.model.train()

  def predict(self, cfg_vecs, preds = 0, errors = 0):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration
    preds : array of shape (no of samples, )
      Predictions of result
    errors : array of shape (no of samples, )
      Errors in predictions of result
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions

    Notes
    -----
    The model must be trained before this function is called.
    """
    if self.model.is_ready():
      model_preds = self.model.predict(cfg_vecs)
      final_preds = self.coef * preds + model_preds[0]
      final_errors = sqrt((self.coef * errors) ** 2.0 + model_preds[1] ** 2.0)
      return final_preds, final_errors
    else:
      return preds, errors

  def select_configuration(self):
    """Suggest a new configuration to evaluate to initialize the model.
       
    Returns
   -------
    dict or None
      Configuration.  None is returned if initialization has completed.
    """
    return self.model.select_configuration()


class SumModel(Model):
  """Sum model

  This model computes predictions by adding a given prediction to one provided
  by an encapsulated model.  This model is similar to the multi-fidelity model
  by C. Lo et al. except that it does not discount predictions from lower
  fidelities because we observed that the discount factor after training was
  always 1.

  Parameters
  ----------
  model : Model
    Model
  cfgs : list of dict
    Configurations for which we have measurements
  results : list of float
    Measurements for each configuration

  """
  def __init__(self, model, *pargs, **kwargs):
    super(SumModel, self).__init__(*pargs, **kwargs)
    self.model = model
    self.cfgs = []
    self.results = []

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
    """
    super(SumModel, self).set_driver(driver)
    self.model.set_driver(driver)

  def clear(self):
    """Remove all measurements in the dataset."""
    self.cfgs = []
    self.results = []

  def add_value(self, cfg, result, pred = 0):
    """Add a result to the model.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    pred : float
      Prediction of result
    """
    self.cfgs.append(cfg)
    self.results.append(result - pred)
  
  def train(self):
    """Train the model with the dataset."""

    self.model.clear()
    for cfg, result in zip(self.cfgs, self.results):
      self.model.add_value(cfg, result)
    self.model.train()

  def predict(self, cfg_vecs, preds = 0, errors = 0):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration
    preds : array of shape (no of samples, )
      Predictions of result
    errors : array of shape (no of samples, )
      Errors in predictions of result
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions

    Notes
    -----
    The model must be trained before this function is called.
    """
    model_preds = self.model.predict(cfg_vecs)
    final_preds = preds + model_preds[0]
    final_errors = sqrt(errors ** 2.0 + model_preds[1] ** 2.0)
    return final_preds, final_errors

  def select_configuration(self):
    """Suggest a new configuration to evaluate to initialize the model.
       
    Returns
   -------
    dict or None
      Configuration.  None is returned if initialization has completed.
    """
    return self.model.select_configuration()


class XorModel(SumModel):
  """Xor model

  This model was designed to make binary predictions.  It computes predictions
  by xor-ing a given prediction to one provided by an encapsulated model.

  Parameters
  ----------
  model : Model
    Model
  cfgs : list of dict
    Configurations for which we have measurements
  results : list of float
    Measurements for each configuration

  """
  def add_value(self, cfg, result, pred = 0):
    """Add a result to the model.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    pred : float
      Prediction of result
    """
    self.cfgs.append(cfg)
    prob = (1.0 - pred) * result + pred * (1.0 - result)
    self.results.append(prob)
  
  def predict(self, cfg_vecs, preds = 0, errors = 0):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration
    preds : array of shape (no of samples, )
      Predictions of result
    errors : array of shape (no of samples, )
      Errors in predictions of result
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions

    Notes
    -----
    The model must be trained before this function is called.
    """
    model_preds = self.model.predict(cfg_vecs)[0]
    final_preds = (1.0 - preds) * model_preds + \
                  preds * (1.0 - model_preds)
    final_errors = sqrt(final_preds * (1.0 - final_preds))
    return final_preds, final_errors


class ClippedSumModel(SumModel):
  """Clipped sum model

  This model was designed to make binary predictions.  It computes predictions
  by summing a given prediction to one provided by an encapsulated model.

  Parameters
  ----------
  model : Model
    Model
  cfgs : list of dict
    Configurations for which we have measurements
  results : list of float
    Measurements for each configuration

  """
  def add_value(self, cfg, result, pred = 0):
    """Add a result to the model.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    pred : float
      Prediction of result
    """
    self.cfgs.append(cfg)
    prob = 1.0 if pred < 0.5 and result >= 0.5 else 0.0
    self.results.append(prob)
  
  def predict(self, cfg_vecs, preds = 0, errors = 0):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration
    preds : array of shape (no of samples, )
      Predictions of result
    errors : array of shape (no of samples, )
      Errors in predictions of result
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions

    Notes
    -----
    The model must be trained before this function is called.
    """
    model_preds = self.model.predict(cfg_vecs)[0]
    final_preds = clip(preds + model_preds, 0.0, 1.0)
    final_errors = sqrt(final_preds * (1.0 - final_preds))
    return final_preds, final_errors


class FilterModel(Model):
  """Filter model

  Parameters
  ----------
  model : Model
    Model

  """
  def __init__(self, model, *pargs, **kwargs):
    super(FilterModel, self).__init__(*pargs, **kwargs)
    self.model = model

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
    """
    super(FilterModel, self).set_driver(driver)
    self.model.set_driver(driver)

  def clear(self):
    """Remove all measurements in the dataset."""
    self.model.clear()

  def add_value(self, cfg, result, pred=0):
    """Add a result to the model.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    pred : float
      Prediction of result
    """
    self.model.add_value(cfg, result)
  
  def train(self):
    """Train the model with the dataset."""
    self.model.train()

  def predict(self, cfg_vecs, preds=0, errors=0):
    """Make a prediction for the given configuration.
    
    Parameters
    ----------
    cfg_vecs : array of shape (no of samples, no of features)
      Configuration
    preds : array of shape (no of samples, )
      Predictions of result
    errors : array of shape (no of samples, )
      Errors in predictions of result
    
    Returns
    -------
    array of shape (no of samples, )
      Expected values of predictions
    array of shape (no of samples, )
      Standard deviations of predictions

    Notes
    -----
    The model must be trained before this function is called.
    """
    if self.model.is_ready():  
      model_preds = self.model.predict(cfg_vecs)[0]
      final_preds = preds + (1.0 - preds) * model_preds
      final_errors = sqrt(final_preds * (1.0 - final_preds))
      # There is no point in filtering if all points are filtered out.
      if final_preds.min() == 1.0:
        return preds, errors
      else:
        return final_preds, final_errors
    else:
      return preds, errors

  def select_configuration(self):
    """Suggest a new configuration to evaluate to initialize the model.
       
    Returns
   -------
    dict or None
      Configuration.  None is returned if initialization has completed.
    """
    return self.model.select_configuration()

