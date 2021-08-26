"""
Machine Learning Search Techniques

Base classes for search techniques that use machine-learning models

Author: Hans Giesen (giesen@seas.upenn.edu)
"""

import abc, logging

from opentuner.search.technique import SearchTechnique

log = logging.getLogger(__name__)


class LearningTechnique(SearchTechnique):
  """Abstract base class for machine-learning search techniques
    
  Parameters
  ----------
  models : list of Model objects
    Model(s) upon which the search technique is based
  retrain : bool
    True if the models must be retrained before making new predictions
  max_fidelity : int
    Number of fidelity levels

  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, models, *pargs, **kwargs):
    super(LearningTechnique, self).__init__(*pargs, **kwargs)
    self.models = models
    self.retrain = True
    self.max_fidelity = max(model.max_fidelity for model in self.models)

  def handle_requested_result(self, result):
    """This callback is invoked by the search driver to report new results.
    
    Parameters
    ----------
    result : Result
      Result
    """
    for model in self.models:
      if model.metric != 'error_prob':
        model.add_result(result)
      else:
        error_prob = 0.0 if result.state == 'OK' else 1.0
        model.add_value(result.configuration, error_prob)
    
    self.retrain = True

  def desired_configuration(self, **kwargs):
    """Suggest a new configuration to evaluate.
   
    Parameters
    ----------
    fidelity : int
      Desired fidelity.  This parameter is optional.  If left out, any fidelity
      can be returned.

    Returns
    -------
    Configuration
      Suggested configuration
    """
    for model in self.models:
      cfg = model.select_configuration()
      if cfg is not None:
        break
      
    if cfg == None:
      if self.retrain:
        for model in self.models:
          model.train()
        self.retrain = False
      cfg = self.select_configuration(**kwargs)

    return cfg

  @abc.abstractmethod
  def select_configuration(self):
    """Callback to suggest a new configuration to evaluate.

    Returns
    -------
    Configuration
      Suggested configuration
    """

  def start_batch(self):
    """Callback called before a new batch of configurations is requested.
    """
    if self.retrain:
      for model in self.models:
        model.train()
      self.retrain = False

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
    """
    super(LearningTechnique, self).set_driver(driver)
    for model in self.models:
      model.set_driver(driver)


class Model(object):
  """Abstract base class for machine-learning models
  
  Parameters
  ----------
  metric : str
    Metric that is modeled
  driver : SearchDriver
    Search driver
  manipulator : ConfigurationManipulator
    Configuration manipulator
  max_fidelity : int
    Number of fidelity levels

  """
  __metaclass__ = abc.ABCMeta
  
  def __init__(self, metric = None):
    self.metric = metric
    self.driver = None
    self.manipulator = None
    self.max_fidelity = 1
 
  def clear(self):
    """Remove all measurements in the dataset."""
    pass

  def add_result(self, result):
    """Add a result to the dataset.
    
    Parameters
    ----------
    result : Result
      Result
    """
    self.add_value(result.configuration, getattr(result, self.metric))

  def add_value(self, cfg, result):
    """Add a result to the dataset.
    
    Parameters
    ----------
    cfg : Configuration
      Configuration
    result : float
      Result
    """
    pass

  def train(self):
    """Train the model with the dataset."""
    pass

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
    """
    self.driver = driver
    self.manipulator = driver.manipulator

  @abc.abstractmethod
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

    Notes
    -----
    The model must be trained before this function is called.
    """

  def select_configuration(self):
    """Suggest a new configuration to evaluate to initialize the model.
    
    Returns
    -------
    Configuration
      Suggested configuration.  None is returned if initialization has completed.
    """
    return None

  def get_vec(self, cfg):
    """Return a vector representation of a configuration
    
    Parameters
    ----------
    cfg : dict
      Configuration

    Returns
    -------
    list of float
      Vector representation
    """
    return self.manipulator.get_vec(cfg)
