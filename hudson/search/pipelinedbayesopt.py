"""
Classes for Bayesian optimization for build pipeline

Created on Jul 9, 2019
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

from .learningtechniques import LearningTechnique
from logging import getLogger
import math
import numpy as np
from numpy.linalg import norm
from opentuner.resultsdb.models import (DebugInfo, BayesIteration, BayesCost,
   BayesError, BayesMetric, BayesPrediction)
from operator import attrgetter
import random
from scipy.special import erfc
from scipy.stats import norm as gaussian
from time import time
from utils import debug
import warnings

log = getLogger(__name__)


class PipelinedBayesOpt(LearningTechnique):
  """Bayesian optimization for build pipeline
  
  This technique searches the optimal configuration using Bayesian optimization
  in the provided model.

  Attributes
  ----------
  thres_sample_cnt : int
    Number of samples used to determine a threshold for selecting the best
    samples in adaptive sampling
  thres_offs : float
    The fraction of samples that is not considered good enough during adaptive
    sampling
  sample_cnt : int
    Number of random samples from which a new configuration is selected.
  args : NameSpace
    Tuner command-line arguments
  random_state : RandomState
    Random state used for generating algorithm-independent deterministic
    configurations during the first iterations.  If no random state is
    provided, the configurations will be random.
  fidelities : dict
    Current fidelity level of each measured configuration
  incomplete_results : list of list of Result
    Results of incomplete builds grouped by fidelity
  model_map : dict
    Maps metrics to models
  opt_metric : str
    Metric that is optimized
  opt_model : Model
    Model of metric that is optimized
  settings : BayesSettings
    Tuner settings of this tuning run
  build_success_cnt : int
    Number of successfully built configurations
  optimum : float
    Predicted value of optimum
  preds : dict
    Maps configuration hashes to predictions
  """

  thres_sample_cnt = 40
  thres_offs = 0.1
  sample_cnt = 10000

  def __init__(self, *pargs, **kwargs):
    self.args = kwargs.pop('args', None)
    super(PipelinedBayesOpt, self).__init__(*pargs, **kwargs)
    self.cfg_vecs = np.empty(0)
    self.fidelities = {}
    self.incomplete_results = None
    self.model_map = {model.metric: model for model in self.models}
    self.opt_metric = None
    self.opt_model = None
    self.settings = None
    self.build_success_cnt = 0
    self.optimum = None
    self.random_state = np.random.RandomState(self.args.seed)
    self.preds = {}

  def handle_requested_result(self, result):
    """This callback is invoked by the search driver to report new results.
    
    Parameters
    ----------
    result : Result
      Result
    """
    for model in self.models:
      if model.metric != 'error_prob':
        value = getattr(result, model.metric)
        if math.isinf(value) and self.args.no_avoid_errors:
          value = 1e30
        if self.args.opt_log:
          value = math.log(1.0 + value)
      else:
        value = 0.0 if result.state == 'OK' else 1.0
      model.add_value(result.configuration, value)

    self.retrain = True

    fidelity = result.configuration.fidelity
    cfg = result.configuration.data

    metric_value = getattr(result, self.opt_metric)
    if fidelity in (0, self.max_fidelity) and result.state == 'OK':
      self.build_success_cnt += 1
      
    incomplete_build = fidelity > 0 and fidelity < self.max_fidelity
    if result.state == 'OK' and incomplete_build:
      self.incomplete_results[fidelity - 1].append(result)

    cfg_hash = self.manipulator.hash_config(cfg)
    error = math.log(metric_value + 1.0) - self.preds[cfg_hash]
    pred_error = BayesError(result=result, error=error)
    self.driver.session.add(pred_error)
    self.driver.session.commit()

  def select_configuration(self, fidelity=1):
    """Suggest a new configuration to evaluate.

    Parameters
    ----------
    fidelity : int
      Desired fidelity

    Returns
    -------
    Configuration
      Suggested configuration
    """
    debug.check()

#    for model in self.models:
#      cnts = []
#      for idx, submodel in enumerate(model.models):
#        cnt = len(submodel.model.results if idx > 0 else submodel.results)
#        cnts.append(cnt)
#      log.info('Init cnts %s: %s', model.metric, str(cnts))

    iteration = self.output_iteration()
    
    if self.models_ready():
      self.optimum = self.get_optimum()

    if fidelity == 1:
      if self.models_ready():
        cfg = self.minimize_cost_func()
      else:
        cfg = self.sample_design_space()
    else:
      results = self.incomplete_results[fidelity - 2]
      if len(results) == 0:
        return None
      cfgs = [result.configuration.data for result in results]
      cfg_vecs = [self.manipulator.get_vec(cfg) for cfg in cfgs]
      costs = self.compute_costs(np.array(cfg_vecs))['cost']
      idx = random.choice(np.where(costs == np.min(costs))[0])
      cfg = cfgs[idx]
      del results[idx]

    cost = self.compute_costs(np.array([self.manipulator.get_vec(cfg)]))
    self.output_cost(cost, iteration, cfg)

    # if self.driver.test_count == 100:
    #   self.output_models()

    # Update the fidelity of the configuration.
    self.increase_fidelity(cfg)

    # Remember the prediction to compute the prediction error.
    cfg_vec = self.manipulator.get_vec(cfg)
    means, std_devs, preds = self.opt_model.predict(np.array([cfg_vec]),
                                                    return_all=True)
    cfg_hash = self.manipulator.hash_config(cfg)
    pred = preds[fidelity - 1]['mean'][0] if fidelity > 0 else means[0]
    self.preds[cfg_hash] = pred

    # Return the configuration.
    return self.driver.get_configuration(cfg, fidelity=fidelity)

  def set_driver(self, driver):
    """Set the search driver.
    
    Parameters
    ----------
    driver : SearchDriver
    """
    super(PipelinedBayesOpt, self).set_driver(driver)
    self.opt_metric = self.objective.opt_metrics[0]
    self.opt_model = self.model_map[self.opt_metric]
    self.incomplete_results = [[] for fidelity in range(self.max_fidelity - 1)]

  def compute_costs(self, cfg_vecs):
    """Compute the cost of a number of configurations.

    Parameters
    ----------
    cfg_vecs : array of shape(no of samples, no of features)
      Configuration vectors

    Returns
    -------
    dict
      Dictionary with final cost and intermediate values contributing to the
      cost for every configuration.  The dictionary has the following members:

        cost : array of shape(no of samples, )
          Cost function results
        expec_impr : array of shape(no of samples, )
          Expected improvements
        success_prob : array of shape(no of samples, )
          Probabilities that build is successful.
        constr_pen : array of shape(no of samples, )
          Constraint penalties
        pred : dict
          Predictions for each metric.  Each member is named after a metric and
          contains a dictionary with the following information about that
          metric:

            mean : array of shape(no of samples, )
              Predictions of metric
            std_dev : array of shape(no of samples, )
              Standard deviations of metric
    """
    # Discourage configurations that probably cannot be built.
    sample_cnt = len(cfg_vecs)
    results = {'pred': {}}
    results['success_prob'] = np.ones(sample_cnt)
    if not self.args.no_avoid_errors:
      model = self.model_map['error_prob']
      if model.is_ready():
        means, std_devs, preds = model.predict(cfg_vecs, return_all=True)
        results['pred']['error_prob'] = preds
        probs = np.clip(1.0 - means, 0.0, 1.0)
        if probs.max() > 0.0:
          results['success_prob'] = probs

    # Discourage configurations that are unlikely to meet constraints.
    constr_metrics = self.objective.constrained_metrics
    constrs = self.objective.constraints
    results['constr_pen'] = np.ones(sample_cnt)
    for constr, metric in zip(constrs, constr_metrics):
      model = self.model_map[metric]
      if model.is_ready():
        means, std_devs, preds = model.predict(cfg_vecs, return_all=True)
        results['pred'][metric] = preds
        with np.errstate(divide='ignore', invalid='ignore'):
          if self.args.opt_log:
            constr = math.log(1.0 + constr)
          z = (constr - means) / std_devs
          probs = gaussian.cdf(z)
          if probs.max() > 0.0:
            results['constr_pen'] *= probs
  
    if self.models_ready():
      # Compute the expected improvement.  Expected improvement values are high
      # for good predictions and/or high uncertainty.
      means, std_devs, preds = self.opt_model.predict(cfg_vecs,
                                                      return_all=True)
      results['pred'][self.opt_metric] = preds
      if self.args.opt_log:
        # If the models are trained to the log of the training outputs, we need
        # the expected improvement formula in equation 6 from "An Experimental
        # Investigation of Model-Based Parameter Optimisation: SPO and Beyond"
        # by Hutter et al.  Since we want to support zeroes as well, we use the
        # tranformation y = log(1 + x) instead.  We compensate for this in the
        # expected improvement function.
        diffs = self.optimum - means
        with np.errstate(divide='ignore', invalid='ignore'):
          v = diffs / std_devs
          c = np.exp(0.5 * std_devs ** 2 + means) * gaussian.cdf(v - std_devs)
          expec_impr = np.exp(self.optimum) * gaussian.cdf(v) - c
          results['expec_impr'] = expec_impr
      else:
        diffs = self.optimum - means
        with np.errstate(divide='ignore', invalid='ignore'):
          z = diffs / std_devs
          exploit_terms = diffs * gaussian.cdf(z)
          explore_terms = std_devs * gaussian.pdf(z)
          results['expec_impr'] = exploit_terms + explore_terms
        results['expec_impr'][std_devs == 0.0] = 0.0
      results['optimum'] = self.optimum
    else:
      # Make sure all costs are ignored.
      results['expec_impr'] = np.ones(sample_cnt)

    # Compute the final cost.
    results['cost'] = -results['success_prob'] * results['constr_pen'] \
                       * results['expec_impr']

    # Occasionally, I see NaNs for the expected improvements.  They are caused
    # by excessive values for the coefficients in the multi-fidelity model. For
    # now, I will set these values to 0 to avoid crashes.
    results['cost'][np.isnan(results['cost'])] = 0.0

    return results
  
  def models_ready(self):
    """Returns whether the models are ready for reliable predictions.

    Returns
    -------
    bool
      True if the models are ready
    """
    return self.build_success_cnt >= self.args.bayes_seed_cnt

  def minimize_cost_func(self):
    """Minimize the cost function.

    Returns
    -------
    array of shape(no. of features)
      Configuration vector minimizing cost function
    """
    if self.args.adaptive_sampling:
      best_cost = float('inf')
      best_entropy = 0.0
      bounds = self.manipulator.get_bounds()
      while best_entropy < 0.9:
        cfg_vecs = self.manipulator.get_random_vecs(self.sample_cnt, bounds)
        costs = self.compute_costs(cfg_vecs)['cost']
        idx = np.argmin(costs)
        cost = costs[idx]
        if cost < best_cost:
          best_cfg_vec = cfg_vecs[idx]
          best_cost = cost
        idx = int(self.thres_sample_cnt * self.thres_offs)
        thres = np.sort(costs[:self.thres_sample_cnt])[idx]
        best_cfg_vecs = cfg_vecs[costs <= thres, :]
        best_entropy = float('inf')
        for dim in np.random.permutation(cfg_vecs.shape[1]):
          min_value = np.min(cfg_vecs[:, dim])
          max_value = np.max(cfg_vecs[:, dim])
          center = (min_value + max_value) / 2.0
          if center == min_value or center == max_value:
            continue
          mask = best_cfg_vecs[:, dim] > center
          prob_higher = np.sum(mask) / float(len(best_cfg_vecs))
          prob_lower = 1.0 - prob_higher
          if prob_lower == 0.0 or prob_higher == 0.0:
            entropy = 0.0
          else:
            entropy = -prob_lower * math.log(prob_lower, 2) - \
                       prob_higher * math.log(prob_higher, 2)
          if entropy < best_entropy:
            best_entropy = entropy
            best_dim = dim
            best_center = center
            if prob_lower != prob_higher:
              best_bound = prob_lower > prob_higher
            else:
              best_bound = np.random.randint(0, 1)
        bounds[best_dim][best_bound] = best_center
      return self.manipulator.get_cfg(best_cfg_vec)
    else:
      bounds = self.manipulator.get_bounds()
      cfg_vecs = self.manipulator.get_random_vecs(self.sample_cnt, bounds)
      costs = self.compute_costs(cfg_vecs)
      return self.manipulator.get_cfg(cfg_vecs[np.argmin(costs['cost'])])
  
  def sample_design_space(self):
    """Sample the design space.

    When not enough working configurations have been found, the expected
    improvement factor of the cost function is not used.  The expected
    improvement encourages configurations in regions with high uncertainty.
    The remaining factors of the cost function do not do this, so without the
    expected improvement, points obtained by optimizing the cost function are
    more likely to be similar.  Therefore, we sample the design space instead,
    taking care that configurations with a lower cost are given a higher
    weight.

    Returns
    -------
    array of shape(no. of features)
      Configuration vector minimizing cost function
    """
    bounds = self.manipulator.get_bounds()
    cfg_vecs = self.manipulator.get_random_vecs(self.sample_cnt, bounds)
    costs = self.compute_costs(cfg_vecs)['cost']
    accum_weights = -costs.cumsum()
    idx = sum(random.uniform(0, accum_weights[-1]) > accum_weights)
    return self.manipulator.get_cfg(cfg_vecs[idx])

  def get_optimum(self):
    """Compute the probably optimum.

    Returns
    -------
    dict
      Configuration of optimum
    float
      Optimum
    """
    bounds = self.manipulator.get_bounds()
    cfg_vecs = self.manipulator.get_random_vecs(self.sample_cnt, bounds)
    means, std_devs = self.opt_model.predict(cfg_vecs)
    preds = means + std_devs

    constr_metrics = self.objective.constrained_metrics
    constrs = self.objective.constraints
    mask = np.zeros(self.sample_cnt, dtype=bool)
    for constr, metric in zip(constrs, constr_metrics):
      model = self.model_map[metric]
      means, std_devs = model.predict(cfg_vecs)
      if self.args.opt_log:
        constr = math.log(1.0 + constr)
      new_mask = np.logical_or(mask, means + std_devs > constr)
      if np.all(new_mask):
        log.warning('Constraint for %s will be ignored.', metric)
      else:
        mask = new_mask
    
    if not self.args.no_avoid_errors:
      model = self.model_map['error_prob']
      means, std_devs = model.predict(cfg_vecs)
      new_mask = np.logical_or(mask, means + std_devs > 0.5)
      if np.all(new_mask):
        log.warning('Not using UCB for error probability.')
        new_mask = np.logical_or(mask, means > 0.5)
      if np.all(new_mask):
        log.warning('Error probability will be ignored.')
      else:
        mask = new_mask

    preds[mask] = float('inf')
    return np.min(preds)

  def get_next_fidelity(self, cfg):
    """Returns the next fidelity of given configuration.

    Parameters
    ----------
    cfg : dict
      Configuration

    Returns
    -------
    int
      Fidelity
    """
    if self.max_fidelity > 1:
      self.manipulator.normalize(cfg)
      cfg_hash = self.manipulator.hash_config(cfg)
      return self.fidelities.setdefault(cfg_hash, 0) + 1
    else:
      return 0
  
  def increase_fidelity(self, cfg):
    """Increase fidelity of given configuration.

    Parameters
    ----------
    cfg : dict
      Configuration
    """
    if self.max_fidelity > 1:
      self.manipulator.normalize(cfg)
      cfg_hash = self.manipulator.hash_config(cfg)
      fidelity = self.fidelities.setdefault(cfg_hash, 0) + 1
      self.fidelities[cfg_hash] = fidelity
  
  def output_iteration(self):
    """Output current iteration to the database for debug purposes.

    Returns
    -------
    BayesIteration
      Database object created for iteration
    """
    iteration = BayesIteration(tuning_run=self.driver.tuning_run,
                               generation=self.driver.generation)
    self.driver.session.add(iteration)
    self.driver.session.commit()
    return iteration

  def output_cost(self, costs, iteration, cfg):
    """Output the cost to the database for debug purposes.

    Parameters
    ----------
    costs : dict
      Dictionary from compute_costs with final cost and intermediate results
    iteration : BayesIteration
      Database object with information about current iteration
    cfg : dict
      Configuration
    """
    fidelity = self.get_next_fidelity(cfg)
    new_cfg = fidelity == 1
    
    cfg = self.driver.get_configuration(cfg, fidelity=fidelity)
    self.driver.session.add(cfg)

    cost = BayesCost(iteration=iteration, new_cfg=new_cfg) 
    cost.configuration = cfg

    for key, value in costs.items():
      if key == 'optimum':
        cost.optimum = value
      elif key != 'pred':
        setattr(cost, key, value[0])

    for metric, metric_info in costs['pred'].items():
      for fidelity, fidelity_info in enumerate(metric_info):
        pred = BayesPrediction(cost=cost, fidelity=fidelity+1)
        pred.metric = BayesMetric.get(self.driver.session, metric)
        pred.mean = fidelity_info['mean'][0]
        pred.std_dev = fidelity_info['std_dev'][0]
        self.driver.session.add(pred)

    self.driver.session.add(cost)
    self.driver.session.commit()

  def output_models(self):
    """Sample all models and output them to the database for debug purposes.
    """
    bounds = self.manipulator.get_bounds()
    cfg_vecs = self.manipulator.get_random_vecs(self.sample_cnt, bounds)

    results = {}
    for model in self.models:
      results[model.metric] = model.sample_models(cfg_vecs)

    info = DebugInfo(tuning_run=self.driver.tuning_run,
                     info=[cfg_vecs, results])
    self.driver.session.add(info)

