"""
Classes for Bayesian optimization using bandit-based fidelity selection

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
from scipy.stats import norm as gaussian
from utils import debug
import warnings

log = getLogger(__name__)


class BanditBasedBayesOpt(LearningTechnique):
  """Bayesian optimization using bandit-based fidelity selection
  
  This technique searches the optimal configuration using Bayesian optimization
  in the provided model.  So far, this approach was not effective because the
  rewards are too sparse.

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
  fid_sel : FidelitySelection
    Fidelity pattern selection object
  patterns : tuple of tuple of int
    A list of fidelity patterns.  Each fidelity patern is a sequence of
    integers indicating how many builds must be performed at the associated
    fidelity.
  next_fidelities : list of int
    The fidelities of the upcoming builds
  results : list of Result
    All results received since the start of the current fidelity pattern.
  pattern : int
    Current fidelity pattern
  """

  thres_sample_cnt = 40
  thres_offs = 0.1
  sample_cnt = 10000

  def __init__(self, *pargs, **kwargs):
    self.args = kwargs.pop('args', None)
    super(BanditBasedBayesOpt, self).__init__(*pargs, **kwargs)
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
    self.patterns = ((1, 1, 1), (2, 1, 1), (4, 1, 1), (4, 4, 1))
    self.fid_sel = FidelitySelection(pattern_cnt=len(self.patterns))
    self.next_fidelities = []
    self.results = []
    self.pattern = None

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

    self.results.append(result)

    cfg_hash = self.manipulator.hash_config(cfg)
    error = math.log(metric_value + 1.0) - self.preds[cfg_hash]
    pred_error = BayesError(result=result, error=error)
    self.driver.session.add(pred_error)
    self.driver.session.commit()

  def select_configuration(self):
    """Suggest a new configuration to evaluate.
    
    Returns
    -------
    Configuration
      Suggested configuration
    """
    debug.check()

    iteration = self.output_iteration()

    if self.models_ready():
      self.optimum = self.get_optimum()

    while True:
      try:
        fidelity = self.next_fidelities.pop(0)
      except:
        if len(self.results) > 0:
          self.fid_sel.handle_results(self.pattern, self.results)
          self.results = []
        if self.models_ready():
          self.pattern = self.fid_sel.select_pattern()
        else:
          self.pattern = 0
        pattern = self.patterns[self.pattern]
        self.next_fidelities = []
        for fidelity, cnt in enumerate(pattern):
          self.next_fidelities += [fidelity + 1 for i in range(cnt)]
        fidelity = self.next_fidelities.pop(0)
      if fidelity == 1 or len(self.incomplete_results[fidelity - 2]) > 0:
        break

    if fidelity == 1:
      if self.random_state is None or self.models_ready():
        cfg = self.minimize_cost_func()
      else:
        bounds = self.manipulator.get_bounds()
        cfg_vecs = self.manipulator.get_random_vecs(1, bounds, self.random_state)
        cfg = self.manipulator.get_cfg(cfg_vecs[0])
    else:
      results = self.incomplete_results[fidelity - 2]
      cfgs = [result.configuration.data for result in results]
      cfg_vecs = [self.manipulator.get_vec(cfg) for cfg in cfgs]
      costs = self.compute_costs(np.array(cfg_vecs))
      idx = np.argmin(costs['cost'])
      del self.incomplete_results[fidelity - 2][idx]
      cfg = self.manipulator.get_cfg(cfg_vecs[idx])

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
    super(BanditBasedBayesOpt, self).set_driver(driver)
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
    if not self.args.no_avoid_errors:
      model = self.model_map['error_prob']
      means, std_devs, preds = model.predict(cfg_vecs, return_all=True)
      results['success_prob'] = np.clip(1.0 - means, 0.0, 1.0)
      # If all success probabilities are 0, other cost factors will be ignored,
      # so we set the probabilities to 1 instead.
      if results['success_prob'].max() == 0.0:
        results['success_prob'] = np.ones(sample_cnt)
      results['pred']['error_prob'] = preds
    else:
      results['success_prob'] = np.ones(sample_cnt)

    if self.models_ready():
      # Discourage configurations that are unlikely to meet constraints.
      constr_metrics = self.objective.constrained_metrics
      constrs = self.objective.constraints
      results['constr_pen'] = np.ones(sample_cnt)
      for constr, metric in zip(constrs, constr_metrics):
        model = self.model_map[metric]
        means, std_devs, preds = model.predict(cfg_vecs, return_all=True)
        results['pred'][metric] = preds
        with np.errstate(divide='ignore', invalid='ignore'):
          if self.args.opt_log:
            constr = math.log(1.0 + constr)
          z = (constr - means) / std_devs
          results['constr_pen'] *= gaussian.cdf(z)
  
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
          results['expec_impr'] = np.exp(self.optimum) * gaussian.cdf(v) - c
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
      results['constr_pen'] = np.ones(sample_cnt)

    # Compute the final cost.
    results['cost'] = -results['success_prob'] * results['constr_pen'] \
                       * results['expec_impr']
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


class FidelitySelection(object):
  """Select fidelity pattern using multi-armed bandit

  A fidelity pattern described the number of builds at each fidelity.  We use
  an algorithm based on the algorithm described in "Comparison-based Adaptive
  Strategy Selection with Bandits in Differential Evoluation" by Fialho et al.

  Attributes
  ----------
  pattern_cnt : int
    Number of fidelity patterns
  results : list of list of bool
    Each pattern has a list that keeps track of whether each time the pattern
    was used improved the objective.
  build_times : list of list of float
    A list of build_times for each pattern.
  pattern : int
    Last pattern that was selected
  weight : float
    Weight that controls balance between exploration and exploitation terms
  """
  weight = 0.05

  def __init__(self, *pargs, **kwargs):
    self.pattern_cnt = kwargs['pattern_cnt']
    self.results = [[] for pattern in range(self.pattern_cnt)]
    self.build_times = [[] for pattern in range(self.pattern_cnt)]
    self.pattern = None

  def handle_results(self, pattern, results):
    """Process new results

    Parameters
    ----------
    pattern : int
      Fidelity pattern associated with results
    results : list of Result
      Results obtained for the current pattern
    """
    improvement = False
    build_time = 0.0
    for result in results:
      best = result.state == 'OK' and result.was_new_best == 1
      improvement = improvement or best
      build_time += result.build_time
    self.results[pattern].append(improvement)
    self.build_times[pattern].append(build_time)

  def select_pattern(self):
    """Select a fidelity pattern for the next builds.

    Returns
    -------
    int
      Number of the fidelity pattern
    """
    scores = [self.compute_score(pattern) \
              for pattern in range(self.pattern_cnt)]
    max_score = np.max(scores)
    patterns = [pattern for pattern, score in enumerate(scores) \
                if score == max_score]
    pattern = random.choice(patterns)
    log.info('Pattern scores: %s, Selected pattern: %d' % (scores, pattern))
    return pattern

  def compute_score(self, pattern):
    """Compute the score of a fidelity pattern.

    This is essentially the identical to the approach described by Fialho et
    al., except that the scores are scaled by the build time to avoid giving
    long builds an unfair advantage.

    Parameters
    ----------
    pattern : int
      Fidelity pattern

    Returns
    -------
    float
      Score
    """
    results = np.array(self.results[pattern])
    result_cnt = len(results)
    if result_cnt == 0:
      return float('inf')

    seq = np.arange(len(results), 0.0, -1.0)
    area = (seq * results).sum()
    max_area = seq.sum()
    exploit_term = area / max_area
    
    total_cnt = sum(len(elem) for elem in self.results)
    explore_term = math.sqrt((2.0 * math.log(total_cnt, 2.0)) / result_cnt)

    score = exploit_term + self.weight * explore_term

    avg_build_time = np.array(self.build_times[pattern]).mean()
    return score / avg_build_time

