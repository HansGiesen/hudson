"""
Classes for assigning configurations in a batch to threads

Created on Feb 12, 2020
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

from logging import getLogger
from math import isinf
from numpy import array

log = getLogger(__name__)


class BatchComposer(object):
  """Class for assigning configurations in a batch to threads

  Attributes
  ----------
  technique : SearchTechniqueBase
    Root search technique
  models : list of Model
    Models to predict build time for each fidelity
  parallelism : int
    Number of measurement threads
  lookahead : DesiredResult
    Configuration that was already selected, but has not been attempted yet 
  """

  # The duration of the first build
  first_build_time = 6541.0
  # Minimum waste decrease for new configuration to be added to batch
  min_waste_dec = 1e-3

  def __init__(self, technique, models, parallelism):
    self.technique = technique
    self.models = models
    self.parallelism = parallelism
    self.lookahead = None
    for model in self.models:
      model.metric = "build_time"

  def add_result(self, result):
    """This callback is invoked by the search driver to report new results.
    
    Parameters
    ----------
    result : Result
      Result
    """
    if not isinf(result.build_time):
      fidelity = result.configuration.fidelity
      fidelity = 1 if fidelity == 0 else fidelity
      self.models[fidelity - 1].add_result(result)

  def compose_batch(self):
    """Select one or more configurations for each thread.

    Returns
    -------
    list of DesiredResult
      A list with desired results.  The thread assignment is given by the
      thread attribute of each desired result.
    """
    if self.parallelism == 1:
      dr = self.technique.desired_result()
      dr.thread = 0
      return [dr]

    for model in self.models:
      model.train()

    done = False
    cfgs = []
    for i in range(self.parallelism):
      if self.lookahead:
        dr = self.lookahead
        self.lookahead = None
      else:
        dr = self.technique.desired_result()
      if dr is None or dr is False:
        done = True
        break
      time = self.predict(dr)
      cfgs.append((time, dr))

    bins, bin_size = self.binary_search(cfgs)
    total_time = sum(time for time, _ in cfgs)
    prev_waste = 1.0 - total_time / self.parallelism / bin_size

    while not done:
      dr = self.technique.desired_result()
      if dr is None or dr is False:
        break
      self.lookahead = dr
      time = self.predict(dr)
      new_cfgs = cfgs + [(time, dr)]
      new_bins, new_bin_size = self.binary_search(new_cfgs) 
      total_time = sum(time for time, _ in new_cfgs)
      waste = 1.0 - total_time / self.parallelism / new_bin_size
      if prev_waste - waste < self.min_waste_dec:
        break
      cfgs = new_cfgs
      bins = new_bins
      bin_size = new_bin_size
      prev_waste = waste

    desired_results = []
    for thread, grp in enumerate(bins):
      for dr in grp:
        dr.thread = thread
        desired_results.append(dr)

    assignment = ", ".join("{}: {}".format(dr.id, dr.thread)
                           for dr in desired_results)
    log.info("Batch assignment: %s", assignment)
    log.info("Expected batch duration: %e s, Time wasted: %f%%",
             bin_size, 100.0 * prev_waste)
    return desired_results
      
  def predict(self, desired_result):
    """Predict the build time for a given configuration.

    Parameters
    ----------
    desired_result : DesiredResult
      Desired result with configuration to be predicted

    Returns
    -------
    float
      Build time
    """
    fidelity = desired_result.configuration.fidelity
    fidelity = 1 if fidelity == 0 else fidelity
    model = self.models[fidelity - 1]

    # Fall back on lower fidelity if no results are available yet.
    if len(model.results) == 0 and fidelity > 1:
      model = self.models[fidelity - 2]

    if len(model.results) > 0:
      cfg_vec = model.get_vec(desired_result.configuration.data)
      time = model.predict(array(cfg_vec).reshape(1, -1))[0][0]
    else:
      time = self.first_build_time

    # Ensure that the build time is positive to avoid issues with bin packing
    # later.
    time = max(time, 1e-8)

    log.info("Predicted build time: %e", time)
    return time

  def binary_search(self, cfgs):
    """Find smallest bin size for which all configurations fit in threads.

    Parameters
    ----------
    cfgs : list of DesiredResult
      Configurations that we wish to test
    
    Returns
    -------
    list of list of DesiredResult
      Bin assignment.  Each sublist represents all desired results for a
      thread.
    float
      Smallest bin size for which all configurations fit
    """
    low = 0.0
    high = 1.0
    while True:
      try:
        bins = self.best_fit(cfgs, high)
        break
      except NoFitException:
        high *= 2.0
    while True:
      curr = (high + low) / 2.0
      if curr < low + 1e-8 or curr > high - 1e-8:
        break
      try:
        bins = self.best_fit(cfgs, curr)
        high = curr
      except NoFitException:
        low = curr
    bins = self.best_fit(cfgs, high)
    return bins, high

  def best_fit(self, cfgs, bin_size):
    """Assign configurations of bins, minimizing number of bins needed

    Parameters
    ----------
    cfgs : list of (float, DesiredResult)
      Configurations that we wish to test
    bin_size : float
      Size of each bin

    Returns
    -------
    list of list of DesiredResult
      Bin assignment.  Each sublist represents all desired results for a
      thread.

    Notes
    -----
    This algorithm is a solution to the bin packing problem.  The strategy
    that we use is "Best Fit Decreasing".  It is not optimal, but the number of
    bins is guaranteed to be no more than 11/9 OPT + 4, where OPT is the
    minimum number of bins.
    """
    cfgs.sort(reverse=True)
    bins = [[] for i in range(self.parallelism)]
    sizes = [0.0] * self.parallelism
    for time, dr in cfgs:
      combined = sorted(zip(sizes, bins), reverse=True)
      sizes = [size for size, _ in combined]
      bins = [b for _, b in combined]
      for i in range(self.parallelism):
        new_size = sizes[i] + time
        if new_size <= bin_size:
          bins[i].append(dr)
          sizes[i] = new_size
          break
      else:
        raise NoFitException()
    return bins


class NoFitException(Exception):
  """Exception thrown when BatchComposer.best_fit fails."""
  pass
