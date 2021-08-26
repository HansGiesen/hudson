"""
Classes for analyzing final configuration

Created on Jun 28, 2021
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

import logging
from opentuner.resultsdb.models import Configuration
from opentuner.search.technique import AsyncProceduralSearchTechnique

log = logging.getLogger(__name__)


class ImpactAnalysis(AsyncProceduralSearchTechnique):
  """Suggest builds for analysis of impact of final configuration parameters

  Attributes
  ----------
  rep_cnt : int
    Number of alternative values to suggest for each parameter
  attempt_cnt : int
    Maximum number of sample to try to find a random value that has not been
    built yet
  base_cfg : int
    ID number in database of final configuration to analyze
  """
  rep_cnt = 3
  attempt_cnt = 100

  def __init__(self, *pargs, **kwargs):
    self.base_cfg = kwargs.pop('base_cfg')
    super(ImpactAnalysis, self).__init__(*pargs, **kwargs)

  def main_generator(self):
    """Suggest configurations to evaluate.
    
    Yields
    -------
    dict
      Suggested configuration
    """
    query = self.driver.session.query(Configuration)
    base_cfg = query.filter_by(id=self.base_cfg).one().data

    for param in self.manipulator.params:
      invalid = [base_cfg[param.name]]
      for rep in range(self.rep_cnt):
        for attempt in range(self.attempt_cnt):
          value = self.manipulator.random()[param.name]
          if value not in invalid:
            break
        else:
          break
        invalid.append(value)
        cfg = base_cfg.copy()
        cfg[param.name] = value
        yield cfg
