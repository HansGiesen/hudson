"""
Classes for building single configuration

Created on Aug 18, 2021
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

import logging
import yaml
from opentuner.search.technique import AsyncProceduralSearchTechnique

log = logging.getLogger(__name__)


class SingleBuild(AsyncProceduralSearchTechnique):
  """Suggest a single configuration to build

  Attributes
  ----------
  param_file : str
    YAML file with configuration to analyze
  """
  def __init__(self, *pargs, **kwargs):
    self.param_file = kwargs.pop('param_file')
    super(SingleBuild, self).__init__(*pargs, **kwargs)

  def main_generator(self):
    """Suggest configurations to evaluate.
    
    Yields
    -------
    dict
      Suggested configuration
    """
    with open(self.param_file, 'r') as input_file:
        data = input_file.read()
    yield yaml.safe_load(data)
