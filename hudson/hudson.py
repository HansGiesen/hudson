#!/usr/bin/env python2
'''
HuDSoN

Created on Jun 17, 2019
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
'''

import argparse
import logging
import os
import numpy as np
import shutil
import signal
import sys
import yaml

tuner_root = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.insert(0, os.path.join(tuner_root, 'opentuner'))
sys.path.insert(0, tuner_root)

from measurement.sdsoc_interface import SDSoCInterface
from models.gaussianprocess import GaussianProcess
from models.multifidelitymodels import (FilterModel, MultiFidelityModel,
                                        ScaledSumModel)
from models.randomforest import RandomForest
from opentuner import argparsers
from opentuner.measurement.inputmanager import FixedInputManager
from opentuner.resultsdb.models import HudsonTuningRun, Platform, Test
from opentuner.search.objective import ThresholdAreaMinimizeTime, MinimizeTime
from opentuner.search.technique import register
import opentuner.tuningrunmain
from opentuner.tuningrunmain import TuningRunMain
from search.impactanalysis import ImpactAnalysis
from search.pipelinedbayesopt import PipelinedBayesOpt
from search.singlebuild import SingleBuild
from search.thresbasedbayesopt import ThresBasedBayesOpt
from search.randomsearch import RandomSearch, RandomSearchAvoidErrors


log = logging.getLogger(__name__)


class Hudson(object):
  """
  Main class of HuDSoN

  Attributes
  ----------
  args : Namespace
    Command-line arguments
  tuner_cfg : dict
    Tuner configuration
  """
  def __init__(self):
    self.args = None
    self.tuner_cfg = None

  def main(self):
    """Main function of HuDSoN
    """
    signal.signal(signal.SIGUSR2, self.signal_handler)

    self.process_args()
    self.prepare_output_dir()
    self.init_logging()
    self.register_techniques()

    self.tuner_cfg['tuner_root'] = tuner_root

    if self.args.objective == 'MinTime':
      objective = MinimizeTime()
    else:
      resource_types = ['luts', 'regs', 'dsps', 'brams']
      constraints = tuple(self.tuner_cfg['platform'][resource_type]
                          for resource_type in resource_types)
      objective = ThresholdAreaMinimizeTime(constraints)

    input_manager = FixedInputManager()

    interf = SDSoCInterface(self.args, tuner_cfg=self.tuner_cfg,
                            objective=objective, input_manager=input_manager)

    main = TuningRunMain(interf, self.args)
    main.fake_commit = False
    main.init()

    tuning_run = main.tuning_run
    tuning_run.seed = self.args.seed

    hudson_tuning_run = HudsonTuningRun(args=self.args, cfg=self.tuner_cfg,
                                        tuning_run=tuning_run)
    main.session.add(hudson_tuning_run)
    main.session.commit()

    platform = Platform.get(main.session, self.args.platform)
    columns = ['luts', 'regs', 'dsps', 'brams', 'proc_freq', 'axi_bus_width']
    for column in columns:
      setattr(platform, column, self.tuner_cfg['platform'][column])
    main.session.commit()

    test = Test.get(main.session, self.args.test, self.args.description)
    tuning_run.test = test
    tuning_run.platform = platform
    main.session.commit()

    main.main()

  def process_args(self):
    """Process command-line arguments.
    """
    parser = argparse.ArgumentParser(description='HuDSoN',
                                     parents=argparsers())
    parser.add_argument('cfg_file', type=os.path.abspath,
                        help='Configuration file')
    parser.add_argument('--test', help='Test name')
    parser.add_argument('--description', default='', help='Test description')
    parser.add_argument('--output-dir', type=os.path.abspath,
                        help='Output directory')
    parser.add_argument('--platform', help='Target platform')
    parser.add_argument('--build-host', help='Build host')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--use-prebuilt-presynth', action='store_true',
                        help='Use prebuilt presynthesis result')
    parser.add_argument('--use-prebuilt-synth', action='store_true',
                        help='Use prebuilt synthesis result')
    parser.add_argument('--use-prebuilt-impl', action='store_true',
                        help='Use prebuilt implementation result')
    parser.add_argument('--original-batching', action='store_true',
                        help='Do not equalize build times')
    parser.add_argument('--explore-offset', type=float, default=0.0,
                        help='Additive exploration constant')
    parser.add_argument('--early-term-thres', type=float, default=1e-8,
                        help='Early termination threshold')
    parser.add_argument('--no-diverse-batches', action='store_true',
                        help='Disable increasing batch diversity')
    parser.add_argument('--relax-bounds', action='store_true',
                        help='Relax GP kernel bounds')
    parser.add_argument('--gp-restarts', type=int, default=2,
                        help='GP fitting restarts')
    parser.add_argument('--objective', default='MinTimeConstrArea',
                        choices=['MinTime', 'MinTimeConstrArea'],
                        help='Tuning objective')
    parser.add_argument('--no-avoid-errors', action='store_true',
                        help='Disable error avoidance')
    parser.add_argument('--no-corr-clk', action='store_true',
                        help='No correlation between clock frequencies')
    parser.add_argument('--bayes-seed-cnt', type=int, default=8,
                        help='Seed configurations for Bayesian optimization')
    parser.add_argument('--init-div-pen', action='store_true',
                        help='Enable diversity penalty initially')
    parser.add_argument('--opt-log', action='store_true',
                        help='Optimize log of results.')
    parser.add_argument('--no-timeouts', action='store_true',
                        help='Disable build timeouts.')
    parser.add_argument('--no-random-sampling', action='store_true',
                        help='Disable random sampling.')
    parser.add_argument('--adaptive-sampling', action='store_true',
                        help='Use adaptive sampling.')
    parser.add_argument('--no-tool-params', action='store_true',
                        help='Do not tune tool parameters.')
    parser.add_argument('--no-interf-params', action='store_true',
                        help='Do not tune interface parameters.')
    parser.add_argument('--no-csim', action='store_true',
                        help='Disable C-simulation.')
    parser.add_argument('--base-cfg', type=int,
                        help='Configuration ID for impact analysis')
    parser.add_argument('--param-file', help='Configuration file')
    parser.add_argument('--use-64-bit-bus', action='store_true',
                        help='Use a 64-bit data bus.')
    parser.add_argument('--no-bugfixes', action='store_true',
                        help='Disable post-deadline bugfixes.')
    args = parser.parse_args()

    self.args = args

    tuner_cfg = self.load_cfg(args.cfg_file)
    self.tuner_cfg = tuner_cfg

    tuner_cfg['project_dir'] = os.path.dirname(args.cfg_file)

    if args.output_dir:
      output_root = args.output_dir
    else:
      output_root = os.path.join(tuner_cfg['project_dir'], 'output')
    tuner_cfg['output_root'] = output_root

    if 'log_file' not in tuner_cfg:
      tuner_cfg['log_file'] = os.path.join(output_root, 'hudson.log')

    if args.platform:
      tuner_cfg['platform'] = tuner_cfg['platforms']['types'][args.platform]
    else:
      tuner_cfg['platform'] = tuner_cfg['platforms']['types']['zcu102']

    platform_dir = tuner_cfg['platform']['dir']
    if not os.path.isabs(platform_dir):
      platform_dir = os.path.join(tuner_root, platform_dir)
    tuner_cfg['platform']['dir'] = platform_dir

    if not args.database:
      if tuner_cfg['database']:
        args.database = tuner_cfg['database']
      else:
        args.database = os.path.join(output_root, 'hudson.db')

    if not args.technique:
      if 'design_space' in tuner_cfg:
        args.technique = ['Bayes']
      else:
        args.technique = ['MultiFidBayes']

    if args.build_host:
      queues = ['*@{}'.format(args.build_host)]
      tuner_cfg['build_interf']['queues'] = queues

    if args.label:
      tuner_cfg['job_name'] = args.label

    if args.test:
      tuner_cfg['job_name'] = '{}_{}'.format(args.test, tuner_cfg['job_name'])
    else:
      args.test = 'unnamed'

  def load_cfg(self, filename):
    """Load the tuner configuration file.

    Parameters
    ----------
    filename : str
      Configuration filename

    Returns
    -------
    dict
      Tuner configuration
    """
    with open(filename, 'r') as cfg_file:
      data = cfg_file.read()
    tuner_cfg = yaml.safe_load(data)
    include_file = tuner_cfg.get('include')
    if include_file:
      if not os.path.isabs(include_file):
        include_file = os.path.join(os.path.dirname(filename), include_file)
      parent = self.load_cfg(include_file)
      parent.update(tuner_cfg)
      tuner_cfg = parent
    return tuner_cfg

  def prepare_output_dir(self):
    """Prepare the output directory.
    """
    output_dir = self.tuner_cfg['output_root']
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

  @staticmethod
  def signal_handler(signum, frame):
    """Raise a keyboard interrupt when a USR2 signal is caught.
    """
    log.info('Received signal USR2.')
    raise KeyboardInterrupt
  
  def init_logging(self):
    """Initialize the logging.
    """
    logging.config.dictConfig({
      'version': 1,
      'disable_existing_loggers': False,
      'formatters': {'console': {'format': '[%(relativeCreated)6.0fs] '
                                           '%(levelname)7s: '
                                           '%(message)s'},
                     'file': {'format': '[%(asctime)-15s] '
                                        '%(levelname)7s: '
                                        '%(message)s '
                                        '@%(filename)s:%(lineno)d'}},
      'handlers': {'console': {'class': 'logging.StreamHandler',
                               'formatter': 'console',
                               'level': 'INFO'},
                   'file': {'class': 'logging.FileHandler',
                            'filename': self.tuner_cfg['log_file'],
                            'formatter': 'file',
                            'level': 'INFO'}},
      'loggers': {'': {'handlers': ['console', 'file'],
                       'level': 'INFO',
                       'propagate': True}}})
    # Avoid initializing the logging again.
    opentuner.tuningrunmain.init_logging = lambda: None

  def register_techniques(self):
    """Register search techniques.
    """
    register(RandomSearch(name="Random"))
    register(RandomSearchAvoidErrors(GaussianProcess(),
                                     name="RandomAvoidErrors"))

    register(ImpactAnalysis(name='ImpactAnalysis', base_cfg=self.args.base_cfg))
    
    register(SingleBuild(name='SingleBuild', param_file=self.args.param_file))

    gp_params = {'relax_bounds': self.args.relax_bounds,
                 'restarts':     self.args.gp_restarts,
                 'avoid_errors': not self.args.no_avoid_errors}

    if self.args.objective == 'MinTime':
      metrics = ['error_prob', 'run_time']
    else:
      metrics = ['error_prob', 'run_time', 'luts', 'regs', 'dsps', 'brams']

    techniques = [('ThresBasedMultiFidBayes', ThresBasedBayesOpt),
                  ('PipelinedMultiFidBayes', PipelinedBayesOpt)]
    for name, cls in techniques:
      models = []
      for metric in metrics:
        if metric == 'error_prob':
          submodels = [GaussianProcess(binary=True, **gp_params),
                       FilterModel(GaussianProcess(binary=True, **gp_params)),
                       FilterModel(GaussianProcess(binary=True, **gp_params))]
        else:
          submodels = [GaussianProcess(**gp_params),
                       ScaledSumModel(GaussianProcess(**gp_params)),
                       ScaledSumModel(GaussianProcess(**gp_params))]
        models.append(MultiFidelityModel(metric, submodels))
      register(cls(models, name=name, args=self.args))

    models = []
    for metric in metrics:
      if metric == 'error_prob':
        submodels = [RandomForest(),
                     FilterModel(RandomForest()),
                     FilterModel(RandomForest())]
      else:
        submodels = [RandomForest(),
                     ScaledSumModel(RandomForest()),
                     ScaledSumModel(RandomForest())]
      models.append(MultiFidelityModel(metric, submodels))
    register(ThresBasedBayesOpt(models, name='RandomForest', args=self.args))

    models = []
    for metric in metrics:
      if metric == 'error_prob':
        models.append(GaussianProcess(metric, binary=True, **gp_params))
      else:
        models.append(GaussianProcess(metric, **gp_params))
    register(ThresBasedBayesOpt(models, name="Bayes", args=self.args))


if __name__ == '__main__':
  Hudson().main()

