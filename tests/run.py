#!/usr/bin/env python3

import os
import stat
import subprocess
import traceback
from multiprocessing.pool import ThreadPool


class Test:
    def __init__(self, technique, benchmark, platform):
        self.technique = technique
        self.benchmark = benchmark
        self.platform = platform
        self.core_map = None
        self.seed_cnt = 6
        self.csim = False
        self.async_compile = False
        self.avoid_errors = True
        self.random_sampling = True
        self.init_cnt = None
        self.base_cfg = None
        self.cfg_file = None
        self.tuning_time = 60.0 * 60.0 * 24.0 * 3.0
        self.subtest = benchmark

    def run(self, seed):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        tuner_root = os.path.realpath(os.path.join(script_dir, '..'))
        python2_path = os.path.join(tuner_root, 'python2_env/bin/python2')
        tuner_path = os.path.join(tuner_root, 'hudson/hudson.py')
        benchmark_dir = os.path.join(tuner_root, 'rosetta')
        cfg_file = '{}.yml'.format(self.benchmark)
        cfg_file = os.path.join(benchmark_dir, self.benchmark, cfg_file)
        subtest = self.subtest
        if self.seed_cnt > 1:
            subtest += ('_' if len(self.subtest) > 0 else '') + str(seed)
        output_dir = os.path.join(script_dir, self.test, subtest)
        script_file = os.path.join(output_dir, 'tune.bash')
    
        os.makedirs(output_dir, exist_ok=True)
        with open(script_file, 'w') as output_file:
            output_file.write('#!/bin/bash -e\n')
            output_file.write('cd %s \n' % output_dir)
            output_file.write('%s %s %s \\\n' % (python2_path,
                                                 tuner_path,
                                                 cfg_file))
            output_file.write('    --output-dir output \\\n')
            output_file.write('    --test %s \\\n' % self.test)
            output_file.write('    --label %s \\\n' % subtest)
            output_file.write('    --seed %d \\\n' % seed)
            if self.core_map is None:
                output_file.write('    --parallelism 4 \\\n')
            else:
                output_file.write('    --core-map %s \\\n' % self.core_map)
            output_file.write('    --original-batching \\\n')
            output_file.write('    --technique %s \\\n' % self.technique)
            output_file.write('    --opt-log \\\n')
            output_file.write('    --objective MinTimeConstrArea \\\n')
            if self.random_sampling:
                output_file.write('    --adaptive-sampling \\\n')
            else:
                output_file.write('    --no-random-sampling \\\n')
            if self.async_compile:
                output_file.write('    --async-compile \\\n')
            output_file.write('    --no-diverse-batches \\\n')
            if not self.csim:
                output_file.write('    --no-csim \\\n')
            if self.init_cnt is not None:
                output_file.write('    --bayes-seed-cnt %d \\\n' %
                                  self.init_cnt)
            if self.base_cfg is not None:
                output_file.write('    --base-cfg %d \\\n' % self.base_cfg)
            if self.cfg_file is not None:
                cfg_file = os.path.join('../..', self.cfg_file)
                output_file.write('    --param-file %s \\\n' % cfg_file)
                output_file.write('    --no-tool-params \\\n')
                output_file.write('    --no-interf-params \\\n')
                output_file.write('    --use-64-bit-bus \\\n')
            output_file.write('    --no-bug-fixes \\\n')
            output_file.write('    --stop-after %d \\\n' % self.tuning_time)
            output_file.write('    --platform %s\n' % self.platform)

        os.chmod(script_file, os.stat(script_file).st_mode | stat.S_IXUSR) 
 
        cmd = [script_file]
        stdout_filename = os.path.join(output_dir, 'stdout.log')
        stderr_filename = os.path.join(output_dir, 'stderr.log')
        with open(stdout_filename, 'w') as stdout_file:
            with open(stderr_filename, 'w') as stderr_file:
                subprocess.run(cmd, stdout=stdout_file, stderr=stderr_file)


class RandomSearchTest(Test):
    def __init__(self):
        super().__init__('Random', 'bnn', 'zcu102')
        self.test = 'random'

class OpenTunerTest(Test):
    def __init__(self, benchmark, platform):
        super().__init__('AUCBanditMetaTechniqueA', benchmark, platform)
        self.test = 'opentuner_' + platform

class BayesTest(Test):
    def __init__(self, benchmark, platform):
        super().__init__('Bayes', benchmark, platform)
        self.async_compile = True
        self.test = 'bayes_' + platform

class PipelineTest(Test):
    def __init__(self, benchmark, platform, core_map):
        super().__init__('PipelinedMultiFidBayes', benchmark, platform)
        self.core_map = core_map
        self.csim = True
        self.test = 'pipe_' + platform
        self.subtest = benchmark + '_' + core_map

class NoErrorModelTest(BayesTest):
    def __init__(self):
        super().__init__('bnn', 'zcu102')
        self.avoid_errors = False
        self.test = 'no_error'
        self.subtest = ''

class NoSamplingTest(BayesTest):
    def __init__(self):
        super().__init__('bnn', 'zcu102')
        self.random_sampling = False
        self.test = 'no_sampling'
        self.subtest = ''

class InitCntTest(BayesTest):
    def __init__(self, init_cnt):
        super().__init__('bnn', 'zcu102')
        self.init_cnt = init_cnt
        self.test = 'init_cnt'
        self.subtest = str(init_cnt)

class SensitivityTest(Test):
    def __init__(self, base_cfg):
        super().__init__('ImpactAnalysis', 'bnn', 'zcu102')
        self.async_compile = True
        self.test = 'sens_anal'
        self.subtest = ''
        self.base_cfg = base_cfg

class UntunedTest(Test):
    def __init__(self, benchmark, platform):
        super().__init__('SingleBuild', benchmark, platform)
        self.seed_cnt = 1
        self.cfg_file = benchmark + '.yml'
        self.test = 'untuned_' + platform
        self.subtest = benchmark

def test_func(args):
    number, (test, seed) = args
    print('Test %d has started...' % number)
    try:
        test.run(seed)
    except Exception:
        print(traceback.format_exc())
    print('Test %d has finished...' % number)


parallelism = 2

# These are the experiments for Figure 8:
tests = [OpenTunerTest('bnn', 'zcu102'),
         RandomSearchTest(),
         PipelineTest('bnn', 'zcu102', '4x1.1x2.1x2')]

# Following are all experiments in the paper:
# tests = [UntunedTest('3d-rendering',      'pynq'),
#          UntunedTest('3d-rendering',      'ultra96'),
#          UntunedTest('3d-rendering',      'zcu102'),
#          UntunedTest('bnn',               'pynq'),
#          UntunedTest('bnn',               'ultra96'),
#          UntunedTest('bnn',               'zcu102'),
#          UntunedTest('digit-recognition', 'zcu102'),
#          UntunedTest('face-detection',    'zcu102'),
#          UntunedTest('optical-flow',      'ultra96'),
#          UntunedTest('optical-flow',      'zcu102'),
#          UntunedTest('spam-filter',       'pynq'),
#          UntunedTest('spam-filter',       'ultra96'),
#          UntunedTest('spam-filter',       'zcu102'),
#          RandomSearchTest(),
#          OpenTunerTest('3d-rendering',      'pynq'),
#          OpenTunerTest('3d-rendering',      'ultra96'),
#          OpenTunerTest('3d-rendering',      'zcu102'),
#          OpenTunerTest('bnn',               'pynq'),
#          OpenTunerTest('bnn',               'ultra96'),
#          OpenTunerTest('bnn',               'zcu102'),
#          OpenTunerTest('digit-recognition', 'zcu102'),
#          OpenTunerTest('face-detection',    'zcu102'),
#          OpenTunerTest('optical-flow',      'ultra96'),
#          OpenTunerTest('optical-flow',      'zcu102'),
#          OpenTunerTest('spam-filter',       'pynq'),
#          OpenTunerTest('spam-filter',       'ultra96'),
#          OpenTunerTest('spam-filter',       'zcu102'),
#          BayesTest('3d-rendering',      'pynq'),
#          BayesTest('3d-rendering',      'ultra96'),
#          BayesTest('3d-rendering',      'zcu102'),
#          BayesTest('bnn',               'pynq'),
#          BayesTest('bnn',               'ultra96'),
#          BayesTest('bnn',               'zcu102'),
#          BayesTest('digit-recognition', 'zcu102'),
#          BayesTest('face-detection',    'zcu102'),
#          BayesTest('optical-flow',      'ultra96'),
#          BayesTest('optical-flow',      'zcu102'),
#          BayesTest('spam-filter',       'pynq'),
#          BayesTest('spam-filter',       'ultra96'),
#          BayesTest('spam-filter',       'zcu102'),
#          PipelineTest('3d-rendering',      'pynq',    '1x1.2x2.3x1'),
#          PipelineTest('3d-rendering',      'ultra96', '1x1.2x2.3x1'),
#          PipelineTest('3d-rendering',      'zcu102',  '1x1.2x2.3x1'),
#          PipelineTest('bnn',               'pynq',    '4x1.1x2.1x2'),
#          PipelineTest('bnn',               'ultra96', '4x1.1x2.1x2'),
#          PipelineTest('bnn',               'zcu102',  '4x1.1x2.1x2'),
#          PipelineTest('bnn',               'zcu102',  '1x1.2x2.3x1'),
#          PipelineTest('bnn',               'zcu102',  '2x1.2x2.2x1'),
#          PipelineTest('bnn',               'zcu102',  '2x1.3x1.3x1'),
#          PipelineTest('digit-recognition', 'zcu102',  '2x1.2x2.2x1'),
#          PipelineTest('face-detection',    'zcu102',  '1x1.2x2.3x1'),
#          PipelineTest('optical-flow',      'pynq',    '4x1.1x2.1x2'),
#          PipelineTest('optical-flow',      'ultra96', '4x1.1x2.1x2'),
#          PipelineTest('optical-flow',      'zcu102',  '4x1.1x2.1x2'),
#          PipelineTest('spam-filter',       'pynq',    '1x1.2x2.3x1'),
#          PipelineTest('spam-filter',       'ultra96', '1x1.2x2.3x1'),
#          PipelineTest('spam-filter',       'zcu102',  '1x1.2x2.3x1'),
#          NoErrorModelTest(),
#          NoSamplingTest(),
#          InitCntTest(3),
#          InitCntTest(4),
#          InitCntTest(6),
#          InitCntTest(16)]

# The sensitivity analysis needs a configuration from a prior experiment to run.
# tests = [SensitivityTest(<Configuration>)]


thread_pool = ThreadPool(parallelism)
tests = [(test, seed) for test in tests for seed in range(test.seed_cnt)]
result = thread_pool.map_async(test_func, enumerate(tests))
thread_pool.close()
thread_pool.join()

