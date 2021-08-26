"""
Measurement interface for SDSoC

@author: Hans Giesen (giesen@seas.upenn.edu)
"""

import abc
import datetime
import glob
import logging
import os
import re
import shutil
import socket
import stat
import subprocess
import time
import traceback
import threading
import xml.etree.ElementTree
import yaml

from hostinterfaces import HostInterface
from opentuner import MeasurementInterface, Result
from parameters import ParameterGenerator, ParameterInjector
from utils import debug

log = logging.getLogger(__name__)


class SDSoCInterface(MeasurementInterface):
  """Measurement interface for SDSoC

  Attributes
  ----------
  tuner_cfg : dict
    Tuner configuration
  build_interf : HostInterface
    Interface to connect to the host used for building the application
  run_interf : HostInterface
    Interface to connect to the host used for measuring performance
  param_gen : ParameterGenerator
    Object for generating all parameters for the application
  param_injector : ParameterInjector
    Object for injecting all parameters in the application sources
  max_fidelity : int
    Number of fidelity levels
  lock : Lock
    Lock for sharing resources between build threads
  build_idx : int
    Sequence number of the next build
  build_map : dict
    A dictionary to map configuration hashes to build objects
  """

  def __init__(self, *pargs, **kwargs):
    tuner_cfg = kwargs.pop('tuner_cfg')
    self.tuner_cfg = tuner_cfg

    kwargs['project_name'] = tuner_cfg['project_name']
    kwargs['program_name'] = tuner_cfg['program_name']

    kwargs['program_version'] = self.get_version()

    super(SDSoCInterface, self).__init__(*pargs, **kwargs)

    self.parallel_compile = True

    self.build_interf = HostInterface.create(tuner_cfg['build_interf'], self)
    self.run_interf = HostInterface.create(tuner_cfg['run_interf'], self)

    self.connect_platform_manager()

    self.param_gen = ParameterGenerator(tuner_cfg, not self.args.no_corr_clk,
                                        not self.args.no_tool_params,
                                        not self.args.no_interf_params)
    self.param_injector = ParameterInjector(tuner_cfg,
                                            not self.args.no_interf_params,
                                            self.args.use_64_bit_bus,
                                            not self.args.no_bugfixes)

    self.max_fidelity = None
    
    self.lock = threading.Lock()

    self.build_idx = 0
    self.build_map = {}

  def connect_platform_manager(self):
    """Connect to the platform manager.
    """
    hostname = self.tuner_cfg['platform_manager']['hostname']
    port = self.tuner_cfg['platform_manager']['port']
    self.platform_manager = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.platform_manager.connect((hostname, port))
    request = 'client %s/%s\n' % (self.args.test, self.args.label)
    self.platform_manager.sendall(request.encode())

  def set_driver(self, driver):
    """Set the measurement driver.
    
    Parameters
    ----------
    driver : MeasurementDriver
    """
    super(SDSoCInterface, self).set_driver(driver)

    search_driver = self.driver.tuning_run_main.search_driver
    max_fidelity = search_driver.root_technique.max_fidelity

    self.max_fidelity = max_fidelity

  def manipulator(self):
    """Return an object that represents the parameters that can be tuned.

    Returns
    -------
    ConfigurationManipulator
      Description of parameters
    """
    return self.param_gen.generate()

  def compile(self, config_data, result_id, fidelity=0, max_threads=None):
    """Build hardware and software for one configuration.

    Parameters
    ----------
    config_data : dict
      Configuration
    result_id : int
      Sequence number of result
    fidelity : int
      Desired fidelity level or 0 if only complete builds are performed
    max_threads : int
      Number of build threads or None if the default value from the
      configuration file must be used

    Returns
    -------
    Result
      Result object
    """
    try:
      with self.lock:
        build_idx = self.build_idx
        self.build_idx += 1

      output_root = self.tuner_cfg['output_root']
      output_dir = os.path.join(output_root, "{0:04d}".format(build_idx))
      os.mkdir(output_dir)

      kwargs = {'sdsoc_interf': self,
                'output_dir': output_dir,
                'result_id': result_id,
                'build_idx': build_idx,
                'config_data': config_data,
                'max_threads': max_threads}
      if self.max_fidelity == 1:
        build = self.build(1, **kwargs)
        build_time = build.build_time
        mem_usage = build.mem_usage

        if build.result.state == 'OK': 
          build = self.build(2, **kwargs)
          if build.build_time:
            build_time += build.build_time
          if build.mem_usage:
            mem_usage += build.mem_usage

        if build.result.state == 'OK':
          build = self.build(3, **kwargs)
          if build.build_time:
            build_time += build.build_time
          if build.mem_usage:
            mem_usage += build.mem_usage

      else:
        build = self.build(fidelity, **kwargs)
        build_time = build.build_time
        mem_usage = build.mem_usage

      build.result.build_time = build_time
      build.result.mem_usage = mem_usage
      return build.result

    except Exception:
      log.error("Exception encountered: %s", traceback.format_exc())
      return Result(state='ERROR', msg='Unknown exception during compilation')

  def build(self, fidelity, **kwargs):
    """Run a build stage.

    Parameters
    ----------
    fidelity : int
      Fidelity level of build
    kwargs : dict
      Keyword arguments for build object

    Returns
    -------
    Build
      Build object
    """
    if fidelity == 1:
      build = Presynthesis(**kwargs)
    elif fidelity == 2:
      build = Synthesis(**kwargs)
    else:
      build = Implementation(**kwargs)

    manipulator = self.driver.tuning_run_main.manipulator
    cfg_hash = manipulator.hash_config(kwargs['config_data'])
    with self.lock:
      self.build_map.setdefault(cfg_hash, []).append(build)
    
    build.run()

    return build

  def run_precompiled(self, desired_result, inp, limit, result, result_id):
    """Evaluate one configuration on the platform.

    Parameters
    ----------
    desired_result : DesiredResult
      Desired result
    inp : Input
      Input
    limit : float
      Runtime limit in seconds
    result : Result
      Build result
    result_id : int
      Sequence number of result

    Returns
    -------
    Result
      Evaluation result
    """
    try:
      config_data = desired_result.configuration.data
      build = self.get_build(config_data, 3)
      if build is None or build.result.state != 'OK':
        return result
      
      build_idx = build.build_idx
      output_root = self.tuner_cfg['output_root']
      output_dir = os.path.join(output_root, "{0:04d}".format(build_idx))
      run = Run(sdsoc_interf=self, output_dir=output_dir, build=build,
                result_id=result_id)
      run.run()
      return run.result

    except Exception:
      log.error("Exception encountered: %s", traceback.format_exc())
      return Result(state='ERROR', msg='Unknown exception while running')

  def get_build(self, config_data, fidelity):
    """Look up and return build object.

    Parameters
    ----------
    config_data : dict
      Configuration
    fidelity : int
      Fidelity
    
    Returns
    -------
    Build
      Build object or None if the given build does not exist
    """
    manipulator = self.driver.tuning_run_main.manipulator
    cfg_hash = manipulator.hash_config(config_data)
    with self.lock:
      try:
        return self.build_map[cfg_hash][fidelity - 1]
      except IndexError:
        return None

  def get_version(self):
    """Returns a version identifying the code.

    Returns
    -------
    str
      Version
    """
    # The returned version should be unique under normal circumstances.  That
    # avoids that configurations will be reused between runs, which would mess
    # up run time experiments.
    time = datetime.datetime.now().strftime('%m/%d/%Y %H:%M')
    cmd = ["git", "-C", self.tuner_cfg['project_dir'], "describe", "--always"]
    null_device = open(os.devnull, 'w')
    try:
      git_commit = subprocess.check_output(cmd, stderr=null_device)
      version = time + ' ' + git_commit.strip()
    except subprocess.CalledProcessError:
      version = time
    return version

  def save_final_config(self, cfg):
    """Store the final configuration.

    Parameters
    ----------
    cfg : Configuration
      Configuration
    """
    pass


class Stage(object):
  """Abstract base class representing one step in a measurement

  Attributes
  ----------
  sdsoc_interf : SDSoCInterface
    SDSoC measurement interface that spawned this build
  output_dir : str
    Output directory
  tuner_cfg : dict
    Tuner configuration
  args : NameSpace
    Command-line arguments
  config_data : dict
    Configuration
  result : Result
    Result object for database
  result_id : int
    Sequence number of result
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, *pargs, **kwargs):
    self.sdsoc_interf = kwargs['sdsoc_interf']
    self.output_dir = kwargs['output_dir']
    self.config_data = kwargs['config_data']
    self.result_id = kwargs['result_id']
    self.tuner_cfg = self.sdsoc_interf.tuner_cfg
    self.args = self.sdsoc_interf.args
    self.result = Result()
  
  @abc.abstractmethod
  def run(self):
    """Run stage.
    """
    pass
  
  def write_cfg(self):
    """Store current configuration in a file.
    """
    with open(os.path.join(self.output_dir, 'cfg.yml'), 'w') as output_file:
      output_file.write("result_id: {}\n".format(self.result_id))
      output_file.write("cfg:\n")
      for key, value in sorted(self.config_data.items()):
        output_file.write("  {}: {}\n".format(key, value))


class Build(Stage):
  """Abstract base class representing one build stage

  Attributes
  ----------
  build_idx : int
    Sequence number of build
  fidelity : int
    Fidelity
  name : str
    Name of build stage
  abbrev : str
    Abbreviation for build stage
  max_threads : int
    Number of processor cores used by build
  prev_build : Build
    Build object of previous build stage
  build_time : float
    Build time in seconds
  mem_usage : float
    Peak memory usage in GB
  """
  
  __metaclass__ = abc.ABCMeta
  
  def __init__(self, *pargs, **kwargs):
    super(Build, self).__init__(*pargs, **kwargs)
    self.build_idx = kwargs['build_idx']
    self.fidelity = kwargs['fidelity']
    self.name = None
    self.abbrev = None
    self.max_threads = None
    self.prev_build = None
    self.build_time = float('inf')
    self.mem_usage = float('inf')

  def run(self):
    """Run stage.
    """
    project_dir = self.tuner_cfg['project_dir']
    fidelity = self.fidelity
    result = self.result
    if fidelity > 1:
      prev_build = self.sdsoc_interf.get_build(self.config_data, fidelity - 1)
      self.prev_build = prev_build
      input_dir = prev_build.output_dir
    else:
      input_dir = self.tuner_cfg['project_dir']

    log.info("Build %d: Running %s...", self.build_idx, self.name)

    if not getattr(self.sdsoc_interf.args, "use_prebuilt_" + self.abbrev):
      start = time.time()
      src_dir = os.path.join(input_dir, self.tuner_cfg['src_dir'])
      dst_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'])
      shutil.copytree(src_dir, dst_dir)
      end = time.time()
      result.copy_time = end - start

      self.write_cfg()

      self.prepare_files()

      self.write_wrapper_script()

      wrapper_script = os.path.join(self.output_dir, 'wrapper.bash')
      job_name = self.tuner_cfg['job_name']
      job_name = '%s_%d_%s' % (job_name, self.build_idx, self.abbrev)
      max_mem = self.tuner_cfg['max_mem'][self.abbrev]
      build_interf = self.sdsoc_interf.build_interf
      build_interf.run(wrapper_script, self.output_dir, job_name,
                       self.max_threads, max_mem)
    else:
      src_dir = os.path.join(project_dir, 'prebuilt', self.abbrev)
      shutil.copytree(src_dir, self.output_dir)
    
    with open(os.path.join(self.output_dir, 'host.yml'), 'r') as host_file:
      data = host_file.read()
    result.host = yaml.safe_load(data)

    self.load_metrics()
    self.process_result()
    self.copy_reports()

    start = time.time()
    self.cleanup()
    end = time.time()
    result.cleanup_time = end - start

    msg = "Build %d: %s (%s)" % (self.build_idx, result.msg, result.state)
    if result.state == 'OK':
      log.info(msg)
    else:
      log.error(msg)
  
  def write_wrapper_script(self):
    """Generates wrapper script.
    """
    src_dir = self.tuner_cfg['src_dir']
    tuner_root = self.tuner_cfg['tuner_root']
    timeout = 60 * self.tuner_cfg['timeouts'][self.abbrev]
    script = os.path.join(self.output_dir, 'build.bash')
    filename = os.path.join(self.output_dir, 'wrapper.bash')
    monitor_script = os.path.join(tuner_root, 'scripts/monitor.py')
    max_mem = self.tuner_cfg['max_mem'][self.abbrev]
    mem_log = os.path.join(self.output_dir, 'mem.log')

    with open(filename, 'w') as output_file:
      output_file.write('#!/bin/bash -e\n')
      
      if self.tuner_cfg['local_storage']:
        output_file.write('cleanup()\n')
        output_file.write('{\n')
        # We run this as another group to guarantee that the results are copied
        # back, even if this process is killed.
        output_file.write('  CMD="/usr/bin/time -f \'Rsync time: %e s\' ' \
                          'rsync -a ${RUN_DIR}/' + src_dir + ' ' + 
                          self.output_dir + '"\n')
        output_file.write('  setsid bash -c "${CMD} || true; rm -fr ' +
                          '${RUN_DIR}"\n')
        output_file.write('}\n')
        output_file.write('\n')
      
        output_file.write('trap cleanup EXIT INT\n')
      
      output_file.write('cd ' + self.output_dir + '\n')

      if self.tuner_cfg['local_storage']:
        output_file.write('mkdir -p /scratch/hudson\n')
        output_file.write('RUN_DIR=$(mktemp -d -p /scratch/hudson)\n')
      else:
        output_file.write('RUN_DIR=' + self.output_dir + '\n')

      script_file = os.path.join(tuner_root, 'scripts/get_machine.py')
      output_file.write(script_file + ' > host.yml\n')
      output_file.write('echo ${RUN_DIR} > build_dir.txt\n')

      if self.tuner_cfg['local_storage']:
        output_file.write('/usr/bin/time -f "Copy time: %e s" cp -rp ' +
                          src_dir + ' ${RUN_DIR}\n')

      output_file.write('cd ${RUN_DIR}\n')
      output_file.write('%s --timeout %e --max-mem %e --log %s %s\n' %
                        (monitor_script, timeout, max_mem, mem_log, script))

      output_file.write('if [ "$?" == 0 ]\n')
      output_file.write('then\n')
      output_file.write('  echo "Wrapper completed successfully."\n')
      output_file.write('fi\n')
    
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
  
  def process_result(self):
    """Process the result of the stage.
    """
    result = self.result

    with open(os.path.join(self.output_dir, 'stdout.log'), 'r') as log_file:
      stdout = log_file.read()
    with open(os.path.join(self.output_dir, 'stderr.log'), 'r') as log_file:
      stderr = log_file.read()

    found = re.search(r'Runtime: (\S+) s', stdout)
    if found:
      self.build_time = float(found.group(1))
    
    found = re.search(r'Peak memory usage: (\S+) GB', stdout)
    if found:
      self.mem_usage = float(found.group(1))
    
    if re.search(r'^Timeout expired.', stderr, re.MULTILINE):
      result.state = 'TIMEOUT'
      result.msg = self.name.capitalize() + ' timed out.'
      self.build_time = 60 * self.tuner_cfg['timeouts'][self.abbrev]
      return
  
    if re.search(r'^Out of memory.', stderr, re.MULTILINE):
      result.state = 'OOM'
      result.msg = self.name.capitalize() + ' ran out of memory.'
      return
  
    found = re.search(r'^ERROR: \[(.*)\] (.*)\n', stdout, re.MULTILINE)
    if found:
      result.state, result.msg = found.groups()
      return
      
    found = re.search(r'^ERROR: (.*)\n', stdout, re.MULTILINE)
    if found:
      result.state = 'ERROR'
      result.msg = found.group(1)
      return

    pattern = r'^Wrapper completed successfully.'
    if not re.search(pattern, stdout, re.MULTILINE):
      result.state = 'ERROR'
      result.msg = 'Unknown ' + self.name + ' error'
      return

    result.state = 'OK'
    result.msg = self.name.capitalize() + ' was successful.' 
  
  @abc.abstractmethod
  def prepare_files(self):
    """Prepare files for the build.
    """
    pass

  @abc.abstractmethod
  def load_metrics(self):
    """Retrieve the metrics from the output files.
    """
    pass

  @abc.abstractmethod
  def copy_reports(self):
    """Bring reports to safety before the build directory is deleted.
    """
    pass

  @abc.abstractmethod
  def cleanup(self):
    """Delete output files that are no longer needed.
    """
    pass


class Presynthesis(Build):
  """Class representing one presynthesis

  Attributes
  ----------
  run_time : float
    This attribute is used to pass the runtime from presynthesis to synthesis.
    Although the runtime is also stored in the result attribute, errors occur
    when accessing it, presumable because synthesis may use a different thread.
  """
  def __init__(self, *pargs, **kwargs):
    kwargs['output_dir'] = os.path.join(kwargs['output_dir'], 'presynth')
    kwargs['fidelity'] = 1
    super(Presynthesis, self).__init__(*pargs, **kwargs)
    self.name = 'presynthesis'
    self.abbrev = 'presynth'
    self.max_threads = 1
    self.run_time = None

  def prepare_files(self):
    """Prepare files for the build.
    """
    src_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'])
    param_injector = self.sdsoc_interf.param_injector
    param_injector.inject(src_dir, self.config_data)
      
    self.write_build_script()
    self.write_tcl_script()
    if self.tuner_cfg.get('csim', False) and not self.args.no_csim:
      self.write_csim_script()
  
  def write_build_script(self):
    """Generates build script.
    """
    tcl_file = os.path.join(self.output_dir, 'config.tcl')

    filename = os.path.join(self.output_dir, 'build.bash') 
    with open(filename, 'w') as output_file:
      output_file.write('#!/bin/bash -e\n')

      output_file.write('source ${SDSOC_ROOT}/settings64.sh\n')

      output_file.write('cd ' + self.tuner_cfg['src_dir'] + '\n')

      if self.tuner_cfg.get('csim', False) and not self.args.no_csim:
        csim_file = os.path.join(self.output_dir, 'csim.tcl')
        output_file.write('vivado_hls ' + csim_file + ' -l csim.log ' \
                          '> /dev/null || true\n')
        output_file.write('cd csim/solution1/csim/build\n')
        output_file.write('make -f csim.mk\n')
        output_file.write('./sim.sh\n')
        output_file.write('cd -\n')

      project_dir = self.tuner_cfg['project_dir']
      src_gen = self.tuner_cfg.get('src_gen')
      if src_gen:
        tuner_root = self.tuner_cfg['tuner_root']
        python3_path = os.path.join(tuner_root, 'python3_env/bin/python3')
        cfg_file = os.path.join(self.output_dir, 'cfg.yml')
        output_file.write(python3_path + ' ' + src_gen + ' ' +
                          project_dir + ' ' + cfg_file + '\n')
     
      config_data = self.config_data

      clk_freqs = self.tuner_cfg['platform']['clk_freqs']
      kernel_clk = clk_freqs.index(config_data['KERNEL_CLOCK'])
      data_mover_clk = clk_freqs.index(config_data['DATA_MOVER_CLOCK'])
    
      bus_width = int(self.tuner_cfg['platform']['axi_bus_width'])

      hls_src_file = self.tuner_cfg['hls_src_files'][0]
      hls_src_files = self.tuner_cfg['hls_src_files'][1:]

      args = []
      args.append('-sds-pf ' + self.tuner_cfg['platform']['dir'])
      args.append('-sds-hw ' + self.tuner_cfg['accel_func'] + ' ' +
                  hls_src_file)
      if hls_src_files != []:
        args.append('-files ' + ','.join(hls_src_files))
      args.append('-clkid %d' % kernel_clk)
      args.append('-hls-tcl ' + tcl_file + ' -sds-end')
      args.append('-dm-sharing %d' % config_data['DATA_MOVER_SHARING'])
      args.append('-dmclkid %d' % data_mover_clk)
      args.append('-maxjobs %d' % self.tuner_cfg['max_jobs'])
      args.append('-maxthreads %d' % self.tuner_cfg['max_threads'])
      args.append('-mno-bitstream')
      args = ' '.join(args)

      platform = self.args.platform
      debug_flag = '-g' if self.tuner_cfg['debug'] else ''
      opt_flag = '-O%d' % self.tuner_cfg['opt_level']
      cflags = self.tuner_cfg['cflags'].format(project_dir=project_dir,
                                               platform=platform)

      param_injector = self.sdsoc_interf.param_injector
      defines = param_injector.get_defines(config_data)

      obj_files = []
      src_files = self.tuner_cfg['src_files'] + self.tuner_cfg['hls_src_files']
      for src_file in src_files:
        obj_file = re.sub(r'\.c(pp)?$', '.o', src_file)
        obj_files.append(obj_file)
        tool = 'sds++' if src_file.endswith('.cpp') else 'sdscc'
        output_file.write(tool + ' -c ' + debug_flag + ' ' + opt_flag + ' ' +
                          cflags + ' ' + args + ' ' + defines + ' ' +
                          src_file + ' -o ' + obj_file + '\n')

      ldflags = self.tuner_cfg['ldflags'].format(project_dir=project_dir,
                                                 platform=platform)
      obj_files = ' '.join(obj_files)
      exe_file = self.tuner_cfg['exe_file']
      output_file.write(tool + ' ' + opt_flag + ' ' + ldflags + ' ' + args +
                        ' ' + obj_files + ' -o ' + exe_file + '\n')

    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
    
  def write_tcl_script(self):
    """Generates TCL script to set clock uncertainty.
    """
    tcl_file = os.path.join(self.output_dir, 'config.tcl')
    with open(tcl_file, 'w') as output_file:
      uncertainty = self.config_data['CLOCK_UNCERTAINTY']
      if not self.args.no_bugfixes or self.args.use_64_bit_bus:
        output_file.write('set_clock_uncertainty %f%%\n' % uncertainty)
      else:
        output_file.write('set_clock_uncertainty %d%%\n' % uncertainty)
  
  def write_csim_script(self):
    """Generates TCL script for C simulation.
    """
    tcl_file = os.path.join(self.output_dir, 'csim.tcl')
    project_dir = self.tuner_cfg['project_dir']
    platform = self.args.platform
    param_injector = self.sdsoc_interf.param_injector
    defines = param_injector.get_defines(self.config_data)
    cflags = self.tuner_cfg['cflags']
    cflags = cflags.format(project_dir=project_dir, platform=platform)
    csim_args = self.tuner_cfg['csim_args'].format(project_dir=project_dir)
    hls_src_files = ' '.join(self.tuner_cfg['hls_src_files'])
    src_files = ' '.join(self.tuner_cfg['src_files'])
    with open(tcl_file, 'w') as output_file:
      output_file.write('set defines "%s"\n' % defines)
      output_file.write('set cflags "%s $defines"\n' % cflags)
      output_file.write('set args "%s"\n' % csim_args)
      output_file.write('open_project csim\n')
      output_file.write('add_files "%s" -cflags $cflags\n' % hls_src_files)
      output_file.write('add_files -tb "%s" -cflags $cflags\n' % src_files)
      output_file.write('open_solution "solution1" -reset\n')
      output_file.write('csim_design -O -argv $args\n')
      output_file.write('exit\n')

  def process_result(self):
    """Process the result of the stage.
    """
    super(Presynthesis, self).process_result()

    with open(os.path.join(self.output_dir, 'stdout.log'), 'r') as log_file:
      lines = log_file.read()
    if re.search(r'Invalid parameter combination.', lines, re.MULTILINE):
      self.result.state = 'INVALID'
      self.result.msg = 'Invalid parameters.'
    if re.search(r'^TEST FAILED', lines, re.MULTILINE):
      self.result.state = 'CSIM'
      self.result.msg = 'C-simulation failed.'

  def load_metrics(self):
    """Retrieve the metrics from the output files.
    """
    src_dir = self.tuner_cfg['src_dir']
    accel_func = self.tuner_cfg['accel_func']
    xml_file = os.path.join(self.output_dir, src_dir, '_sds/vhls', accel_func,
                            'solution/syn/report/csynth.xml')
    try:
      tree = xml.etree.ElementTree.parse(xml_file)
    except IOError:
      return

    node = tree.find('PerformanceEstimates/SummaryOfOverallLatency')
    # Average case latency would be better, but Vivado HLS sets it to
    # undefined when it is outside the range determined by static analysis.
    latency = int(node.find('Worst-caseLatency').text)
    kernel_freq = self.config_data['KERNEL_CLOCK'] * 1e6
    proc_freq = float(self.tuner_cfg['platform']['proc_freq'])
    result = self.result
    result.run_time = latency / kernel_freq * proc_freq
    self.run_time = result.run_time

    node = tree.find('AreaEstimates/Resources')
    result.luts = int(node.find('LUT').text)
    result.regs = int(node.find('FF').text)
    result.brams = int(node.find('BRAM_18K').text)
    result.dsps = int(node.find('DSP48E').text)
  
  def copy_reports(self):
    """Bring reports to safety before the build directory is deleted.
    """
    src_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'])
    reports_dir = os.path.join(self.output_dir, 'reports', self.abbrev)
    os.makedirs(reports_dir)
    hls_dir = os.path.join(src_dir, '_sds/vhls')
    accel = self.tuner_cfg['accel_func']
    reports = os.path.join(hls_dir, accel, 'solution/syn/report/*')
    for report in glob.glob(reports):
      shutil.copy(report, reports_dir)
    hls_log = os.path.join(hls_dir, accel + '_vivado_hls.log')
    try:
      shutil.copy(hls_log, reports_dir)
    except IOError:
      pass
    csim_log = os.path.join(src_dir, 'csim.log')
    try:
      shutil.copy(csim_log, reports_dir)
    except IOError:
      pass

  def cleanup(self):
    """Delete output files that are no longer needed.
    """
    if self.result.state in ['OK', 'TIMEOUT', 'INVALID', 'XFORM 203-504']:
      hls_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'],
                             '_sds/vhls')
      shutil.rmtree(hls_dir, ignore_errors=True)


class Synthesis(Build):
  """Class representing one synthesis
  """
  def __init__(self, *pargs, **kwargs):
    kwargs['output_dir'] = os.path.join(kwargs['output_dir'], 'synth')
    kwargs['fidelity'] = 2
    super(Synthesis, self).__init__(*pargs, **kwargs)
    self.name = 'synthesis'
    self.abbrev = 'synth'
    if kwargs['max_threads'] is not None:
      self.max_threads = kwargs['max_threads']
    else:
      self.max_threads = self.tuner_cfg['max_jobs']

  def prepare_files(self):
    """Prepare files for the build.
    """
    self.write_build_script()
    self.write_tcl_script()

  def write_build_script(self):
    """Generates build script.
    """
    param_injector = self.sdsoc_interf.param_injector
    args = param_injector.get_vpl_args(self.config_data)

    platform_dir = self.tuner_cfg['platform']['dir']
    platform_name = self.args.platform + '.xpfm'
    platform_file = os.path.join(platform_dir, platform_name)
    accel_func = self.tuner_cfg['accel_func']
    tcl_file = os.path.join(self.output_dir, 'synth.tcl')

    filename = os.path.join(self.output_dir, 'build.bash') 
    with open(filename, 'w') as output_file:
      output_file.write('#!/bin/bash -e\n')

      output_file.write('source ${SDSOC_ROOT}/settings64.sh\n')

      output_file.write('cd ' + self.tuner_cfg['src_dir'] + '\n')

      output_file.write('vpl --iprepo _sds/iprepo/repo \\\n')
      output_file.write('    --iprepo ${SDSOC_ROOT}/data/ip/xilinx \\\n')
      output_file.write('    --platform ' + platform_file + ' \\\n')
      output_file.write('    --temp_dir _sds/p0 \\\n')
      output_file.write('    --output_dir _sds/p0/vpl \\\n')
      output_file.write('    --input_file _sds/p0/.xsd/top.bd.tcl \\\n')
      output_file.write('    --target hw \\\n')
      output_file.write('    --save_temps \\\n')
      output_file.write('    --kernels ' + accel_func + ':adapter \\\n')
      if args != '':
        output_file.write(args + '\n')
      output_file.write('    --export_script\n')

      output_file.write('vivado -nolog -nojournal -mode batch -source ' +
                        tcl_file + '\n')
    
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

  def write_tcl_script(self):
    """Generates TCL script.
    """
    max_jobs = self.tuner_cfg['max_jobs']
    tcl_file = os.path.join(self.output_dir, 'synth.tcl')
    with open(tcl_file, 'w') as output_file:
      output_file.write('open_project _sds/p0/vivado/vpl/prj/prj.xpr\n')

      output_file.write('reset_run synth_1\n')
      output_file.write('launch_runs synth_1 -jobs %d\n' % max_jobs)
      output_file.write('wait_on_run synth_1\n')

      output_file.write('open_run synth_1 -name synth_1\n')
      output_file.write('report_utilization\n')

      output_file.write('set run [get_runs synth_1]\n')
      output_file.write('set status [get_property STATUS $run]\n')
      output_file.write('set progress [get_property PROGRESS $run]\n')
      output_file.write('if {$status == "synth_design Complete!" && ')
      output_file.write('$progress == "100%"} {\n')
      output_file.write('  exit 0\n')
      output_file.write('}\n')
      output_file.write('puts "Synthesis failed."\n')
      output_file.write('exit 1\n')
    
    os.chmod(tcl_file, os.stat(tcl_file).st_mode | stat.S_IXUSR)
  
  def process_result(self):
    """Process the result of the stage.
    """
    super(Synthesis, self).process_result()

  def load_metrics(self):
    """Retrieve the metrics from the output files.
    """
    self.result.run_time = self.prev_build.run_time
    with open(os.path.join(self.output_dir, 'stdout.log'), 'r') as log_file:
      lines = log_file.read()
    if re.search(r'Utilization Design Information', lines):
      pattern = r'\| (CLB|Slice) LUTs\*?\s+\|\s+(\d+)\s+\|'
      self.result.luts = int(re.search(pattern, lines).group(2))
      pattern = r'\| (CLB|Slice) Registers\s+\|\s+(\d+)\s+\|'
      self.result.regs = int(re.search(pattern, lines).group(2))
      pattern = r'\|   RAMB36/FIFO\*\s+\|\s+(\d+)\s+\|'
      bram36s = int(re.search(pattern, lines).group(1))
      pattern = r'\|   RAMB18\s+\|\s+(\d+)\s+\|'
      bram18s = int(re.search(pattern, lines).group(1))
      self.result.brams = 2 * bram36s + bram18s
      match = re.search(r'\| DSPs\s+\|\s+(\d+)\s+\|', lines)
      self.result.dsps = int(match.group(1))
  
  def copy_reports(self):
    """Bring reports to safety before the build directory is deleted.
    """
    pass

  def cleanup(self):
    """Delete output files that are no longer needed.
    """
    if self.result.state in ['OK', 'TIMEOUT']:
      src_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'])
      cache_dir = os.path.join(src_dir, '_sds/p0/vivado/vpl/prj/prj.cache')
      shutil.rmtree(cache_dir, ignore_errors=True)

    prev_output_dir = self.prev_build.output_dir
    prev_src_dir = os.path.join(prev_output_dir, self.tuner_cfg['src_dir'])
    shutil.rmtree(os.path.join(prev_src_dir, '_sds'), ignore_errors=True)

    
class Implementation(Build):
  """Class representing one implementation
  """
  def __init__(self, *pargs, **kwargs):
    kwargs['output_dir'] = os.path.join(kwargs['output_dir'], 'impl')
    kwargs['fidelity'] = 3
    super(Implementation, self).__init__(*pargs, **kwargs)
    self.name = 'implementation'
    self.abbrev = 'impl'
    if kwargs['max_threads'] is not None:
      self.max_threads = kwargs['max_threads']
    else:
      self.max_threads = self.tuner_cfg['max_jobs']

  def prepare_files(self):
    """Prepare files for the build.
    """
    self.write_build_script()
    self.write_tcl_script()
    self.write_bif_file()

  def write_build_script(self):
    """Generates build script.
    """
    tcl_file = os.path.join(self.output_dir, 'impl.tcl')
    bit_file = self.tuner_cfg['exe_file'] + '.bit'

    filename = os.path.join(self.output_dir, 'build.bash') 
    with open(filename, 'w') as output_file:
      output_file.write('#!/bin/bash -e\n')

      output_file.write('source ${SDSOC_ROOT}/settings64.sh\n')

      output_file.write('cd ' + self.tuner_cfg['src_dir'] + '\n')

      output_file.write('vivado -nolog -nojournal -mode batch -source ' +
                        tcl_file + '\n')

      output_file.write('cp _sds/p0/vpl/address_map.xml _sds/p0/.cf_work\n')
      output_file.write('MAKEFILE=_sds/swstubs/Makefile.pi_relink\n')
      output_file.write('OLD_DIR=$(sed -n "s|.*:=\(.*\)/_sds/.*|\\1|p" ' +
                        '${MAKEFILE})\n')
      output_file.write('sed "s:${OLD_DIR}:${PWD}:g" ${MAKEFILE} > ' + 
                        '${MAKEFILE}.fixed\n')
      output_file.write('make -f ${MAKEFILE}.fixed\n')

      output_file.write('cp _sds/p0/vivado/vpl/prj/prj.runs/impl_1/' +
                        '*_wrapper.bit ' + bit_file + '\n')

      if self.args.platform == 'pynq':
        output_file.write('bootgen -image fpga.bif -w -process_bitstream bin\n')
    
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

  def write_tcl_script(self):
    """Generates TCL script.
    """
    tcl_file = os.path.join(self.output_dir, 'impl.tcl')
    max_threads = self.tuner_cfg['max_threads']
    with open(tcl_file, 'w') as output_file:
      output_file.write('open_project _sds/p0/vivado/vpl/prj/prj.xpr\n')

      output_file.write('set_param general.maxThreads %d\n' % max_threads)

      output_file.write('reset_run impl_1\n')
      output_file.write('launch_runs impl_1 -to_step write_bitstream\n')
      output_file.write('wait_on_run impl_1\n')

      output_file.write('open_run impl_1 -name impl_1\n')
      output_file.write('report_utilization\n')

      output_file.write('set run [get_runs impl_1]\n')
      output_file.write('set status [get_property STATUS $run]\n')
      output_file.write('set progress [get_property PROGRESS $run]\n')
      output_file.write('if {$status == "write_bitstream Complete!" && ')
      output_file.write('$progress == "100%"} {\n')
      output_file.write('  exit 0\n')
      output_file.write('}\n')
      output_file.write('puts "Implementation failed."\n')
      output_file.write('exit 1\n')
    
    os.chmod(tcl_file, os.stat(tcl_file).st_mode | stat.S_IXUSR)

  def write_bif_file(self):
    """Generates BIF file.
    """
    if self.args.platform == 'pynq':
      bit_file = os.path.join('.', self.tuner_cfg['exe_file'] + '.bit')
      src_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'])
      with open(os.path.join(src_dir, 'fpga.bif'), 'w') as output_file:
        output_file.write('all: { ' + bit_file + '}\n')

  def process_result(self):
    """Process the result of the stage.
    """
    super(Implementation, self).process_result()
    
    with open(os.path.join(self.output_dir, 'stdout.log'), 'r') as log_file:
      lines = log_file.read()    
    if re.search(r'design did not meet timing', lines, re.MULTILINE):
      self.result.state = 'TIMING'
      self.result.msg = 'Timing not met.'
      return

  def load_metrics(self):
    """Retrieve the metrics from the output files.
    """
    with open(os.path.join(self.output_dir, 'stdout.log'), 'r') as log_file:
      lines = log_file.read()

    cell_map = {'LUT': 'luts', 'Register': 'regs', 'RAMB18': 'brams'}
    for line in lines.split('\n'):
      match = re.search(r'\[Place 30-640\].*more (\S+).*requires (\d+) ', line)
      if match:
        cell, cnt = match.groups()
        if cell in cell_map:
          setattr(self.result, cell_map[cell], int(cnt))
      
    if re.search(r'Utilization Design Information', lines):
      pattern = r'\| (CLB|Slice) LUTs\*?\s+\|\s+(\d+)\s+\|'
      self.result.luts = int(re.search(pattern, lines).group(2))
      pattern = r'\| (CLB|Slice) Registers\s+\|\s+(\d+)\s+\|'
      self.result.regs = int(re.search(pattern, lines).group(2))
      pattern = r'\|   RAMB36/FIFO\*\s+\|\s+(\d+)\s+\|'
      bram36s = int(re.search(pattern, lines).group(1))
      pattern = r'\|   RAMB18\s+\|\s+(\d+)\s+\|'
      bram18s = int(re.search(pattern, lines).group(1))
      self.result.brams = 2 * bram36s + bram18s
      match = re.search(r'\| DSPs\s+\|\s+(\d+)\s+\|', lines)
      self.result.dsps = int(match.group(1))

  def copy_reports(self):
    """Bring reports to safety before the build directory is deleted.
    """
    src_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'])
    reports_dir = os.path.join(self.output_dir, 'reports', self.abbrev)
    os.makedirs(reports_dir)
    for report in glob.glob(os.path.join(src_dir, '_sds/reports/*')):
      shutil.copy(report, reports_dir)

  def cleanup(self):
    """Delete output files that are no longer needed.
    """
    if self.result.state in ['OK', 'TIMEOUT', 'TIMING', 'Place 30-640']:
      src_dir = os.path.join(self.output_dir, self.tuner_cfg['src_dir'])
      shutil.rmtree(os.path.join(src_dir, '_sds'), ignore_errors=True)

    prev_output_dir = self.prev_build.output_dir
    prev_src_dir = os.path.join(prev_output_dir, self.tuner_cfg['src_dir'])
    shutil.rmtree(os.path.join(prev_src_dir, '_sds'), ignore_errors=True)


class Run(Stage):
  """Class representing one platform run

  Parameters
  ----------
  build : str
    Build to be run on the platform
  """
  def __init__(self, *pargs, **kwargs):
    kwargs['output_dir'] = os.path.join(kwargs['output_dir'], 'run')
    build = kwargs['build']
    kwargs['config_data'] = build.config_data
    super(Run, self).__init__(*pargs, **kwargs)
    self.build = build
    self.result = self.build.result
  
  def run(self):
    """Run stage.
    """
    os.makedirs(self.output_dir) 

    log.info("Build %d: Running on platform...", self.build.build_idx)

    host = self.acquire_platform()

    try:
      self.write_cfg()
      self.write_launch_script(host)
      self.write_run_script()
      self.write_wrapper_script()
          
      wrapper_script = os.path.join(self.output_dir, 'wrapper.bash')
      job_name = "%s_%d_run" % (self.tuner_cfg['job_name'], self.build.build_idx)
      max_mem = self.tuner_cfg['max_mem']['run']
      run_interf = self.sdsoc_interf.run_interf
      run_interf.run(wrapper_script, self.output_dir, job_name, 1, max_mem)
    finally:
      self.release_platform()

    self.process_result()
    
    result = self.result
    msg = "Build %d: %s (%s)" % (self.build.build_idx, result.msg, result.state)
    if result.state == 'OK':
      log.info(msg)
    else:
      log.error(msg)

  def acquire_platform(self):
    """Obtain a platform hostname from the platform manager.

    Returns
    -------
    str
      Hostname
    """
    request = ('acquire %s\n' % self.args.platform).encode()
    self.sdsoc_interf.platform_manager.sendall(request)
    response = self.sdsoc_interf.platform_manager.recv(64)
    host = response.decode().strip()
    log.info('Running on %s...', host)
    return host

  def release_platform(self):
    """Return the acquired platform to the platform manager.
    """
    request = ('release\n').encode()
    self.sdsoc_interf.platform_manager.sendall(request)

  def write_launch_script(self, host):
    """Generates script for launching application of platform.

    Parameters
    ----------
    host : str
      Hostname of platform
    """
    exe_file = self.tuner_cfg['exe_file']
    bit_file = exe_file + '.bit'
    if self.args.platform == 'pynq':
      bit_file += '.bin'
    run_script = os.path.join(self.output_dir, 'run.bash')
    sources = bit_file + ' ' + exe_file + ' ' + run_script
    impl_dir = os.path.join(self.build.output_dir, self.tuner_cfg['src_dir'])
    filename = os.path.join(self.output_dir, 'launch.bash')
    with open(filename, 'w') as output_file:
      output_file.write('#!/bin/bash -e\n')
      output_file.write('RUN_DIR=/run/media/mmcblk0p1\n')
      output_file.write('echo "Copying files to target..."\n')
      output_file.write('cd ' + impl_dir + '\n')
      output_file.write('scp -o ConnectTimeout=10 ' + sources + ' ' +
                        host + ':/run/media/mmcblk0p1\n')
      output_file.write('ssh -o ConnectTimeout=10 ' + host +
                        ' /run/media/mmcblk0p1/run.bash\n')
    
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

  def write_run_script(self):
    """Generates script that must be run on the platform.
    """
    exe_file = self.tuner_cfg['exe_file']
    bit_file = exe_file + '.bit'
    if self.args.platform == 'pynq':
      bit_file += '.bin'
    run_args = self.tuner_cfg['run_args']
    filename = os.path.join(self.output_dir, 'run.bash')
    with open(filename, 'w') as output_file:
      output_file.write('#!/bin/bash -e\n')
      output_file.write('mkdir -p /lib/firmware\n')
      output_file.write('cd /run/media/mmcblk0p1\n')
      output_file.write('cp ' + bit_file + ' /lib/firmware\n')
      output_file.write('echo Programming platform...\n')
      output_file.write('echo 0 > /sys/class/fpga_manager/fpga0/flags\n')
      output_file.write('echo ' + bit_file +
                        ' > /sys/class/fpga_manager/fpga0/firmware\n')
      output_file.write('echo "Running application..."\n')
      output_file.write('./' + exe_file + ' ' + run_args + '\n')
    
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

  def write_wrapper_script(self):
    """Generates wrapper script.
    """
    tuner_root = self.tuner_cfg['tuner_root']
    timeout = 60 * self.tuner_cfg['timeouts']['run']
    script = os.path.join(self.output_dir, 'launch.bash')
    filename = os.path.join(self.output_dir, 'wrapper.bash')

    with open(filename, 'w') as output_file:
      output_file.write('#!/bin/bash -e\n')
      output_file.write('cleanup()\n')
      output_file.write('{\n')
      # The timeout tool changes its process group, so we have to kill it
      # explicitly.
      output_file.write('  if [ -n "${PID}" ]\n')
      output_file.write('  then\n')
      output_file.write('    kill "${PID}" 2> /dev/null || true\n')
      output_file.write('  fi\n')
      output_file.write('}\n')
      output_file.write('\n')
      
      output_file.write('trap cleanup EXIT INT\n')
      
      output_file.write('cd ' + self.output_dir + '\n')

      script_file = os.path.join(tuner_root, 'scripts/get_machine.py')
      output_file.write(script_file + ' > host.yml\n')
      
      # We apply a timeout here because we don't want to limit the time needed
      # for issuing a job to the grid.
      output_file.write(('/usr/bin/timeout %d /usr/bin/time -f ' +
                         '"Runtime: %%e s" -o /dev/stdout %s &\n') %
                        (timeout, script))
      output_file.write('PID=$!\n')
      output_file.write('wait -n && EXIT_CODE=0 || EXIT_CODE=$?\n')

      output_file.write('if [ "${EXIT_CODE}" == 124 ]\n')
      output_file.write('then\n')
      output_file.write('  echo "Timeout in wrapper."\n')
      output_file.write('elif [ "${EXIT_CODE}" == 0 ]\n')
      output_file.write('then\n')
      output_file.write('  echo "Wrapper completed successfully."\n')
      output_file.write('fi\n')
    
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
  
  def process_result(self):
    """Process the result of the stage.
    """
    result = self.result

    with open(os.path.join(self.output_dir, "stdout.log"), 'r') as log_file:
      lines = log_file.read()

    if re.search(r'Run timed out.', lines) != None:
      result.state = 'TIMEOUT'
      result.msg = 'Run timed out.'
      return

    if not re.search(r'TEST PASSED', lines):
      result.state = 'ERROR'
      result.msg = 'Run did not pass test.'
      return

    found = re.search(r'The hardware test took (\S+) cycles.', lines)
    cycles = int(found.group(1))
    result.state = 'OK'
    result.msg = 'Run was successful.'
    result.run_time = cycles

