"""
Classes for parameter handling

Created on Jun 24, 2020
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

import logging
import os
import re

from opentuner.search.manipulator import (ConfigurationManipulator,
        BooleanParameter, EnumParameter, FloatParameter, LogIntegerParameter,
        PowerOfTwoParameter, CorrelatedEnumParameter)

log = logging.getLogger(__name__)


class ParameterGenerator(object):
  """Class for generating design parameters

  Attributes
  ----------
  tuner_cfg : dict
    Tuner configuration
  corr_clock : bool
    If false, clock frequencies are enumerated by integers, which may hide
    correlation.  If true, clock frequencies are represented as is.
  tune_tool_params : bool
    Whether to tune the tool parameters
  tune_interf_params : bool
    Whether to tune the accelerator interface parameters
  """
  def __init__(self, tuner_cfg, corr_clk, tune_tool_params,
               tune_interf_params):
    self.tuner_cfg = tuner_cfg
    self.corr_clk = corr_clk
    self.tune_tool_params = tune_tool_params
    self.tune_interf_params = tune_interf_params

  def generate(self):
    """Return a manipulator with all parameters that can be tuned.

    Returns
    -------
    ConfigurationManipulator
      Description of parameters
    """
    manipulator = ConfigurationManipulator()
    self.add_user_params(manipulator)
    if self.tune_interf_params:
      self.add_interf_params(manipulator)
    self.add_presynth_params(manipulator)
    if self.tune_tool_params:
      self.add_synth_params(manipulator)
      self.add_impl_params(manipulator)
    return manipulator
  
  def add_user_params(self, manipulator):
    """Add user-defined parameters to a configuration manipulator.

    Parameters
    ----------
    manipulator : ConfigurationManipulator
      Manipulator
    """
    params = self.tuner_cfg.get('params', {})
    params = {} if params is None else params
    for name, info in params.items():
      param_type = '{}Parameter'.format(info['type'])
      exec('from opentuner.search.manipulator import {}'.format(param_type))
      if 'args' in info:
        args = '"{}", {}'.format(name, info['args'])
      else:
        args = '"{}"'.format(name)
      param = eval('{}({})'.format(param_type, args))
      manipulator.add_parameter(param)

  def add_interf_params(self, manipulator):
    """Add accelerator interface parameters to a configuration manipulator.

    Parameters
    ----------
    manipulator : ConfigurationManipulator
      Manipulator
    """
    data_movers = ["AXIFIFO", "AXIDMA_SIMPLE", "AXIDMA_SG", "FASTDMA"]
    sys_ports = self.tuner_cfg['platform']['sys_ports']
    patterns = ["SEQUENTIAL", "RANDOM"]

    all_arrays = self.tuner_cfg.get('arrays')
    if not all_arrays:
      return
    for accel, arrays in all_arrays.items():
      for array, info in arrays.items():
        ext = '{}_{}'.format(accel, array)
        params = [EnumParameter("DATA_MOVER_{}".format(ext), data_movers),
                  EnumParameter("SYS_PORT_{}".format(ext), sys_ports),
                  EnumParameter("ACCESS_PATTERN_{}".format(ext), patterns),
                  BooleanParameter("ZERO_COPY_{}".format(ext)),
                  BooleanParameter("CONTIGUOUS_{}".format(ext)),
                  BooleanParameter("CACHEABLE_{}".format(ext))]
        for param in params:
          manipulator.add_parameter(param)

  def add_presynth_params(self, manipulator):
    """Add presynthesis parameters to a configuration manipulator.

    Parameters
    ----------
    manipulator : ConfigurationManipulator
      Manipulator
    """
    freqs = self.tuner_cfg['platform']['clk_freqs']
    if self.corr_clk:
      params = [CorrelatedEnumParameter("DATA_MOVER_CLOCK", freqs),
                CorrelatedEnumParameter("KERNEL_CLOCK", freqs),
                FloatParameter("CLOCK_UNCERTAINTY", 0, 100),
                EnumParameter("DATA_MOVER_SHARING", range(4))]
    else:
      params = [EnumParameter("DATA_MOVER_CLOCK", freqs),
                EnumParameter("KERNEL_CLOCK", freqs),
                FloatParameter("CLOCK_UNCERTAINTY", 0, 100),
                EnumParameter("DATA_MOVER_SHARING", range(4))]

    for param in params:
      manipulator.add_parameter(param)

  def add_synth_params(self, manipulator):
    """Add synthesis parameters to a configuration manipulator.

    Parameters
    ----------
    manipulator : ConfigurationManipulator
      Manipulator
    """
    directives = ["Default", "RuntimeOptimized", "AreaOptimized_high",
                  "AreaOptimized_medium", "AlternateRoutability",
                  "AreaMapLargeShiftRegToBRAM", "AreaMultThresholdDSP",
                  "FewerCarryChains"]
    fsm_encodings = ["auto", "one_hot", "sequential", "johnson", "gray",
                     "off"]

    params = [LogIntegerParameter("FANOUT_LIMIT", 1, 1000000),
              EnumParameter("SYNTH_DIRECTIVE", directives),
              BooleanParameter("RETIMING"),
              EnumParameter("FSM_ENCODING", fsm_encodings),
              BooleanParameter("KEEP_EQUIV_REGS"),
              BooleanParameter("RESOURCE_SHARING"),
              LogIntegerParameter("CTRL_SET_OPT_THRES", 0, 16),
              BooleanParameter("NO_LUT_COMBINING"),
              BooleanParameter("NO_SHIFT_REG_EXTRACTION"),
              LogIntegerParameter("SHIFT_REG_MIN_SIZE", 0, 100)]

    for param in params:
      manipulator.add_parameter(param)

  def add_impl_params(self, manipulator):
    """Add implementation parameters to a configuration manipulator.

    Parameters
    ----------
    manipulator : ConfigurationManipulator
      Manipulator
    """
    logic_opt_directives = ["Default", "Explore", "ExploreArea",
                            "ExploreSequentialArea", "AddRemap",
                            "RuntimeOptimized", "NoBramPowerOpt",
                            "ExploreWithRemap"]
    place_directives = ["Default", "Explore", "WLDrivenBlockPlacement",
                        "ExtraNetDelay_high", "ExtraNetDelay_low",
                        "AltSpreadLogic_low", "AltSpreadLogic_medium",
                        "AltSpreadLogic_high", "ExtraPostPlacementOpt",
                        "ExtraTimingOpt", "SSI_SpreadLogic_high",
                        "SSI_SpreadLogic_low", "SSI_SpreadSLLs",
                        "SSI_BalanceSLLs", "SSI_BalanceSLRs",
                        "SSI_HighUtilSLRs", "RuntimeOptimized", "Quick",
                        "EarlyBlockPlacement"]
    place_opt_directives = ["Default", "Explore", "ExploreWithHoldFix",
                            "AggressiveExplore", "AlternateReplication",
                            "AggressiveFanoutOpt", "AddRetime",
                            "AlternateFlowWithRetiming", "RuntimeOptimized",
                            "ExploreWithAggressiveHoldFix"]
    route_directives = ["Default", "Explore", "NoTimingRelaxation",
                        "MoreGlobalIterations", "HigherDelayCost",
                        "AdvancedSkewModeling", "RuntimeOptimized",
                        "Quick", "AlternateCLBRouting", "AggressiveExplore"]
    route_opt_directives = ["Default", "Explore", "AggressiveExplore",
                            "AddRetime", "ExploreWithAggressiveHoldFix"]

    params = [BooleanParameter("ENABLE_LOGIC_OPT"),
              EnumParameter("LOGIC_OPT_DIRECTIVE", logic_opt_directives),
              EnumParameter("PLACE_DIRECTIVE", place_directives),
              BooleanParameter("ENABLE_PLACE_OPT"),
              EnumParameter("PLACE_OPT_DIRECTIVE", place_opt_directives),
              EnumParameter("ROUTE_DIRECTIVE", route_directives),
              BooleanParameter("ENABLE_ROUTE_OPT"),
              EnumParameter("ROUTE_OPT_DIRECTIVE", route_opt_directives)]

    for param in params:
      manipulator.add_parameter(param)


class ParameterInjector(object):
  """Class for injecting design parameter values into sources

  Attributes
  ----------
  tuner_cfg : dict
    Tuner configuration
  tune_interf_params : bool
    Whether to tune the accelerator interface parameters
  use_64_bit_bus : bool
    Use a 64-bit bus instead of the optimal width.
  bugfixes : bool
    Apply fixes for bugs found after the paper deadline.
  """
  def __init__(self, tuner_cfg, tune_interf_params, use_64_bit_bus, bugfixes):
    self.tuner_cfg = tuner_cfg
    self.tune_interf_params = tune_interf_params
    self.use_64_bit_bus = use_64_bit_bus
    self.bugfixes = bugfixes

  def inject(self, src_dir, cfg_data):
    """Inject parameters into sources.
    
    Parameters
    ----------
    src_dir : str
      Source directory
    cfg_data : dict
      Dictionary assigning values to parameters
    """
    tuner_cfg = self.tuner_cfg
    if not tuner_cfg.get('arrays'):
      return
    if not self.tune_interf_params:
      return

    src_files = tuner_cfg['hls_hdr_files'] + tuner_cfg['hls_src_files']
    for src_file in src_files:
      self.add_pragmas(os.path.join(src_dir, src_file), cfg_data)

    self.gen_inc_file(os.path.join(src_dir, tuner_cfg['inc_file']), cfg_data)

  def add_pragmas(self, filename, cfg_data):
    """Add pragmas to a source file.

    Parameters
    ----------
    filename : str
      File to which pragmas must be added.
    cfg_data : dict
      Dictionary assigning values to parameters
    """
    orig_name = '{}.orig'.format(filename)
    try:
      os.rename(filename, orig_name)
    except OSError:
      return
    arrays = self.tuner_cfg.get('arrays')
    pattern = re.compile(r'\s*#pragma\s+tuner\s+(\S+)\s*$')
    with open(orig_name, 'r') as input_file:
      with open(filename, 'w') as output_file:
        for line in input_file:
          found = re.match(pattern, line)
          if found:
            accel = found.group(1)
            for idx, (array, info) in enumerate(arrays[accel].items()):
              self.generate_pragmas(accel, array, info, cfg_data, output_file)
          else:
            output_file.write(line)

  def generate_pragmas(self, accel, array, info, cfg_data, output_file):
    """Generate pragmas for one array input or output of an accelerator.

    Parameters
    ----------
    accel : str
      Accelerator
    array : str
      Array
    info : dict
      Information about the array
    cfg_data : dict
      Dictionary assigning values to parameters
    output_file : file
      Open file to which the pragmas must be added
    """
    ext = '{}_{}'.format(accel, array)

    if isinstance(info['size'], list):
      size = ''.join('[0:{}]'.format(elem) for elem in info['size'])
    else:
      size = '[0:{}]'.format(info['size'])

    zero_copy = cfg_data['ZERO_COPY_{}'.format(ext)]
    if zero_copy:
      pragma = '#pragma SDS data zero_copy({}{})\n'
      output_file.write(pragma.format(array, size))
    else:
      pragma = '#pragma SDS data copy({}{})\n'
      output_file.write(pragma.format(array, size))
    
      data_mover = cfg_data['DATA_MOVER_{}'.format(ext)]
      pragma = '#pragma SDS data data_mover({}:{})\n'
      output_file.write(pragma.format(array, data_mover))
  
    access_pattern = cfg_data['ACCESS_PATTERN_{}'.format(ext)]
    pragma = '#pragma SDS data access_pattern({}:{})\n'
    output_file.write(pragma.format(array, access_pattern))
 
    sys_port = cfg_data['SYS_PORT_{}'.format(ext)]
    pragma = '#pragma SDS data sys_port({}:{})\n'
    output_file.write(pragma.format(array, sys_port))

    contiguous = cfg_data['CONTIGUOUS_{}'.format(ext)]
    pragma = '#pragma SDS data mem_attribute({}:{}PHYSICAL_CONTIGUOUS)\n'
    prefix = '' if contiguous else 'NON_'
    output_file.write(pragma.format(array, prefix))

  def generate_malloc(self, accel, array, info, cfg_data, output_file):
    """Generate allocation function for a given array.
    
    Parameters
    ----------
    accel : str
      Accelerator that the array is passed to or from
    array : str
      Array
    info : dict
      Information about the array
    cfg_data : dict
      Dictionary assigning values to parameters
    output_file : file
      Open file to which the function must be added
    """
    ext = '{}_{}'.format(accel, array)
    contiguous = cfg_data['CONTIGUOUS_{}'.format(ext)]
    zero_copy = cfg_data['ZERO_COPY_{}'.format(ext)]
    cacheable = cfg_data['CACHEABLE_{}'.format(ext)]
    output_file.write('\n'
                      'inline void * {}(size_t size)\n'
                      '{{\n'.format(info['alloc']))
    if not contiguous and not zero_copy:
      output_file.write('  return malloc(size);\n')
    elif cacheable:
      output_file.write('  return sds_alloc(size);\n')
    else:
      output_file.write('  return sds_alloc_non_cacheable(size);\n')
    output_file.write('}\n')

  def generate_free(self, accel, array, info, cfg_data, output_file):
    """Generate deallocation function for a given array.
    
    Parameters
    ----------
    accel : str
      Accelerator that the array is passed to or from
    array : str
      Array
    info : dict
      Information about the array
    cfg_data : dict
      Dictionary assigning values to parameters
    output_file : file
      Open file to which the function must be added
    """
    ext = '{}_{}'.format(accel, array)
    contiguous = cfg_data['CONTIGUOUS_{}'.format(ext)]
    zero_copy = cfg_data['ZERO_COPY_{}'.format(ext)]
    output_file.write('\n'
                     'inline void {}(void * ptr)\n'
                     '{{\n'.format(info['free']))
    if not contiguous and not zero_copy:
      output_file.write('  free(ptr);\n')
    else:
      output_file.write('  sds_free(ptr);\n')
    output_file.write('}\n')

  def gen_inc_file(self, filename, cfg_data):
    """Generate header file with allocation and deallocation functions.
    
    Parameters
    ----------
    filename : str
      Filename
    cfg_data : dict
      Dictionary assigning values to parameters
    """
    with open(filename, 'w') as output_file:
      output_file.write('#include <stdlib.h>\n'
                     '#include <sds_lib.h>\n')
      for accel, arrays in self.tuner_cfg['arrays'].items():
        for array, info in arrays.items():
          self.generate_malloc(accel, array, info, cfg_data, output_file)
          self.generate_free(accel, array, info, cfg_data, output_file)
      output_file.write('\n')

  def get_defines(self, cfg_data):
    """Generate #defines for SDSoC compiler.

    Parameters
    ----------
    cfg_data : dict
      Dictionary assigning values to parameters

    Returns
    -------
    str
      #Defines
    """
    args = []
    params = self.tuner_cfg.get('params', {})
    params = {} if params is None else params
    for param, info in params.items():
      value = cfg_data[param]
      if info['type'] == 'Boolean':
        value = int(value)
      args.append('-D{}={}'.format(param, value))
    if self.use_64_bit_bus:
      args.append('-DAXI_BUS_WIDTH=64')
    else:
      bus_width = int(self.tuner_cfg['platform']['axi_bus_width'])
      args.append('-DAXI_BUS_WIDTH={}'.format(bus_width))
    if self.bugfixes:
      args.append('-DBUGFIXES')
    if self.tune_interf_params and self.bugfixes:
      args.append('-DTUNE_INTERF_PARAMS')
    return ' '.join(args)

  def get_vpl_args(self, cfg_data):
    """Generate command-line arguments for VPL.

    Parameters
    ----------
    cfg_data : dict
      Dictionary assigning values to parameters

    Returns
    -------
    str
      VPL arguments
    """
    param_map = \
        {'FANOUT_LIMIT':            (0, 'FANOUT_LIMIT'),
         'SYNTH_DIRECTIVE':         (0, 'DIRECTIVE'),
         'RETIMING':                (0, 'RETIMING'),
         'FSM_ENCODING':            (0, 'FSM_EXTRACTION'),
         'KEEP_EQUIV_REGS':         (0, 'KEEP_EQUIVALENT_REGISTERS'),
         'RESOURCE_SHARING':        (0, 'RESOURCE_SHARING'),
         'CTRL_SET_OPT_THRES':      (0, 'CONTROL_SET_OPT_THRESHOLD'),
         'NO_LUT_COMBINING':        (0, 'NO_LC'),
         'NO_SHIFT_REG_EXTRACTION': (0, 'NO_SRLEXTRACT'),
         'SHIFT_REG_MIN_SIZE':      (0, 'SHREG_MIN_SIZE'),
         'ENABLE_LOGIC_OPT':        (1, 'IS_ENABLED'),
         'LOGIC_OPT_DIRECTIVE':     (1, 'ARGS.DIRECTIVE'),
         'PLACE_DIRECTIVE':         (2, 'DIRECTIVE'),
         'ENABLE_PLACE_OPT':        (3, 'IS_ENABLED'),
         'PLACE_OPT_DIRECTIVE':     (3, 'ARGS.DIRECTIVE'),
         'ROUTE_DIRECTIVE':         (4, 'DIRECTIVE'),
         'ENABLE_ROUTE_OPT':        (5, 'IS_ENABLED'),
         'ROUTE_OPT_DIRECTIVE':     (5, 'ARGS.DIRECTIVE')}
    prefix_map = {0: 'synth_1.STEPS.SYNTH_DESIGN.ARGS',
                  1: 'impl_1.STEPS.OPT_DESIGN',
                  2: 'impl_1.STEPS.PLACE_DESIGN.ARGS',
                  3: 'impl_1.STEPS.PHYS_OPT_DESIGN',
                  4: 'impl_1.STEPS.ROUTE_DESIGN.ARGS',
                  5: 'impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN'}
    args = []
    for param, (prefix_no, vpl_param) in param_map.items():
      value = cfg_data.get(param)
      if value is None:
        continue
      if param == 'RESOURCE_SHARING':
        value = 'on' if value else 'off'
      name = 'vivado_prop:run.{}.{}'.format(prefix_map[prefix_no], vpl_param)
      args.append('    --xp {}={} \\'.format(name, value))
    return '\n'.join(args)
