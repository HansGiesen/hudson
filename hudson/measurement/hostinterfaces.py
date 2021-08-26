"""
Interfaces for running programs on various kinds of hosts

Created on Sep 17, 2019
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

import abc
import re


class HostInterface(object):
  """Abstract interface for running a program on various hosts

  Attributes
  ----------
  interf : MeasurementInterface
    Measurement interface using this host interface
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, interf):
    self.interf = interf

  @abc.abstractmethod
  def run(self, program, run_dir, name, max_threads, max_mem):
    """Run a program on another host via the interface.

    Parameters
    ----------
    program : str
      File path to program
    run_dir : str
      Directory in which program must executed
    name : str
      Identifier for this execution
    max_threads : int
      Number of processor cores that program uses
    max_mem : int
      Maximum amount of memory in GB that program uses
    """
    pass

  @staticmethod
  def create(cfg, interf):
    """Factory method for creating host interface objects.

    Parameters
    ----------
    cfg : dict
      Tuner configuration
    interf : MeasurementInterface
      Measurement interface using this host interface
    """
    if cfg['type'] == 'local':
      return LocalHostInterface(interf)
    elif cfg['type'] == 'ssh':
      return SSHHostInterface(interf, cfg['host'])
    elif cfg['type'] == 'grid':
      return GridHostInterface(interf, cfg['queues'])


class LocalHostInterface(HostInterface):
  """Interface to run program on local host
  """

  def run(self, program, run_dir, name, max_threads, max_mem):
    """Run a program on the local host.

    Parameters
    ----------
    program : str
      File path to program
    run_dir : str
      Directory in which program must executed
    name : str
      Identifier for this execution (unused)
    max_threads : int
      Number of processor cores that program uses (unused)
    max_mem : int
      Maximum amount of memory in GB that program uses (unused)
    """
    result = self.interf.call_program(program)
    with open(run_dir + '/stdout.log', 'w') as log_file:
      log_file.write(result['stdout'])
    with open(run_dir + '/stderr.log', 'w') as log_file:
      log_file.write(result['stderr'])
    return result


class SSHHostInterface(HostInterface):
  """Interface to run program via SSH on remote host

  Attributes
  ----------
  host : str
    Host to run program on
  """

  def __init__(self, interf, host):
    super(SSHHostInterface, self).__init__(interf)
    self.host = host


  def run(self, program, run_dir, name, max_threads, max_mem):
    """Run a program via SSH on a remote host.

    Parameters
    ----------
    program : str
      File path to program
    run_dir : str
      Directory in which program must executed
    name : str
      Identifier for this execution
    max_threads : int
      Number of processor cores that program uses
    max_mem : int
      Maximum amount of memory in GB that program uses
    """
    result = self.interf.call_program('ssh ' + self.host + ' ' + program)
    with open(run_dir + '/stdout.log', 'w') as log_file:
      log_file.write(result['stdout'])
    with open(run_dir + '/stderr.log', 'w') as log_file:
      log_file.write(result['stderr'])
    return result


class GridHostInterface(HostInterface):
  """Interface to run program via grid engine on remote host

  Parameters
  ----------
  queues : list of str
    Job queues to which the grid engine should try to submit the job.
  """

  def __init__(self, interf, queues):
    super(GridHostInterface, self).__init__(interf)
    self.queues = queues


  def run(self, program, run_dir, name, max_threads, max_mem):
    """Run a program on the grid.
    
    The queues are tried in the listed order.  If no hosts are available to run
    the job immediately, we try the next listed queue, etc.  If the last queue
    is not available either, we wait until it becomes available.
    
    Parameters
    ----------
    program : str
      File path to program
    run_dir : str
      Directory in which program must executed
    name : str
      Identifier for this execution
    max_threads : int
      Number of processor cores that program uses
    max_mem : int
      Maximum amount of memory in GB that program uses
    """
    for queue in range(len(self.queues)):
      qsub_params = '-q \'' + self.queues[queue] + '\''
      if queue < len(self.queues) - 1:
        qsub_params += ' -now y'
# The parallel environment is apparently not enabled on icgrid63.  The number
# of slots on the icgrid6x machines is only half the number of cores.
# Moreover, I always get fewer slots than the icgrid4x and icgrid5x machines
# support.  Hence, it seem better not to use these and do my own allocation.
#      cmd = 'qsub -S /bin/bash' \
#            ' -wd ' + run_dir + \
#            ' -o ' + run_dir + '/stdout.log' + \
#            ' -e ' + run_dir + '/stderr.log' + \
#            ' -N ' + name + \
#            ' ' + qsub_params + \
#            ' -sync y' \
#            ' -notify' \
#            ' -pe onenode ' + str(max_threads) + \
#            ' -l mem=' + str(float(max_mem) / max_threads) + 'g' \
#            ' ' + program
      cmd = 'qsub -S /bin/bash' \
            ' -wd ' + run_dir + \
            ' -o ' + run_dir + '/stdout.log' + \
            ' -e ' + run_dir + '/stderr.log' + \
            ' -N ' + name + \
            ' ' + qsub_params + \
            ' -sync y' \
            ' -notify' \
            ' ' + program
      result = self.interf.call_program(cmd)
      pattern = r"Your qsub request could not be scheduled"
      if not re.search(pattern, result['stderr']):
        break

    with open(run_dir + '/qsub_stdout.log', 'w') as log_file:
      log_file.write(result['stdout'])
    with open(run_dir + '/qsub_stderr.log', 'w') as log_file:
      log_file.write(result['stderr'])

    if re.search(r'Interrupted!', result['stderr']):
      raise KeyboardInterrupt

    return result

