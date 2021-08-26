#!/usr/bin/env python2

import os
import re
import socket
import sys
import yaml

def get_cpu_type():
  try:
    return re.search(r"model name\s*:\s*([^\n]*)",
                     open("/proc/cpuinfo").read()).group(1)
  except:
    pass
  try:
    # for OS X
    import subprocess

    # The close_fds argument makes sure that file descriptors that should be
    # closed do not remain open because they are copied to the subprocess.
    subproc = subprocess.Popen(["sysctl", "-n", "machdep.cpu.brand_string"],
                               stdout=subprocess.PIPE, close_fds=True)
    return subproc.communicate()[0].strip()
  except:
    sys.stderr.write("failed to get cpu type\n")
  return None

def get_cpu_count():
  try:
    return int(os.sysconf("SC_NPROCESSORS_ONLN"))
  except:
    pass
  try:
    return int(os.sysconf("_SC_NPROCESSORS_ONLN"))
  except:
    pass
  try:
    return int(os.environ["NUMBER_OF_PROCESSORS"])
  except:
    pass
  try:
    return int(os.environ["NUM_PROCESSORS"])
  except:
    sys.stderr.write("failed to get the number of processors\n")
  return None

def get_memory_size():
  try:
    return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
  except:
    pass
  try:
    return int(os.sysconf("_SC_PHYS_PAGES") * os.sysconf("_SC_PAGE_SIZE"))
  except:
    pass
  try:
    # for OS X
    import subprocess

    # The close_fds argument makes sure that file descriptors that should be
    # closed do not remain open because they are copied to the subprocess.
    subproc = subprocess.Popen(["sysctl", "-n", "hw.memsize"],
                               stdout=subprocess.PIPE, close_fds=True)
    return int(subproc.communicate()[0].strip())
  except:
    sys.stderr.write("failed to get total memory\n")
  return None

cfg = {}
cfg['name'] = socket.gethostname().split('.')[0]
cfg['cpu'] = get_cpu_type()
cfg['cores'] = get_cpu_count()
cfg['memory_gb'] = get_memory_size() / 1024.0 ** 3

print(yaml.dump(cfg))
