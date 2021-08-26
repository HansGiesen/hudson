#!/usr/bin/env python3

interval = 10

import argparse
import os
import psutil
import signal
import subprocess
import sys
import time

def signal_handler(signum, frame):
  print('Caught signal', signum)
  terminate()

def terminate():
  try:
    proc = psutil.Process(subproc.pid)
  except psutil.NoSuchProcess:
    proc = None
  if proc:
    procs = [proc] + proc.children(recursive=True)
    for proc in procs:
      print('Terminating', proc.pid)
      try:
        proc.terminate()
      except psutil.NoSuchProcess:
        pass
    gone, alive = psutil.wait_procs(procs, timeout=3)
    for proc in alive:
      print('Killing', proc.pid)
      try:
        proc.kill()
      except psutil.NoSuchProcess:
        pass
  sys.exit(1)

description='Run a command an monitor it.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('cmd', help='Command')
parser.add_argument('--timeout', type=float, default=float('inf'),
                                 help='Timeout in s')
parser.add_argument('--max-mem', type=float, help='Memory limit in GB')
parser.add_argument('--log', help='Log file')
args = parser.parse_args()

subproc = None

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

start_time = time.time()
subproc = subprocess.Popen(args.cmd, shell=True)

try:
  parent = psutil.Process(subproc.pid)
except psutil.NoSuchProcess:
  parent = None

peak_usage = 0.0
time_left = args.timeout
while parent:
  wait_time = min(time_left, interval)
  try:
    subproc.wait(timeout=wait_time)
  except subprocess.TimeoutExpired:
    pass
  else:
    end_time = time.time()
    break

  time_left -= wait_time
  if time_left == 0.0:
    print('Timeout expired.', file=sys.stderr)
    terminate()

  procs = [parent]
  try:
    procs += parent.children(recursive=True)
  except psutil.NoSuchProcess:
    pass

  mem_usage = 0.0
  for proc in procs:
    try:
      mem_usage += proc.memory_info().rss
    except psutil.NoSuchProcess:
      pass
  mem_usage /= 1024.0 ** 3

  peak_usage = max(mem_usage, peak_usage)

  if args.log:
    with open(args.log, 'a') as log_file:
      log_file.write('%e\n' % mem_usage)

  if args.max_mem and mem_usage > args.max_mem:
    print('Out of memory.', file=sys.stderr)
    terminate()

print('Runtime: %e s' % (end_time - start_time))
print('Peak memory usage: %e GB' % peak_usage)

sys.exit(subproc.returncode)
