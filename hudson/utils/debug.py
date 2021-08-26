"""
Debug functionality

Created on Feb 11, 2021
@author: Hans Giesen (giesen@seas.upenn.edu)
"""

from logging import getLogger
import os
import pdb
import stat
import time


log = getLogger(__name__)


def run_debugger():
  """Start the interactive debugger.
  
  To communicate with the debugger, two named pipes called "debug_stdin" and
  "debug_stdout" are created.  To use the debugger, run the following commands:

  $ touch debug
  $ cat debug_stdout &
  $ cat > debug_stdin

  """
  log.info('Starting interactive debugger...')

  try:
    os.mkfifo('fifo_stdin')
  except OSError:
    pass

  try:
    os.mkfifo('fifo_stdout')
  except OSError:
    pass

  mypdb = pdb.Pdb(stdin=open('fifo_stdin', 'r'), stdout=open('fifo_stdout', 'w'))
  mypdb.set_trace()

  log.info('Continuing...')


def check():
  """Invoke debugger if a file called "debug" exists in the current directory.

  If this function is invoked periodically, it can be used to start an
  interactive debug session on scripts that were not started interactively.

  """
  if not os.path.isfile('debug'):
    return
  
  os.remove('debug')

  run_debugger()
