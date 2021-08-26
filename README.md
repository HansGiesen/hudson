HuDSoN
======

Introduction
------------

This repository contains the source code of HuDSoN autotuner, together with
scripts to run and process the experiments of the HuDSoN paper submitted to
FPT'21.

Directory organization
----------------------

Every benchmark has a configuration file written in YAML.  For benchmark
`<benchmark>`, the configuration file is at
`rosetta/<benchmark>/<benchmark>.yml`.

Experiments produce results in subdirectories of the `tests` directory.  Tuning
runs (or subtests) have labels.  Moreover, tuning runs are grouped into groups
(or tests), which also have labels.  Hence, the results of subtest `<subtest>`,
which is part of test `<test>` are in `tests/<test>/<subtest>`.  The tuner log
is located in `tests/<test>/<subtest>/output/hudson.log`.  Every tuning
run consists of iterations (which may run concurrently).  Every iteration, one
or more build stages of a configuration are completed.  We currently use 3
stages: presynthesis (called HLS in the paper), synthesis (logic synthesis),
and implementation (place and route).  The input and output of stage `<stage>`
during iteration `<iter>` is in `tests/<test>/<subtest>/output/<iter>/<stage>`.
In this directory, the files `stdout.log` and `stderr.log` contain the messages
output to the standard output and standard error by the build stage.

Software requirements
---------------------

HuDSoN requires the following software:
- Linux-based operating system.  We used openSUSE Leap 15.2.
- Python 2.  Version 2.7.18 worked for us.
- Python 3.  Version 3.6.12 worked for us.
- GCC.  Version 7.5.0 worked for us. 
- MySQL database.  MySQL Server 5.5.52 worked for us.

Installation
------------

To install HuDSoN follow these steps:
1)  Set the `SDSOC_ROOT` environment variable to the root directory of the
    SDSoC installation, i.e. the directory with `settings64.sh`:
        ```
        export SDSOC_ROOT=<SDSoC path>
        ```
2)  Set up a MySQL database for the tuner results.
3)  Run `setup.bash` in the root of this repository.
4)  Load the generated Python 3 virtual environment:
        ```
        source python3_env/bin/activate
        ```
5)  Follow the instructions in `platforms/README.md` to install the platforms on
    your boards.
6)  Adapt the configuration in `cfg.yml` to match your setup.  Expressions in
    angle brackets must be replaced with suitable values.  Furthermore, platform
    instances should be added or removed to match your platform(s).
7)  Start the platform manager by running the following command in the `scripts`
    directory:
        ```
        ./platform_manager.py --log platform_manager.log &
        ```
8)  Compile the libraries needed for the BNN:
        ```
        rosetta/bnn/setup.bash
        ```
9)  Install the datasets on the boards by running the following command in the
    root the repository:
        ```
        ./install_datasets.py
        ```

Running experiments
-------------------

To run one or more experiment, follow these steps:
1)  Select the tuner experiments to run by altering the `tests` variable in
    the `tests/run.py` script.  Running all experiments is rather
    time-consuming unless you have many processor cores, so we recommend
    starting with a subset of the experiments.  The currently enabled
    experiments are sufficient to produce Figure 8.  The complete set of
    experiments used for the paper has been commented out.
2)  Change the parallelism variable in `tests/run.py` to select the number of
    tuning runs to perform in parallel.  The selected experiments require 9
    processor cores for a tuning run.  We could run up to 12 tuning runs on the
    BNN in parallel on a machine with 188 GB of memory, so a tuning run requires
    roughly 16 GB.  However, the memory consumption is not constant.  With large
    numbers of tuning runs (such as 12), the worst-case memory consumption that
    you are likely to encounter is close to the average memory consumption.
    For small numbers, this is not the case.  For instance, on a machine with
    47 GB, we already faced out-of-memory errors with more than 1 tuning run.
3)  Start the experiments by running `tests/run.py`.
4)  Keep an eye on the tuner logs to ensure that the number of builds that time
    out or run out of memory is acceptable.  To locate timeouts, search for
    `TIMEOUT` in the tuner log.  For out-of-memory errors, search for `OOM`.
    The timeouts and memory limits can be adjusted in the benchmark
    configuration files.
5)  Generate the graphs and tables from the experimental results.  All scripts
    for generating graphs and tables are in the `analysis/graphs` and
    `analysis/tables` directories respectively.  They are named in accordance
    with the figure and table numbers in the paper.  Graphs are output to
    PDF-files with the same names as the generating scripts.  Tables are output
    to TEX-files that were originally included into a tabular environment of
    the paper TEX-file.  Some scripts, especially those in `analysis/callouts`
    generate TEX-files with assignments to variables.  These TEX-files were
    originally loaded into the paper TEX-file, which uses the variables to
    fill out numbers in the text, such as speedups that were measured.  These
    same values are also printed when the scripts generating them are executed.

Questions and feedback
----------------------

Should you encounter problems while building, installing, or using the
platform, or should you have feedback, feel free to contact the authors at
giesen@seas.upenn.edu.  We are happy to hear from you!

