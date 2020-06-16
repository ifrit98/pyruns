# pyruns


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
`Python >= 3.6.9`

pyruns is a python module inspired by rstudio's [tfruns](https://github.com/rstudio/tfruns) R package to intelligently manage training experiments.

## Installation
```{bash}
git clone https://github.com/ifrit98/pyruns.git
cd pyruns && pip install .
```

## Basic Usage
```{python}
from pyruns import training_run
training_run()
```


## API Documentation (Under Construction)
`training_run(file_=, flags=None, flags_file='flags.yaml', run_dir=None, runs_dir=None, exclude='*.git', encoding='utf-8')`

Initialize and perform a training run given `file_` training script, usually named `train.py`.  `run_dir` is created automatically unless otherwise specified.  The top-level `runs_dir` is assumed to be a user-expandable string path of the form `'~/runs'` if none is specified.

Keyword Arguments:
file_ -- training python script
flags -- flags object or dictionary if already loaded
run_dir -- the unique run directory to place experiment metadata
runs_dir -- high level runs directory that houses all runs
exclude -- comma separated string of files or directories to exlucde from rsync

`initialize_run(run_dir=None, flags=None, flags_file='flags.yaml', file_=None)`

Initialize variables and class objects for a new run.  This calls `unique_run_dir()` internally, loads the FLAGS yaml object from `flags_file` as a python dictionary, stored in `_globals['flags']`.

Keyword Arguments:
run_dir -- the unique run directory to place experiment metadata.
flags -- flags object or dictionary if already loaded.
flags_file -- filepath to yaml flags file.
runs_dir -- high level runs directory that houses all runs.
file_ -- training python script string name

`do_training_run(file_, run_dir, exclude='.git*', meta_file = 'metadata.json', logfile='stdout.log', backup_dir='~/backups')`


`run_dir()` -- Returns the current run directory, if one exists.

`is_run_active()` --  Returns a boolean signfying if a run already active in the current session.

`unique_run_dir()` -- Returns a unique, timestamped directory filepath.

`clear_run()` -- Clears the current cache in the `_globals` dict.
