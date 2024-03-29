# pyruns


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
`Tested with Python >= 3.6.8`

pyruns is a python module inspired by rstudio's [tfruns](https://github.com/rstudio/tfruns) R package to intelligently manage training experiments.

TL;DR
This module will create a unique run directory for each new experiment, `rsync` over 
any necessary source files, launch the run from the `run_dir`, and saves all related metrics as `metrics.json`, as well as learning curve plots in `plots/training_run_x.pdf`.

## Installation
```{bash}
git clone https://git.brsc.local/stgeorge/pyruns.git
cd pyruns && pip install .
```

## Basic Usage
```{python}
from pyruns import training_run
training_run()
```


## Usage
The code below can be thrown into a python script that can be placed in your
project's base directory and called from the command line to execute your run.
```{python}
from sys import argv
from pyruns import training_run
import os

if len(argv) == 1:
    raise ValueError('Must supply a runs_dir path.')

runs_dir = argv[1]
if not os.path.exists(runs_dir):
    print('Runs_dir path supplied does not exist. Creating it now...')
    try:
        os.mkdir(argv[1])
    except Exception as e:
        print(e)

print('Starting training run in top-level run directory: {}'.format(runs_dir))

training_run(runs_dir=runs_dir, exclude='*git')
```

NOTE:
Upon importing, pyruns checks for a file named `launch_training_run.py` in your cwd that would contain something like the text above.

If this is not found, it will ask if you would like to create the script automatically
in your proejct dir:

```{python}
import pyruns

# `launch_training_run.py` script not found in cwd.
#
# Would you like to create it now? [y|n]:
#
# y # This will place the script in your cwd.
```



## API Documentation (Under Construction)
`training_run(file_='train.py', flags=None, flags_file='flags.yaml', run_dir=None, runs_dir=None, exclude='*.git', encoding='utf-8')`

Initialize and perform a training run given `file_` training script, usually named `train.py`.  `run_dir` is created automatically unless otherwise specified.  The top-level `runs_dir` is assumed to be a user-expandable string path of the form `'~/runs'` if none is specified.

Keyword Arguments:
```
file_ -- training python script
flags -- flags object or dictionary if already loaded
run_dir -- the unique run directory to place experiment metadata
runs_dir -- high level runs directory that houses all runs
exclude -- comma separated string of files or directories to exlucde from rsync
```

`initialize_run(run_dir=None, flags=None, flags_file='flags.yaml', file_=None)`

Initialize variables and class objects for a new run.  This calls `unique_run_dir()` internally, loads the FLAGS yaml object from `flags_file` as a python dictionary, stored in `_globals['flags']`.

Keyword Arguments:
```
run_dir -- the unique run directory to place experiment metadata.
flags -- flags object or dictionary if already loaded.
flags_file -- filepath to yaml flags file.
runs_dir -- high level runs directory that houses all runs.
file_ -- training python script string name
```

`do_training_run(file_, run_dir, exclude='.git*', meta_file = 'metadata.json', logfile='stdout.log', backup_dir='~/backups')` -- Does the legwork of copying to new run_dir
and starts run from there, logging along the way.


`run_dir()` -- Returns the current run directory, if one exists.

`is_run_active()` --  Returns a boolean signfying if a run already active in the current session.

`unique_run_dir()` -- Returns a unique, timestamped directory filepath.

`clear_run()` -- Clears the current cache in the `_globals` dict.
