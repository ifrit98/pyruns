import os
import time
import yaml
import numpy as np
import logging
from pprint import pformat, pprint
from contextlib import redirect_stdout

import threading


logger = logging.getLogger('pyruns')


friendly_timestamp = lambda: time.strftime(
    "%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))


HPARAM_DEFAULTS = {
    "epochs": 10,
    "steps_per_epoch": 100,
    "units": 128,
    "batch_size": 32,
    "init_learning_rate": 1e-3,
    "max_learning_rate": 1,
    "min_learning_rate": 1e-6,
    "buffer_size": 100,
    "verbose": False
}

_globals = {
    'runs_dir': None,
    'run_dir': {
        'path': None,
        'config': None,
        'flags': None,
        'flags_file': None
    }
}

HISTORY = None


def clear_run():
    """
    Clears `_globals` dictionary by setting all to `None`
    """
    _globals['runs_dir'] = None
    _globals['run_dir']['path'] = None
    _globals['run_dir']['config'] = None
    _globals['run_dir']['flags'] = None
    _globals['run_dir']['flags_file'] = None


def unique_run_dir(runs_dir = None, format_="%m_%d_%y_%H-%M-%S"):
    """Returns a unique run directory filepath to house an experiment"""
    runs_dir = runs_dir or _globals['runs_dir']
    run_dir = time.strftime(format_, time.strptime(time.asctime()))
    return os.path.join(runs_dir, run_dir)


# TODO: Add tar functionality to `do_training_run()`
def tar(tarpath, files, compression='gzip'):
    import tarfile
    tar = tarfile.open(tarpath, "w")
    for name in files:
        tar.add(name)
    tar.close()


def is_run_active():
    return _globals['run_dir']['path'] is None


def run_dir():
    return _globals['run_dir']['path'] if is_run_active() else os.getcwd()


def do_training_run(file_,
                    run_dir,
                    exclude='.git*', # must be a str of csv (comma-separated-values)
                    meta_file = 'metadata.json',
                    logfile='stdout.log',
                    backup_dir='~/backups'):
    """
    Perform the training run with current run_dir. `rsync`s files in cwd minus those
    passed in to `exclude` to unique run directory, sets cwd to run_dir, and then 
    sources `file_` to execute training.  Logs created by redirecting stdout to `logfile`.

    Clears cache and returns to original working directory before returning.

    Args: 
        file_: String filepath to training script. Usually `train.py`
        run_dir: String path to current run directory.
        exclude: String of comma separated (regex) values to exclude from rsync.
        meta_file: json filepath to metadata output file for dumping `globals`. 
          Defaults to `metadata.json`
        logfile: String filepath to desired logfile. Defaults to stdout.log
        backup_dir: expandable filepath to desired backup directory for `rsync` cmd. 
          Defaults to `backup_dir`.
    
    Returns:
        None
    """

    _globals['start_time'] = friendly_timestamp()
    logger.info('Start time: {}'.format(_globals['start_time']))

    # Copy src contents over to run_dir
    logger.info("rsyncing src directory over to run_dir: {}".format(run_dir))
    src_dir = os.getcwd() + '/'
    backup_dir = os.path.expanduser(backup_dir)
    cmd = "rsync -abvv --backup-dir {} --exclude-from={} --info=backup2,progress2 {} {}".format(
        backup_dir, exclude, src_dir, run_dir)
    logger.info("RSYNC CMD: {}".format(cmd))
    os.system(cmd)

    logger.info("Using run directory: {}".format(run_dir))
    owd = os.getcwd()
    os.chdir(run_dir)
    
    # Rename _file to 'train.py' so we can use import statments
    os.system('mv {} train.py'.format(file_))


    logger.info("Executing train.py...")
    with open(logfile, 'w') as f:
        with redirect_stdout(f):
            logger.info('Now logging train.py stdout to {}'.format(logfile))
            # Executes training script and sources history object into namespace
            global HISTORY
            src = open('train.py').read()
            exec(src)
            try:
                HISTORY = history.history
            except:
                HISTORY = {'loss': [0]}
                logger.error("ERROR: importing history object from training script.\
                         \nMetrics will not be saved by pyruns.\nPlease manually " +
                            "save your history object in the train script itself.")

            loss = HISTORY['loss']

    try:
        epochs = _globals['run_dir']['flags']['epochs']
    except:
        logger.warning("No epochs parameter found in `flags.yaml`.")
        epochs = 0

    # Write out plots
    import matplotlib.pyplot as plt

    if not os.path.exists('plots'):
        os.mkdir('plots')

    plt.figure() 
    plt.plot(range(len(loss)), loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig('plots/training_loss_{}.pdf'.format(epochs))

    # Save history object
    import pickle

    if not os.path.exists('history'):
        os.mkdir('history')

    hist_path = 'history/history-{}.pickle'.format(epochs)  # os.getcwd() + 
    with open(hist_path, 'wb') as f:
        pickle.dump(HISTORY, f)

    logger.info("History:\n{}".format(pformat(HISTORY)))

    # Record end time
    _globals['end_time'] = friendly_timestamp()
    logger.info('End time: {}'.format(_globals['end_time']))

    # Save and log results
    import json
    with open(meta_file, 'w') as f:
      json.dump(_globals,  f)


    logger.info('_globals\n: {}'.format(pformat(_globals)))

    clear_run()
    os.chdir(owd)



def handle_flags_update(config, flags, flags_file):
    og_flags = flags.copy()

    logger.info("Updating flags.yaml with cmd line config...")
    flags.update(config)

    logger.info("flags after update:".format(flags))
    fs = yaml.dump(flags)

    # Move old yaml
    logger.info("Moved old {} to flags_og.yaml".format(flags_file))
    og = open('flags_og.yaml', 'w')
    og.writelines(yaml.dump(og_flags))
    og.close()

    s = open(flags_file, 'w')
    s.writelines(fs)
    s.close()
    logger.info("Written new flags to {}".format(flags_file))

    _globals['run_dir']['config'] = flags
    del og_flags



def initialize_run(run_dir=None,
                   flags=None,
                   config=None,
                   flags_file='flags.yaml',
                   file_=None):
    """
    Initializes training run variables.

    Args:
        run_dir: String path to current run directory.
        flags: flags object, as a python dictionary from yaml file.
        config: configuration object (dict) to override flags.yaml

    """

    if _globals['runs_dir'] is None:
        _globals['runs_dir'] = os.path.expanduser('~/runs')

    if run_dir is None:
        run_dir = unique_run_dir()

    # Create dir if necessary
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # Import flags dict from yaml
    if not flags:
      stream = open(flags_file, 'r')
      flags = yaml.load(stream, Loader=yaml.SafeLoader)

    if config:
        if config['overwrite_flags'] is True:
            handle_flags_update(config, flags, flags_file)


    logger.info("FLAGS:\n{}".format(pformat(flags)))

    # Save to globals
    _globals['run_dir']['path'] = run_dir
    _globals['run_dir']['config'] = config
    _globals['run_dir']['flags'] = flags
    _globals['run_dir']['flags_file'] = flags_file

    # Initialize logger
    hdlr = logging.FileHandler(os.path.join(run_dir, 'pyrun.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    logger.info("Initialized run_dir {}".format(run_dir))
    logger.info("Globals:\n{}".format(pformat(_globals)))

    return run_dir



def training_run(file_='train.py',
                 flags=None,
                 flags_file='flags.yaml',
                 run_dir=None,
                 runs_dir=None,
                 exclude='',
                 config=None,
                 use_gitignore=True,
                 encoding='utf-8'):
    """Initialize and perform a training run given `file_` training script.
    Args:
        file_ -- training python script
        flags -- flags object or dictionary if already loaded
        flags_file -- path to flags yaml file. Defaults to `flags.yaml`.
        run_dir -- the unique run directory to place experiment metadata
        runs_dir -- high level runs directory that houses all runs. Defaults to `~/runs`
        exclude -- comma separated list of files or directories to exlucde from rsync
    """
    clear_run()

    files = os.listdir('.')
    if file_ not in files:
        raise ValueError("Train py script `{}` not found in cwd".format(file_))

    if flags_file not in files:
        logger.info("\n\nFlags yaml file `{}` not found in cwd\n".format(flags_file))
        logger.info("\nWriting common defaults appended with any cmd line args...\n")
        handle_flags_update(HPARAM_DEFAULTS, {}, flags_file)
        
    runs_dir = runs_dir or os.path.expanduser('~/runs')
    _globals['runs_dir'] = runs_dir

    if exclude is '' and use_gitignore:
        exclude = "'.gitignore'"
        logger.info("New exclude: {}".format(exclude))

    run_dir = initialize_run(run_dir=run_dir,
                             flags=flags,
                             flags_file=flags_file,
                             file_=file_,
                             config=config)

    do_training_run(file_, run_dir, exclude=exclude)
    
    logger.info("Training run completed: {}".format(run_dir))


def parse_gitignore():
    if not os.path.exists('.gitignore'):
        raise ValueError(".gitignore not found in cwd")
    with open('.gitignore', 'r') as f:
        x = f.readlines()
    ignore = list(map(lambda s: s.split('\n')[:-1], x))
    ignore[-1] = [x[-1]]
    return "{'" + "', '".join(unlist(ignore)) + "'}"

def unlist(x):
    if len(x[0]) > 1:
        raise ValueError("inner dim of `x` must == 1")
    return list(map(lambda l: l[0], x))