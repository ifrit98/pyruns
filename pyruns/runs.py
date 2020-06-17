import os
import time
import numpy as np
import logging
from pprint import pformat
from contextlib import redirect_stdout


logger = logging.getLogger('pyruns')


friendly_timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))


_globals = {
    'runs_dir': None,
    'run_dir': {
        'path': None,
        'config': None,
        'flags': None,
        'flags_file': None
    },
    'pending_writes': None
}



def clear_run():
    """Clears `_globals` dictionary by setting all to `None`"""
    _globals['runs_dir'] = None
    _globals['run_dir']['path'] = None
    _globals['run_dir']['config'] = None
    _globals['run_dir']['flags'] = None
    _globals['run_dir']['flags_file'] = None
    _globals['pending_writes'] = None


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

    _globals['start_time'] = friendly_timestamp()
    logger.info('Start time: {}'.format(_globals['start_time']))

    # Copy src contents over to run_dir
    logger.info("rsyncing src directory over to run_dir: {}".format(run_dir))
    src_dir = os.getcwd() + '/'
    backup_dir = os.path.expanduser(backup_dir)
    cmd = "rsync -abvv --backup-dir {} --exclude {} --info=backup2,progress2 {} {}".format(
        backup_dir, exclude, src_dir, run_dir)
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
            from train import history

    epochs = _globals['run_dir']['flags']['epochs']
    loss = history.history['loss']

    # Write out plots
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(loss)), loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig('plots/training_loss_{}.pdf'.format(epochs))

    # Save history object
    import pickle
    hist_path = 'history/history-{}.pickle'.format(epochs)  # os.getcwd() + 
    with open(hist_path, 'wb') as f:
        pickle.dump(history.history, f)

    logger.info("History:\n{}".format(pformat(history.history)))

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



def initialize_run(run_dir=None,
                   flags=None,
                   config=None,
                   flags_file='flags.yaml',
                   file_=None):

    if _globals['runs_dir'] is None:
        _globals['runs_dir'] = os.path.expanduser('~/runs')

    if run_dir is None:
        run_dir = unique_run_dir()

    # Create dir if necessary
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # Import flags dict from yaml
    if not flags:
      import yaml
      stream = open(flags_file, 'r')
      flags = yaml.load(stream)

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
                 encoding='utf-8'):
    """Initialize and perform a training run given `file_` training script.

    Keyword arguments:
    file_ -- training python script
    flags -- flags object or dictionary if already loaded
    run_dir -- the unique run directory to place experiment metadata
    runs_dir -- high level runs directory that houses all runs
    exclude -- comma separated list of files or directories to exlucde from rsync
    """
    clear_run()

    files = os.listdir('.')
    if file_ not in files:
        raise ValueError("Train py script `{}` not found in cwd".format(file_))

    if flags_file not in files:
        raise ValueError("Flags yaml file `{}` not found in cwd".format(flags_file))

    runs_dir = runs_dir or os.path.expanduser('~/runs')
    _globals['runs_dir'] = runs_dir

    run_dir = initialize_run(run_dir=run_dir,
                             flags=flags,
                             flags_file=flags_file,
                             file_=file_)

    do_training_run(file_, run_dir)
    
    logger.info("Training run completed: {}".format(run_dir))

