import os
import time
import numpy as np
from datetime import datetime

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


# TODO: break out into R package file structure (tfruns/R)
# TODO: Integrate python logger to record stdout to logfile

def clear_run():
    # _globals['runs_dir'] = None
    _globals['run_dir']['path'] = None
    _globals['run_dir']['config'] = None
    _globals['run_dir']['flags'] = None
    _globals['run_dir']['flags_file'] = None
    _globals['pending_writes'] = None


def unique_run_dir(runs_dir = None, format_="%m_%d_%y_%H-%M-%S"):
    runs_dir = runs_dir or _globals['runs_dir']
    run_dir = time.strftime(format_, time.strptime(time.asctime()))
    return os.path.join(runs_dir, run_dir)



def write_run_property(name, value):
    pass


def tar(tarpath, files, compression='gzip'):
    import tarfile
    tar = tarfile.open(tarpath, "w")
    for name in files:
        tar.add(name)
    tar.close()


def write_source_archive(sources_dir, data_dir, archive):
    # normalize paths since we'll be changing the working dir
    sources_dir = os.path.abspath(sources_dir)
    data_dir = os.path.abspath(data_dir)

    # change to sources_dir
    wd = os.getcwd()
    # on exit, set to original working directory (equivalent in python?)
    os.chdir(sources_dir)

    # enumerate source files
    files = np.asarray([f if f.split('.')[-1] == 'py' else None for f in os.listdir()])
    pyfiles = np.array(list(map(lambda f: f.split('.')[-1] == 'py', os.listdir())))
    pyfiles = files[pyfiles]

    # TODO: proper use of tempfile to mirror R's tempfile() behavior
    # create temp dir for sources
    import tempfile
    tmpfile = 'tfruns-sources'
    sources_tmp_dir = os.path.join(tmpfile, 'source')
    os.mkdir(sources_tmp_dir)
    # on.exit(unlink(sources_tmp_dir), add = TRUE)

    # copy sources to the temp dir
    # TODO: maybe build rsync cmd for these instead?
    for f in files:
        dir_ = os.path.dirname(f)
        target_dir = os.path.join(sources_tmp_dir, dir_)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
    # file.copy(from = f, to = target_dir)


    # TODO: correctly create tarball
    # create the tarball
    os.chdir(os.path.join(sources_tmp_dir, ".."))
    tar(os.path.join(data_dir, archive), files = 'source')

    os.chdir(wd)



def write_metrics_json(data, path):
    raise NotImplementedError


# get the meta dir for a run dir
def meta_dir(run_dir, create=True):
  meta_dir = os.path.join(run_dir, "tfruns.d")
  if create and not os.path.exists(meta_dir):
    os.mkdir(meta_dir)
  return meta_dir


def is_run_active():
    return _globals['run_dir']['path'] is None


def run_dir():
    return _globals['run_dir']['path'] if is_run_active() else os.getcwd()



def write_run_metadata(type_, data):

  # we need to create a write_fn so that the write can be deferred
  # until after a run_dir is actually established. Create the function
  # automatically for known types, for unknown types the `data`
  # argument is the write_fn

    # helper function to write dictionary of values
    def named_list_write_fn(type_):
        def lambda_write_fn(data_dir):
            import json
            path = os.path.join(data_dir, type_ + '.json')
            with open(path, 'wb') as f:
                json.dump(data, f)
        return lambda_write_fn


    if type_ in ['flags', 'evaluation', 'error']:
        write_fn = named_list_write_fn(type_)
    elif type_ == 'properties':
        def write_fn(data_dir):
            properties_dir = os.path.join(data_dir, 'properties')
            # if file doesn't exist:
            os.mkdir(properties_dir)

            for name in data.keys():
                property_file = os.path.join(properties_dir, name)
                # write out to prop file
                # writeLines(as.character(data[[name]]), property_file)
    elif type_ == 'metrics':
        def write_fn(data_dir):
            # what datatype incoming? in R its a data.frame, pandas df or dictionary?
            write_metrics_json(data, os.path.join(data_dir, 'metrics.json'))
    elif type_ == 'source':
        def write_fn(data_dir):
            write_source_archive(data, data_dir, 'source.tar.gz')
    else:
        raise NotImplementedError

    # check for a run_dir. if we have one write the run data, otherwise
    # defer the write until we (maybe) acquire a run_dir later
    if not os.path.exists(run_dir):
        write_fn(meta_dir(run_dir)) # what is meta_dir() function?
    elif is_run_active():
        write_fn(meta_dir(run_dir()))
    else:
        _globals['run_dir']['pending_writes'][type_] = write_fn



def do_training_run(file_, 
                    run_dir, 
                    artifacts_dir, 
                    echo, 
                    encoding, 
                    exclude='.git*', 
                    meta_file = 'metadata.json',
                    backup_dir='~/backups'):

    # Copy src contents over to run_dir
    src_dir = os.getcwd() + '/'


    start = datetime.now()
    START_TIME = str(start).replace(' ', '_')[:-7]

    print("rsyncing src directory over to run_dir")
    backup_dir = os.path.expanduser(backup_dir)
    cmd = "rsync -abvv --backup-dir {} --exclude {} --info=backup2,progress2 {} {}".format(
        backup_dir, exclude, src_dir, run_dir)
    os.system(cmd)

    print("Using run directory:", run_dir)
    owd = os.getcwd()
    os.chdir(run_dir)

    # TODO: Do training run (source train.py)
    # exec(open(file_).read())
    # try:
    #     with open(file_) as fd:
    #         exec(fd.read())
    # except Exception as e:
    #     print(e)

    from train import history, EPOCHS
    # Write results, plots, etc to files

    import matplotlib.pyplot as plt
    # epochs = _globals['run_dir']['flags']['epochs']
    loss = history.history['loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig('plots/training_loss_{}'.format(EPOCHS) + '.pdf')

    # Save history object
    import pickle
    hist_file = os.getcwd() + '/history/history-{}'.format(EPOCHS)
    with open(hist_file, 'wb') as f:
        pickle.dump(dict(history.history), f)


    # Record end time
    end = datetime.now()
    END_TIME = str(end).replace(' ', '_')[:-7]

    # _globals['start_time_obj'] = start
    # _globals['end_time_obj'] = end
    _globals['start_time'] = START_TIME
    _globals['end_time'] = END_TIME

    # Save and log results
    import json
    with open(meta_file, 'w') as f:
      json.dump(_globals,  f)

    clear_run()



def initialize_run(run_dir=None, flags=None, config=None, flags_file='flags.yaml', 
                   type_="training", properties=None, context="local", file_=None):

    clear_run()

    if _globals['runs_dir'] is None:
      _globals['runs_dir'] = os.path.expanduser('~/runs')

    if run_dir is None:
        run_dir = unique_run_dir()

    # Create dir if necessary
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    if not flags:
      import yaml
      stream = open(flags_file, 'r')
      flags = yaml.load(stream)

    # Save to globals
    _globals['run_dir']['path'] = run_dir
    _globals['run_dir']['config'] = config
    _globals['run_dir']['flags'] = flags
    _globals['run_dir']['flags_file'] = flags_file

    # Write type and context
    # write_run_metadata("properties", {'type_': type_, 'context': context})

    # write properties
    # write_run_metadata("properties", properties)

    # write source files
    # write_run_metadata("source", os.path.dirname(file_))

    return run_dir



def training_run(file_='train.py', 
                 context='local', 
                 config='',
                 flags=None,
                 flags_file='flags.yaml',
                 run_dir=None,
                 runs_dir='/media/jason/freya/runs',
                 artifacts_dir=None,
                 properties=None,
                 echo=True,
                 encoding='utf-8'):

    files = os.listdir('.')
    if not file_ in files:
        raise ValueError("train.py not found in cwd")

    runs_dir = runs_dir or os.path.expanduser('~/runs')
    _globals['runs_dir'] = runs_dir

    run_dir = initialize_run(run_dir=run_dir,
                             context=context,
                             flags=flags,
                             flags_file=flags_file,
                             properties=properties,
                             config=config,
                             file_=file_)

    do_training_run(file_, run_dir, artifacts_dir, echo = echo, encoding = encoding)
    
    print("Training run completed:", run_dir)
