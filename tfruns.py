import os
import time

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
    # _globals['runs_dir'] = None
    _globals['run_dir']['path'] = None
    _globals['run_dir']['config'] = None
    _globals['run_dir']['flags'] = None
    _globals['run_dir']['flags_file'] = None
    _globals['pending_writes'] = None


def unique_run_dir(runs_dir = _globals['runs_dir'], format="%m_%d_%y_%H:%M:%S"):
    run_dir = time.strftime(format, time.strptime(time.asctime()))
    return os.path.join(runs_dir, run_dir)


def write_run_metadata(type_, data):
    pass



def do_training_run(file_, run_dir, artifacts_dir, echo, encoding, backup_dir='~/backups'):
    # TODO:
    # filter out files within the run_dir we don't want

    # Copy src contents over to run_dir
    src_dir = os.getcwd() + '/'

    backup_dir = os.path.expanduser(backup_dir)
    cmd = "rsync -abvv --backup-dir {} --info=backup2,progress2 {} {}".format(backup_dir, src_dir, run_dir)
    os.system(cmd)

    # TODO: Do training run (source train.py)



def initialize_run(run_dir=None, flags=None, config=None, flags_file=None, 
                   type_="training", properties=None, context="local", file_=None):

    clear_run()

    if run_dir is None:
        run_dir = unique_run_dir()

    # Create dir if necessary
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # Save to globals
    _globals['run_dir']['path'] = run_dir
    _globals['run_dir']['config'] = config
    _globals['run_dir']['flags'] = flags
    _globals['run_dir']['flags_file'] = flags_file

    # TODO: deal with FLAGS appropriately and save (write_run_metadata())
    # if flags is a YAML file, then read from the file

    # Write type and context
    write_run_metadata("properties", {'type_': type_, 'context': context})

    # write properties
    write_run_metadata("properties", properties)

    # write source files
    write_run_metadata("source", os.path.dirname(file_))

    return run_dir



def training_run(file_='train.py', 
                 context='local', 
                 config='',
                 flags=None,
                 flags_file=None,
                 run_dir=None,
                 runs_dir=None,
                 artifacts_dir=None,
                 properties=None,
                 echo=True,
                 encoding='utf-8'):

    files = os.listdir('./')
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


# Remaining R CODE to convert
"""
  # check for forced view
  force_view <- isTRUE(view)

  # result "auto" if necessary
  if (identical(view, "auto"))
    view <- interactive()

  # print completed message
  message('\nRun completed: ', run_dir, '\n')

  # prepare to return the run
  run_return <- return_runs(run_record(run_dir))

  # force_view means we do the view (i.e. we don't rely on printing)
  if (force_view) {

    view_run(run_dir)
    invisible(run_return)

  # regular view means give it a class that will result in a view
  # when executed as a top-level statement
  } else if (isTRUE(view)) {

    class(run_return) <- c("tfruns_viewed_run", class(run_return))
    run_return

  # save a copy of the run view
  } else if (identical(view, "save")) {

    save_run_view(run_dir, file.path(run_dir, "tfruns.d", "view.html"))
    invisible(run_return)

  # otherwise just return invisibly
  } else {

    invisible(run_return)

  }
}
"""