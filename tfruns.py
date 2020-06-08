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

def clear_run():
    # _globals['runs_dir'] = None
    _globals['run_dir']['path'] = None
    _globals['run_dir']['config'] = None
    _globals['run_dir']['flags'] = None
    _globals['run_dir']['flags_file'] = None
    _globals['pending_writes'] = None


def unique_run_dir(runs_dir = _globals['runs_dir'], format_="%m_%d_%y_%H:%M:%S"):
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
    pass


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
                    exclude='', 
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

    # TODO: Do training run (source train.py)
    # exec(open(file_).read())
    try:
        with open(file_) as fd:
            exec(fd.read())
    except Exception as e:
        print(e)

    # Write results, plots, etc to files

    # Record end time
    end = datetime.now()
    END_TIME = str(end).replace(' ', '_')[:-7]

    _globals['start_time_obj'] = start
    _globals['end_time_obj'] = end

    clear_run()



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
                 flags_file=None,
                 run_dir=None,
                 runs_dir=None,
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


    """{R}
    # prepare to return the run
    run_return = return_runs(run_record(run_dir))

    save_run_view(run_dir, file.path(run_dir, "tfruns.d", "view.html"))

    return run_return
    """

# Remaining R CODE to convert
"""

run_record <- function(run_dir) {

  # validate that it exists
  if (!utils::file_test("-d", run_dir))
    stop("Run directory ", run_dir, " does not exist", call. = FALSE)

  # compute run name and meta dir
  run <- basename(run_dir)
  meta_dir <- file.path(run_dir, "tfruns.d")
  props_dir <- file.path(meta_dir, "properties")
  if (!utils::file_test("-d", props_dir))
    props_dir <- NULL

  # read all properties into a list
  read_properties <- function() {
    if (!is.null(props_dir) && file.exists(props_dir)) {
      properties <- list.files(props_dir)
      values <- lapply(properties, function(file) {
        paste(readLines(file.path(props_dir, file)), collapse = "\n")
      })
      names(values) <- properties

      # default 'type' and 'context' (data migration)
      if (is.null(values$type) || identical(values$type, 'local'))
        values$type <- 'training'
      if (is.null(values$context))
        values$context <- 'local'

      # return values
      values
    } else {
      list()
    }
  }

  # type converters for properties
  as_type <- function(properties, name, converter) {
    value <- properties[[name]]
    if (is.null(value))
      NULL
    else
      converter(value)
  }
  as_numeric <- function(properties, name) {
    as_type(properties, name, as.numeric)
  }
  as_integer <- function(properties, name) {
    as_type(properties, name, as.integer)
  }
  as_logical <- function(properties, name) {
    as_type(properties, name, function(value) {
      if (value %in% c("TRUE", "true", "yes", "1"))
        value <- TRUE
      else if (value %in% c("FALSE", "false", "no", "0"))
        value <- FALSE
      as.logical(value)
    })
  }

  # function to read columns from a json file
  read_json_columns <- function(file, prefix) {
    json_path <- file.path(meta_dir, file)
    if (file.exists(json_path)) {
      columns <- jsonlite::read_json(json_path)
      if (length(columns) > 0) {
        names(columns) <- paste0(prefix, "_", names(columns))
      }
      columns
    } else {
      NULL
    }
  }

  # core columns
  columns <- list()
  columns$run_dir <- run_dir

  # read properties and do type conversions for known values
  properties <- read_properties()
  properties$start <- as_numeric(properties, "start")
  properties$end <- as_numeric(properties, "end")
  properties$samples <- as_integer(properties, "samples")
  properties$validation_samples <- as_integer(properties, "validation_samples")
  for (unit in valid_steps_units)
    properties[[unit]] <- as_integer(properties, unit)
  properties$batch_size <- as_integer(properties, "batch_size")
  properties$completed <- as_logical(properties, "completed")
  properties$learning_rate <- as_numeric(properties, "learning_rate")
  properties$cloudml_created <- as_integer(properties, "cloudml_created")
  properties$cloudml_start <- as_integer(properties, "cloudml_start")
  properties$cloudml_end <- as_integer(properties, "cloudml_end")
  properties$cloudml_ml_units <- as_numeric(properties, "cloudml_ml_units")

  # add properties to columns
  columns <- append(columns, properties)

  # evaluation
  columns <- append(columns, read_json_columns("evaluation.json", "eval"))

  # metrics
  epochs_completed <- 0L
  metrics_json_path <- file.path(meta_dir, "metrics.json")
  if (file.exists(metrics_json_path)) {
    # read metrics
    metrics <- jsonlite::read_json(metrics_json_path, simplifyVector = TRUE)
    if (length(metrics) > 0) {
      for (metric in names(metrics)) {
        if (metric == "epoch")
          next
        values <- metrics[[metric]]
        available_values <- values[!is.na(values)]
        epochs_completed <- length(available_values)
        if (epochs_completed > 0) {
          last_value <- available_values[[epochs_completed]]
          columns[[paste0("metric_", metric)]] <- last_value
        }
      }
    }
  }

  steps_completed_unit <- get_steps_completed_unit(get_steps_unit(columns))
  # epochs completed
  columns[[steps_completed_unit]] <- epochs_completed

  # flags
  columns <- append(columns, read_json_columns("flags.json", "flag"))

  # error
  error_json_path <- file.path(meta_dir, "error.json")
  if (file.exists(error_json_path)) {
    error <- jsonlite::read_json(error_json_path, simplifyVector = TRUE)
    columns[["error_message"]] <- error$message
    columns[["error_traceback"]] <- paste(error$traceback, collapse = "\n")
  }


  # add metrics and source fields
  meta_dir <- meta_dir(run_dir, create = FALSE)
  metrics_json <- file.path(meta_dir, "metrics.json")
  if (file.exists(metrics_json))
    columns$metrics <- metrics_json
  source_code <- file.path(meta_dir, "source.tar.gz")
  if (file.exists(source_code))
    columns$source_code <- source_code

  # convert to data frame for calls to rbind
  as.data.frame(columns, stringsAsFactors = FALSE)
}





return_runs <- function(runs, order = NULL) {

  # re-order columns
  select_cols <- function(cols) {
    intersect(cols, colnames(runs))
  }
  cols_with_prefix <- function(prefix) {
    cols <- colnames(runs)
    cols[grepl(paste0("^", prefix, "_"), cols)]
  }
  cols <- character()
  cols <- c(cols, cols_with_prefix("eval"))
  cols <- c(cols, cols_with_prefix("metric"))
  cols <- c(cols, cols_with_prefix("flag"))
  cols <- c(cols, select_cols(c("samples", "validation_samples")))
  cols <- c(cols, select_cols(c("batch_size")))
  for (unit in valid_steps_units)
    cols <- c(cols, select_cols(c(unit, paste0(unit, "_completed"))))
  cols <- c(cols, select_cols(c("metrics")))
  cols <- c(cols, select_cols(c("model", "loss_function", "optimizer", "learning_rate")))
  cols <- c(cols, select_cols(c("script", "source")))
  cols <- c(cols, select_cols(c("start", "end", "completed")))
  cols <- c(cols, select_cols(c("output", "error_message", "error_traceback")))
  cols <- c(cols, select_cols(c("source_code")))
  cols <- c(cols, select_cols(c("context", "type")))
  cols <- c(cols, setdiff(colnames(runs), cols))

  # promote any ordered columns to the front
  if (identical(unname(order), "start"))
    order <- NULL
  initial_cols <- c(select_cols(c("run_dir")), order)
  cols <- setdiff(cols, initial_cols)
  cols <- c(initial_cols, cols)

  # re-order cols (always have type and run_dir at the beginning)
  runs <- runs[, cols]

  # apply special class and add order attribute
  class(runs) <- c("tfruns_runs_df", class(runs))
  attr(runs, "order") <- order
  attr(runs, "original_cols") <- colnames(runs)

  # return runs
  runs
}

"""