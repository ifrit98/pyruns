from pyruns import training_run
import os

from sys import platform
if platform.__eq__('win32'):
    raise EnvironmentError(
        "\n`mv` and `rsync` commands require a unix-like operating system.\n\n\
        Get on ubuntu or WSL if you need to use pyruns. \n\n\
        Check back later for windows compatibility.")


import argparse
parser = argparse.ArgumentParser(
    description='pyruns: experiments made easy with better living through chemistry, TM®')


parser.add_argument('-e', '--exclude', type=str, #default='*git',
                    help='string of csv denoting files/dirs to ignore')
parser.add_argument('-r', '--runs_dir', type=str, default=os.path.expanduser('~/runs'),
                    help='Path to desired global runs directory')
parser.add_argument('--epochs', type=int,
                    help='Number of epochs to train') #, default=10)                    
parser.add_argument('--units', type=int,
                    help='Number of units for ANNs')#, default=128)
parser.add_argument('--rnn_units', type=int,
                    help='Number of units for RNNs') #, default=256)                    
parser.add_argument('--batch_size', type=int,
                    help='Size of each batch for training')#, default=64)                    
parser.add_argument('--min_lr', type=float,
                    help='Minimum learning rate value') #, default=1e-8)
parser.add_argument('--init_lr', type=float,
                    help='Initial learning rate value')#, default=0.01)                    
parser.add_argument('--max_lr', type=float,
                    help='Maximum learning rate value')#, default=1)                    
parser.add_argument('--steps_per_epoch', type=int,
                    help='Number of steps_per_epoch for training')#, default=None)                    
parser.add_argument('--decay_rate', type=float, #default=0.96,
                    help='Float value of decay rate')
parser.add_argument('--decay_epochs', type=int,
                    help='Number of decay epochs') #, default=5)      
parser.add_argument('--buffer_size', type=int,
                    help='Shuffle buffer size (int)') #, default=100)      
parser.add_argument('--patience', type=int,
                    help='Epoch patience for lr_plateau callback')#, default=10)      
parser.add_argument('--lr_factor', type=float, #default=2,
                    help='Factor to multiply lr reduction by each epoch/iter')      
parser.add_argument('--verbose', action='store_true',
                    help='(Bool) Set verbosity to True')    
parser.add_argument('--overwrite_flags', action='store_true',
                    help='(Bool) Overwrite flags.yaml in training directory so that\
                        command line arguments take precedence.')


config = vars(parser.parse_args())


runs_dir = config['runs_dir']
if not runs_dir:
    raise ValueError('\nMust supply a runs_dir path.\nE.g. `~/runs` or `C:/runs`\n')


if not os.path.exists(runs_dir):
    print('\nRuns_dir path supplied does not exist. Creating it now...\n')
    try:
        os.mkdir(runs_dir)
    except Exception as e:
        print(e)


print('\n\nStarting training run in top-level run directory: {}'.format(runs_dir.upper))

training_run(runs_dir=runs_dir, exclude='*git, data, ', config=config)