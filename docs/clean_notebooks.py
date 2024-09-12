#!/usr/bin/env python

import argparse
import json
import os
import pathlib
import subprocess


EXCLUDE_DIRS = ('_build', '.ipynb_checkpoints')


class bcolors:
    FILE = '\033[96m'  # blue
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _clear_notebook_outputs(path=os.getcwd(), dry_run=True):

    # Find all the notebooks
    notebooks = []
    if pathlib.Path(path).is_dir():
        for dirpath, dirs, files in os.walk(path, topdown=True):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            notebooks.extend(pathlib.Path(dirpath).glob('*.ipynb'))
        if not notebooks:
            print(f'{bcolors.FAIL}No notebooks found.{bcolors.ENDC}')
            return
    elif path.endswith('.ipynb'):
        notebooks.append(path)
    else:
        print(f'{bcolors.FILE}{path}{bcolors.ENDC}{bcolors.FAIL} is not a notebook file.{bcolors.ENDC}')
        return

    num_cleaned = 0
    for file in notebooks:
        with open(file) as f:
            json_data = json.load(f)
            for i in json_data['cells']:
                if 'execution_count' in i and i['execution_count'] is not None:
                    break
            else:
                # Notebook clean, process next notebook.
                continue
        num_cleaned += 1
        if dry_run:
            print(f'Would clear outputs from {bcolors.FILE}{file}{bcolors.ENDC}. '
                  f'{bcolors.WARNING}(dryrun = True){bcolors.ENDC}')
        else:
            print(f'Clearing {bcolors.FILE}{file}{bcolors.ENDC}...', end='')
            subprocess.Popen(f"jupyter nbconvert --clear-output --inplace {file}",
                             shell=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT).wait()
            print('done')

    if num_cleaned == 0:
        print('No unclean notebooks found.')

    return num_cleaned



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clear notebook outputs in place.')
    parser.add_argument('path', nargs='?', default='.', help='The or directory path to clean.')
    parser.add_argument('-d', '--dryrun', action='store_true',
                        help='Print notebooks with outputs but do not clean them.')
    args = parser.parse_args()
    num_cleaned = _clear_notebook_outputs(path=args.path, dry_run=args.dryrun)
    if num_cleaned >0 and  args.dryrun:
        exit(1)
