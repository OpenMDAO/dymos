#!/usr/bin/env python
import argparse
import os
import subprocess
import pathlib


_this_file = pathlib.Path(__file__).resolve()
REPO_ROOT = _this_file.parent
BOOK_DIR = pathlib.Path(REPO_ROOT, 'dymos_book')


def build_book(book_dir=BOOK_DIR, clean=True, ignore_warnings=False):
    """
    Clean (if requested), build, and copy over necessary files for the JupyterBook to be created.
    Parameters
    ----------
    book_dir
    clean
    """
    save_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    if clean:
        subprocess.run(['jupyter-book', 'clean', book_dir])  # nosec: trusted input
    if ignore_warnings:
        subprocess.run(['jupyter-book', 'build', '--keep-going', book_dir])  # nosec: trusted input
    else:
        subprocess.run(['jupyter-book', 'build', '-W', '--keep-going', book_dir])  # nosec: trusted input
    os.chdir(save_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a JupyterBook and automatically copy over'
                                                 'necessary build artifacts')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Clean the old book out before building (default is False).')
    parser.add_argument('-b', '--book', action='store', default='dymos_book',
                        help="The name of the book to be built (default is 'dymos_book').")
    parser.add_argument('--ignore_warnings', action='store_true', help='Allow the book to be built despite warnings.')
    args = parser.parse_args()
    build_book(book_dir=args.book, clean=args.clean)


