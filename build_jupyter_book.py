#!/usr/bin/env python
import argparse
import os
import pathlib

from copy_build_artifacts import copy_build_artifacts


_this_file = pathlib.Path(__file__).resolve()
REPO_ROOT = _this_file.parent
BOOK_DIR = pathlib.Path(REPO_ROOT, 'docs')


def build_book(book_dir=BOOK_DIR, clean=True):
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
        os.system(f'jupyter-book clean {book_dir}')
    # Letting the errors through until the final PR.
    os.system(f'jupyter-book build -W {book_dir} --keep-going')
    copy_build_artifacts(book_dir)
    os.chdir(save_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a JupyterBook and automatically copy over'
                                                 'necessary build artifacts')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Clean the old book out before building (default is False).')
    parser.add_argument('-b', '--book', action='store', default='docs',
                        help="The name of the book to be built (default is 'docs').")
    args = parser.parse_args()
    build_book(book_dir=args.book, clean=args.clean)


