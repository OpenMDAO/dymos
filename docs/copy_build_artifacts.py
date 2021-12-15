#!/usr/bin/env python
import os
import pathlib
import fnmatch
import shutil

def copy_build_artifacts(book_dir='openmdao_book'):
    """
    Copy build artifacts (html files, images, etc) to the output _build directory.
    Parameters
    ----------
    book_dir : str
        The directory containing the Jupyter-Book to be created.
    """
    PATTERNS_TO_COPY = ('*.html', '*.png')
    TARGET_DIR = '_build'
    EXCLUDE_DIRS = ('_build', '.ipynb_checkpoints')

    for dirpath, dirs, files in os.walk(book_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_path = pathlib.PurePath(dirpath).parts[1:]
        target_path = pathlib.PurePath(book_dir, TARGET_DIR, 'html', *rel_path)
        files_to_copy = set()
        for pattern in PATTERNS_TO_COPY:
            files_to_copy |= set(fnmatch.filter(files, pattern))
        for f in files_to_copy:
            src = pathlib.PurePath(dirpath, f)
            dst = pathlib.PurePath(target_path, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)


if __name__ == '__main__':
    copy_build_artifacts('openmdao_book')


