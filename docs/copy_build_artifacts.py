#!/usr/bin/env python
"""Copy static files into the Sphinx HTML output that Sphinx does not collect."""
import fnmatch
import os
import pathlib
import shutil
import sys


def copy_build_artifacts(executed_book_dir='_executed_book'):
    """
    Copy static files that Sphinx does not collect into the HTML output.

    Sphinx copies images referenced by Markdown syntax or Sphinx image
    directives. Files referenced only from raw HTML, such as the water-rocket
    N2 diagrams, are not collected and must be copied explicitly.

    HTML report files and PNGs are intentionally not copied here. Those were
    previously copied by this helper, but inlining them with
    ``IPython.display.HTML`` made that unnecessary and bloated the built docs.

    Parameters
    ----------
    executed_book_dir : str or path-like
        Directory containing the pre-executed notebooks and Sphinx build
        output. Files are copied from this directory into
        ``<executed_book_dir>/_build/html/``, preserving relative paths.
    """
    executed_book_dir = pathlib.Path(executed_book_dir).resolve()
    patterns_to_copy = ('*.svg',)
    build_html = executed_book_dir / '_build' / 'html'
    exclude_dirs = {'_build', '.ipynb_checkpoints'}
    n_copied = 0

    for dirpath, dirs, files in os.walk(executed_book_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        rel_path = pathlib.Path(dirpath).resolve().relative_to(executed_book_dir)
        target_path = build_html / rel_path

        files_to_copy = set()
        for pattern in patterns_to_copy:
            files_to_copy |= set(fnmatch.filter(files, pattern))

        for filename in files_to_copy:
            src = pathlib.Path(dirpath) / filename
            dst = target_path / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            n_copied += 1

    print(f'Copied {n_copied} static artifact(s) into {build_html}')
    return n_copied


if __name__ == '__main__':
    book_dir = sys.argv[1] if len(sys.argv) > 1 else '_executed_book'
    copy_build_artifacts(book_dir)
