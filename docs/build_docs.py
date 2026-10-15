"""
Build or clean the Dymos documentation.

This is the Dymos counterpart of OpenMDAO's ``openmdao.devtools.build_docs``.

Usage:
    python docs/build_docs.py build             # full build
    python docs/build_docs.py build --no-exec   # skip notebook execution
    python docs/build_docs.py build --fast      # parallel Sphinx, no warnings-as-errors
    python docs/build_docs.py build --force     # re-execute all notebooks
    python docs/build_docs.py clean             # remove all generated files
    python docs/build_docs.py view              # serve the built docs locally

Output: docs/_executed_book/_build/html/index.html

Notebook execution details
--------------------------
The source notebooks in docs/dymos_book are executed with papermill and the executed
copies are written to docs/_executed_book. Sphinx then builds the HTML from the executed
copies without re-executing them.

MPI-dependent notebooks (those with "mpi": true in their top-level metadata) are
executed serially, one at a time, outside the parallel pool. All other notebooks are
executed in parallel using a multiprocessing pool.

Notebooks that need OpenMDAO reports (those with "reports": true in their top-level
metadata) are executed with OPENMDAO_REPORTS=1. All others run with OPENMDAO_REPORTS=0
to avoid the overhead of generating HTML reports for every notebook.

Each notebook is executed with OPENMDAO_SCRIPT_NAME set to its own path so that the
default problem name, and therefore the problem output directory, is unique to each
notebook. This prevents notebooks in the same directory from clobbering each other's
outputs when they are executed concurrently.

A notebook is skipped when its output in _executed_book/ already exists and is newer
than the source. All failures are collected and reported at the end.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from multiprocessing.pool import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / 'dymos_book'
OUT_DIR = HERE / '_executed_book'
HTML_DIR = OUT_DIR / '_build' / 'html'
ROOT_DOC = 'index.html'

EXCLUDE_PARTS = ('_build', '.ipynb_checkpoints')
KERNEL_START_ATTEMPTS = 3

try:
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TaskProgressColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    _RICH = True
except ImportError:
    _RICH = False


def _banner(msg):
    bar = '=' * 61
    print(f'\n{bar}')
    print(msg)
    print(bar, flush=True)


def _make_progress(no_rich=False):
    """
    Return a Rich Progress context manager, or None if rich is unavailable or disabled.

    Parameters
    ----------
    no_rich : bool
        If True, do not use rich even if it is available.

    Returns
    -------
    Progress or None
        The rich progress bar, or None if plain-text output should be used.
    """
    if not _RICH or no_rich:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn('[cyan]elapsed:'),
        TimeElapsedColumn(),
        TextColumn('[cyan]remaining:'),
        TimeRemainingColumn(),
    )


def _is_excluded(rel):
    """
    Return True if the given path relative to the book directory should be ignored.

    Parameters
    ----------
    rel : Path
        Path relative to the book directory.

    Returns
    -------
    bool
        True if the path lies within an excluded directory.
    """
    return any(part in EXCLUDE_PARTS for part in rel.parts)


def _load_nb(nb_path):
    """
    Return the notebook at the given path as a dict, or None on error.

    Parameters
    ----------
    nb_path : Path
        Path to the notebook.

    Returns
    -------
    dict or None
        The notebook JSON data.
    """
    try:
        return json.loads(Path(nb_path).read_text(encoding='utf-8'))
    except Exception:
        return None


def _nb_meta(nb_path):
    """
    Return the top-level metadata dict for a notebook, or {} on error.

    Parameters
    ----------
    nb_path : Path
        Path to the notebook.

    Returns
    -------
    dict
        The top-level notebook metadata.
    """
    nb = _load_nb(nb_path)
    return {} if nb is None else nb.get('metadata', {})


def _has_code_cells(nb_path):
    """
    Return True if the notebook contains at least one code cell.

    Parameters
    ----------
    nb_path : Path
        Path to the notebook.

    Returns
    -------
    bool
        True if the notebook has a code cell, or if it could not be read.
    """
    nb = _load_nb(nb_path)
    if nb is None:
        return True
    return any(c.get('cell_type') == 'code' for c in nb.get('cells', []))


def _is_up_to_date(src, out):
    """
    Return True if out exists and is newer than src.

    Parameters
    ----------
    src : Path
        Source file.
    out : Path
        Generated file.

    Returns
    -------
    bool
        True if the generated file does not need to be regenerated.
    """
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime


def _nb_env(nb_path):
    """
    Return the subprocess environment for executing a notebook.

    Parameters
    ----------
    nb_path : Path
        Path to the source notebook.

    Returns
    -------
    dict
        The environment variables for the papermill subprocess.
    """
    env = os.environ.copy()
    env['OPENMDAO_REPORTS'] = '1' if _nb_meta(nb_path).get('reports', False) else '0'
    env['OPENMDAO_SCRIPT_NAME'] = str(nb_path)
    return env


def _run_notebook(nb):
    """
    Execute a single notebook with papermill.

    Parameters
    ----------
    nb : Path
        Path to the source notebook.

    Returns
    -------
    tuple
        The tuple (rel_path, returncode, stdout, stderr).
    """
    rel = nb.relative_to(SRC_DIR)
    out = OUT_DIR / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'papermill', str(nb), str(out), '--no-progress-bar',
           '--kernel', 'python3', '--cwd', str(out.parent)]
    for _ in range(KERNEL_START_ATTEMPTS):
        result = subprocess.run(cmd, capture_output=True, text=True, env=_nb_env(nb))
        # Starting many kernels concurrently can occasionally fail due to a port collision.
        # Retry in that case, since the notebook itself never ran.
        if result.returncode == 0 or 'Kernel died before replying to kernel_info' not in result.stderr:
            break
    if result.returncode != 0:
        # papermill writes the output notebook even when execution fails. Remove it so that
        # the failed notebook is not considered up to date on the next build.
        out.unlink(missing_ok=True)
    return rel, result.returncode, result.stdout, result.stderr


def _execute_group(label, notebooks, results, no_rich):
    """
    Execute a group of notebooks, reporting progress, and return the failures.

    Parameters
    ----------
    label : str
        Label for the group used in progress output.
    notebooks : list of Path
        The notebooks in the group.
    results : iterable
        An iterable of (rel_path, returncode, stdout, stderr), one per notebook,
        which performs the execution lazily as it is iterated.
    no_rich : bool
        If True, use plain-text progress output.

    Returns
    -------
    list of Path
        Relative paths of the notebooks that failed.
    """
    failed = []
    n = len(notebooks)
    progress = _make_progress(no_rich)

    if progress is not None:
        with progress:
            task_id = progress.add_task(label, total=n)
            for rel, rc, stdout, stderr in results:
                progress.advance(task_id)
                if rc != 0:
                    progress.print(f'  [red]FAILED[/red] {rel}')
                    for stream in (stdout, stderr):
                        if stream.strip():
                            progress.print(stream)
                    failed.append(rel)
    else:
        for i, (rel, rc, stdout, stderr) in enumerate(results, 1):
            status = 'FAILED' if rc != 0 else 'ok'
            print(f'  [{i}/{n}] [{status}] {rel}', flush=True)
            if rc != 0:
                for stream in (stdout, stderr):
                    if stream.strip():
                        print(stream, file=sys.stderr)
                failed.append(rel)

    if failed:
        print(f'\n{label} summary: {len(failed)} failed, {n - len(failed)} succeeded.')
        for f in failed:
            print(f'  FAILED: {f}')
    else:
        print(f'\n{label} summary: all {n} notebooks succeeded.')

    return failed


def _execute_notebooks(workers, force, no_serial, no_mpi, no_rich):
    """
    Execute all documentation notebooks using papermill.

    Parameters
    ----------
    workers : int
        Number of parallel workers used to execute the non-MPI notebooks.
    force : bool
        If True, execute notebooks even if their executed output is up to date.
    no_serial : bool
        If True, skip execution of the non-MPI notebooks.
    no_mpi : bool
        If True, skip execution of the MPI notebooks.
    no_rich : bool
        If True, use plain-text progress output.
    """
    all_notebooks = sorted(nb for nb in SRC_DIR.rglob('*.ipynb')
                           if not _is_excluded(nb.relative_to(SRC_DIR)))

    markdown_only = [nb for nb in all_notebooks if not _has_code_cells(nb)]
    notebooks = [nb for nb in all_notebooks if _has_code_cells(nb)]

    if not force:
        stale = [nb for nb in notebooks
                 if not _is_up_to_date(nb, OUT_DIR / nb.relative_to(SRC_DIR))]
        skipped = len(notebooks) - len(stale)
        notebooks = stale
    else:
        skipped = 0

    mpi_notebooks = [] if no_mpi else [nb for nb in notebooks if _nb_meta(nb).get('mpi', False)]
    serial_notebooks = [] if no_serial else [nb for nb in notebooks
                                             if not _nb_meta(nb).get('mpi', False)]

    total = len(serial_notebooks) + len(mpi_notebooks)
    print(f'Found {len(all_notebooks)} notebooks '
          f'({len(serial_notebooks)} serial, {len(mpi_notebooks)} MPI, '
          f'{skipped} up-to-date, {len(markdown_only)} markdown-only).')

    if total == 0:
        print('All notebooks are up to date.')
        return

    failed = []

    # Serial notebooks are executed in parallel with one another.
    if serial_notebooks:
        worker_count = min(workers, len(serial_notebooks))
        print(f'\nExecuting {len(serial_notebooks)} serial notebooks '
              f'with {worker_count} workers...')
        with Pool(processes=worker_count) as pool:
            results = pool.imap_unordered(_run_notebook, serial_notebooks)
            failed += _execute_group('Serial notebooks', serial_notebooks, results, no_rich)

    # MPI notebooks are executed one at a time so each gets the necessary resources.
    if mpi_notebooks:
        print(f'\nExecuting {len(mpi_notebooks)} MPI notebooks serially...')
        results = (_run_notebook(nb) for nb in mpi_notebooks)
        failed += _execute_group('MPI notebooks', mpi_notebooks, results, no_rich)

    if failed:
        print(f'\nOverall: {len(failed)} notebook(s) failed:', file=sys.stderr)
        for f in failed:
            print(f'  {f}', file=sys.stderr)
        raise RuntimeError(f'{len(failed)} notebook(s) failed.')

    print(f'\nOverall: {total} notebook(s) executed successfully'
          f'{f", {skipped} skipped (up to date)" if skipped else ""}.')


def _sync_source_tree():
    """
    Copy the supporting files of the source book into _executed_book/.

    All non-notebook files (markdown pages, data files, scripts, static assets, conf.py)
    are copied so that they are present alongside the notebooks when papermill executes
    them. Notebooks with code cells are NOT copied here; papermill writes them directly
    to _executed_book/, which allows the timestamps to be used to skip up-to-date
    notebooks. Notebooks with no code cells are copied since papermill never touches them.

    Document sources in _executed_book/ whose counterpart no longer exists in the source
    book are removed so that Sphinx does not build stale pages.
    """
    for item in SRC_DIR.rglob('*'):
        rel = item.relative_to(SRC_DIR)
        if not item.is_file() or _is_excluded(rel):
            continue
        if item.suffix == '.ipynb' and _has_code_cells(item):
            continue
        target = OUT_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)

    for item in OUT_DIR.rglob('*'):
        rel = item.relative_to(OUT_DIR)
        if item.is_file() and item.suffix in ('.ipynb', '.md') and not _is_excluded(rel) \
                and not (SRC_DIR / rel).exists():
            print(f'Removing stale {rel}')
            item.unlink()


def cmd_view(args):
    """
    Serve the built docs on a local HTTP server and open them in the default browser.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    import functools
    import threading
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    if args.path is not None:
        # User supplied a path, resolve it to the directory containing the root document.
        html_dir = Path(args.path).resolve()
        if html_dir.is_file():
            html_dir = html_dir.parent
    else:
        html_dir = HTML_DIR

    if not (html_dir / ROOT_DOC).exists():
        if args.path is not None:
            print(f'{ROOT_DOC} not found in {html_dir}', file=sys.stderr)
        else:
            print('Docs have not been built yet. Run: python docs/build_docs.py build',
                  file=sys.stderr)
        sys.exit(1)

    url = f'http://localhost:{args.port}/{ROOT_DOC}'

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(html_dir))
    server = HTTPServer(('', args.port), handler)
    print(f'Serving docs at {url}  (Ctrl-C to stop)')
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')


def cmd_clean(args):
    """
    Remove all files generated by the build.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    if OUT_DIR.exists():
        _banner(f'Removing {OUT_DIR.relative_to(HERE)}')
        shutil.rmtree(OUT_DIR)
    print('\nDone.')


def cmd_build(args):
    """
    Run the full documentation build pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    os.environ.setdefault('PYDEVD_DISABLE_FILE_VALIDATION', '1')
    os.environ.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    os.environ.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')

    _banner('Copy source tree to _executed_book/')
    _sync_source_tree()

    if not args.no_exec:
        _banner('Execute notebooks')
        _execute_notebooks(
            workers=args.workers if args.workers is not None else os.cpu_count(),
            force=args.force,
            no_serial=args.no_serial,
            no_mpi=args.no_mpi,
            no_rich=args.no_rich,
        )
    else:
        print('Skipping notebook execution (--no-exec)')

    if args.no_serial or args.no_mpi:
        print('\nSkipping Sphinx build (partial notebook execution).')
        return

    _banner('Build HTML with Sphinx')
    sphinx_cmd = [sys.executable, '-m', 'sphinx', '-b', 'html', '--keep-going']
    sphinx_cmd += ['-j', 'auto'] if args.fast else ['-W']
    sphinx_cmd += [str(OUT_DIR), str(HTML_DIR)]
    subprocess.run(sphinx_cmd, check=True)

    print(f'\nDone. Docs available at: {(HTML_DIR / ROOT_DOC).relative_to(HERE.parent)}')


def main():
    """
    Parse arguments and dispatch to the appropriate subcommand.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='subcommand')

    build_p = sub.add_parser('build', help='Build the documentation (default).')
    build_p.add_argument('--no-exec', action='store_true',
                         help='Skip notebook execution; use existing _executed_book/.')
    build_p.add_argument('--no-serial', action='store_true',
                         help='Skip execution of serial (non-MPI) notebooks.')
    build_p.add_argument('--no-mpi', action='store_true',
                         help='Skip execution of MPI notebooks.')
    build_p.add_argument('--force', action='store_true',
                         help='Execute all notebooks, even those that are up to date.')
    build_p.add_argument('--fast', action='store_true',
                         help='Parallel Sphinx build (-j auto); skips warnings-as-errors.')
    build_p.add_argument('--workers', type=int, default=None,
                         help='Number of parallel workers (default: cpu_count).')
    build_p.add_argument('--no-rich', action='store_true',
                         help='Use plain-text progress output; recommended for CI.')

    sub.add_parser('clean', help='Remove all generated files.')

    view_p = sub.add_parser('view', help='Serve the built docs on a local HTTP server.')
    view_p.add_argument('--port', type=int, default=8000,
                        help='Port to serve on (default: 8000).')
    view_p.add_argument('--path', default=None,
                        help=f'Path to a built docs directory or its {ROOT_DOC} file. '
                             'Use this to serve a downloaded CI artifact.')

    if len(sys.argv) == 1:
        args = parser.parse_args(['build'])
    else:
        args = parser.parse_args()

    if args.subcommand == 'build':
        cmd_build(args)
    elif args.subcommand == 'clean':
        cmd_clean(args)
    elif args.subcommand == 'view':
        cmd_view(args)


if __name__ == '__main__':
    main()
