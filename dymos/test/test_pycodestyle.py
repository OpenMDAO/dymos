import os
import sys
import unittest
from io import StringIO

import pycodestyle

import dymos

EXCLUDE_FILES = ['crm_data.py']


def _discover_python_files(path):
    """
    Recursively walk through the path and find all python files.

    Parameters
    ----------
    path : str
        The path to be traversed.

    Returns
    -------
    list
        All the python files contained within the given path

    """
    python_files = []
    for root, dirs, files in os.walk(path):
        python_files += [os.path.join(root, file) for file in files if file.endswith('.py')]
    return python_files


class TestPyCodeStyle(unittest.TestCase):

    def test_pycodestyle(self):
        """ Tests that all files in this, directory, the parent directory, and test
        sub-directories are PEP8 compliant.

        Notes
        -----
        max_line_length has been set to 130 for this test.
        """
        dymos_path = os.path.split(dymos.__file__)[0]
        pyfiles = _discover_python_files(dymos_path)

        style = pycodestyle.StyleGuide(ignore=['E226',  # missing whitespace around arithmetic operator
                                               'E241',  # multiple spaces after ','
                                               'W504',  # line break after binary operator
                                               'W605',  # invalid escape sequence
                                               'E722',  # do not use bare except
                                               'E741'   # ambiguous variable name
                                               ])
        style.options.max_line_length = 130

        report = style.check_files(pyfiles)

        if report.total_errors > 0:
            # the report just writes the failures to stdout which is swallowed by testflo by
            # default, so temporarily replace stdout with a StringIO so we can include the failure
            # descriptions in the failure message.
            from io import StringIO
            save = sys.stdout
            try:
                sys.stdout = buff = StringIO()
                report.get_file_results()
            finally:
                sys.stdout = save
                fails = buff.getvalue()

            self.fail(f"Found {report.total_errors} pycodestyle errors:\n{fails}")


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
