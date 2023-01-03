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

        style = pycodestyle.StyleGuide(ignore=['E201', 'E225', 'E226', 'E241', 'E275', 'E402', 'W504', 'W605', 'E722',
                                               'E741'])
        style.options.max_line_length = 130

        save = sys.stdout
        sys.stdout = msg = StringIO()
        try:
            report = style.check_files(pyfiles)
        finally:
            sys.stdout = save

        if report.total_errors > 0:
            self.fail(f"Found {report.total_errors} pycodestyle errors:\n{msg.getvalue()}")


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
