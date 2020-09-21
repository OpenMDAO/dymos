import os
import sys
import pep8
import unittest
from io import StringIO

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


class TestPep8(unittest.TestCase):

    def test_pep8(self):
        """ Tests that all files in this, directory, the parent directory, and test
        sub-directories are PEP8 compliant.

        Notes
        -----
        max_line_length has been set to 130 for this test.
        """
        dymos_path = os.path.split(dymos.__file__)[0]
        pyfiles = _discover_python_files(dymos_path)

        style = pep8.StyleGuide(ignore=['E201', 'E226', 'E241', 'E402'])
        style.options.max_line_length = 130

        save = sys.stdout
        sys.stdout = msg = StringIO()
        try:
            report = style.check_files(pyfiles)
        finally:
            sys.stdout = save

        if report.total_errors > 0:
            self.fail("Found pep8 errors:\n%s" % msg.getvalue())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
