from __future__ import print_function, division, absolute_import

import os
import pep8
import unittest

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
        max_line_length has been set to 100 for this test.
        """
        dymos_path = os.path.split(dymos.__file__)[0]
        pyfiles = _discover_python_files(dymos_path)

        style = pep8.StyleGuide(ignore=['E201', 'E226', 'E241', 'E402'])
        style.options.max_line_length = 100

        report = style.check_files(pyfiles)

        self.assertEqual(report.total_errors, 0, msg='Found pep8 errors')


if __name__ == "__main__":
    unittest.main()
