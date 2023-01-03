"""Test Dymos Linkage report GUI with using Playwright."""
import unittest
import os

os.system("playwright install")
from linkage_report_ui_test import dymos_linkage_gui_test_case  # nopep8: E402

if __name__ == "__main__":
    unittest.main()
