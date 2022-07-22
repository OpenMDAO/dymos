"""Test Dymos Linkage report GUI with using Playwright."""
import unittest
import os

os.system("playwright install")
from test_linkage_report_ui import dymos_linkage_gui_test_case

if __name__ == "__main__":
    unittest.main()
