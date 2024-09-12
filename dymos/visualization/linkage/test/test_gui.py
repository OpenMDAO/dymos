"""Test Dymos Linkage report GUI with using Playwright."""
import unittest
import os
try:
    import playwright
except ImportError:
    playwright = None


if playwright is not None:
    os.system("playwright install")
    from dymos.visualization.linkage.test.linkage_report_ui_test import dymos_linkage_gui_test_case  # noqa: E402, F401


if __name__ == "__main__":
    unittest.main()
