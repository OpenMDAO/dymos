import os
import shutil
import tempfile


def _new_setup(self):
    self.startdir = os.getcwd()
    self.tempdir = tempfile.mkdtemp(prefix='testdir-')
    os.chdir(self.tempdir)
    if hasattr(self, 'original_setUp'):
        self.original_setUp()


def _new_teardown(self):
    if hasattr(self, 'original_tearDown'):
        self.original_tearDown()
    self.tempdir = os.getcwd()
    os.chdir(self.startdir)
    try:
        shutil.rmtree(self.tempdir)
    except OSError:
        pass


def use_tempdirs(cls):
    """
    Decorator used to run each test in a unittest.TestCase in its own directory.

    TestCase methods setUp and tearDown are replaced with _new_setup and
    _new_teardown, above.  Method _new_setup creates a temporary directory
    in which to run the test, stores it in self.tempdir, and then calls
    the original setUp method.  Method _new_teardown first runs the original
    tearDown method, and then returns to the original starting directory
    and deletes the temporary directory.
    """

    if getattr(cls, 'setUp', None):
        setattr(cls, 'original_setUp', getattr(cls, 'setUp'))
    setattr(cls, 'setUp', _new_setup)

    if getattr(cls, 'tearDown', None):
        setattr(cls, 'original_tearDown', getattr(cls, 'tearDown'))
    setattr(cls, 'tearDown', _new_teardown)

    return cls
