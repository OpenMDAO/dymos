import os
import shutil
import tempfile

from openmdao.utils.mpi import MPI


def _new_setup(self):
    self._startdir_ = os.getcwd()
    if MPI is None:
        self._tempdir_ = tempfile.mkdtemp(prefix='testdir-')
    elif MPI.COMM_WORLD.rank == 0:
        self._tempdir_ = tempfile.mkdtemp(prefix='testdir-')
        MPI.COMM_WORLD.bcast(self._tempdir_, root=0)
    else:
        self._tempdir_ = MPI.COMM_WORLD.bcast(None, root=0)

    os.chdir(self._tempdir_)
    if hasattr(self, 'original_setUp'):
        self.original_setUp()


def _new_teardown(self):
    if hasattr(self, 'original_tearDown'):
        self.original_tearDown()

    os.chdir(self._startdir_)

    if MPI is not None:
        # make sure everyone's out of that directory before rank 0 deletes it
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            try:
                shutil.rmtree(self._tempdir_)
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
