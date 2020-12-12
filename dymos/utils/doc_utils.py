import inspect
import pathlib
import sys


class tee:
    """
    Mimics the behavior of the linux tee command by allowing the user to create new streams
    that output to multiple destinations.

    For instance:

    ```
    stdout_save = sys.stdout
    sys.stdout = tee(sys.stdout, f)
    print(foo)
    sys.stdout = stdout_save
    ```

    Will print foo to both stdout and the stream f.
    """

    def __init__(self, *files):
        self._files = files

    def __del__(self):
        for f in self._files:
            if f != sys.stdout and f != sys.stderr:
                try:
                    f.close()
                except:
                    pass

    def getvalue(self):
        """
        This exists only so pycharm won't have issues with the class if an exception is raised.

        Call getvalue on the first stream.
        """
        for f in self._files:
            return f.getvalue()

    def write(self, text):
        for f in self._files:
            f.write(text)

    def flush(self):
        for f in self._files:
            f.flush()


def save_for_docs(method, transparent=False):
    """
    Decorator used to save output and matplotlib figures from tests to be used in documentation.

    Parameters
    ----------
    method : callable
        The wrapped test method.

    Returns
    -------
    A wrapped version of method which runs the test method while doing the following:
    - Creates a new directory `_output` in the test file's directory.
    - Echoes all stdout and stderr to a file named '{classname}.{testname}.out' within the _output directory
    - Switches the matplotlib backend to 'Agg'
    - Saves all matplotlib figures created during the test to '{classname}.{testname}_{i}.png'
    - Restores the backend, stdout, and stderr.
    """
    def wrapped(self):
        import os
        import sys
        import matplotlib
        import matplotlib.pyplot as plt

        stdout_save = sys.stdout
        stderr_save = sys.stderr
        classname = self.__class__.__name__
        testname = method.__name__

        output_dir = pathlib.Path(inspect.getfile(method)).parent.joinpath('_output')
        output_path = output_dir.joinpath(f'{classname}.{testname}.out')

        if not pathlib.Path(output_dir).is_dir():
            os.mkdir(output_dir)

        backend_save = matplotlib.get_backend()
        plt.switch_backend('Agg')

        f = open(output_path, 'w')
        sys.stdout = tee(sys.stdout, f)
        sys.stderr = tee(sys.stdout, f)

        try:
            method(self)
        finally:
            f.close()
            sys.stdout = stdout_save
            sys.stderr = stderr_save

        for i in plt.get_fignums():
            plt.figure(i)

            if transparent:
                for ax in plt.gcf().get_axes():
                    ax.patch.set_alpha(0.0)

            plt.tight_layout()
            plt.savefig(output_dir.joinpath(f'{classname}.{testname}_{i}.png'),
                        transparent=transparent)

        plt.switch_backend(backend_save)

    wrapped._method = method
    return wrapped
