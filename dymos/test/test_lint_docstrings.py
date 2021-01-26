"""
Tests all of dymos' docstrings using the Numpydocs validator.
"""
import ast
import importlib
import inspect
import os
import re
import unittest

try:
    from numpydoc.docscrape import NumpyDocString
    from numpydoc import validate
except ImportError:
    NumpyDocString = None


# Directories to exclude from docstring linting.
exclude = [
    'doc',
    'test',
    'examples',
    'plots',
    '_html',
    '__pycache__',
    'coloring_files',
]

# Error Codes to Ignore
ignore = [
    'ES01',    # No extended summary found
    'EX01',    # No examples section found
    'SA01',    # See Also section not found
    'SS05',    # Summary must start with infinitive verb, not third person (e.g. use "Generate" instead of "Generates")')
]

# Build a list of dirs in which to do linting.
directories = []

top = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for root, dirs, files in os.walk(top, topdown=True):
    dirs[:] = [d for d in dirs if d not in exclude]
    for item in dirs:
        directories.append(os.path.join(root, item))


@unittest.skipUnless(NumpyDocString, "requires 'numpydoc' >= 1.1")
class DocstringLintTestCase(unittest.TestCase):

    def test_docstrings(self):
        failures = {}

        # Loop over directories
        for dirpath in sorted(directories):

            # Loop over files
            for file_name in sorted(os.listdir(dirpath)):
                if not file_name.startswith("_") and file_name[-3:] == '.py' \
                   and not os.path.isdir(file_name):

                    # To construct module name, remove part of abs path that
                    # precedes 'dymos', and then replace '/' with '.' in the remainder.
                    mod1 = re.sub(r'.*dymos', 'dymos', dirpath).replace('/', '.')

                    # Then, get rid of the '.py' to get final part of module name.
                    mod2 = file_name[:-3]

                    module_name = f'{mod1}.{mod2}'

                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError:
                        # e.g. PETSc is not installed
                        failures[module_name] = [('EEE', f'Error when loading module {module_name}.')]
                        continue

                    classes = [x for x in dir(mod)
                               if not x.startswith('_') and inspect.isclass(getattr(mod, x)) and
                               getattr(mod, x).__module__ == module_name]

                    # Loop over classes.
                    for class_name in classes:
                        full_class_path = f'{module_name}.{class_name}'
                        try:
                            result = validate.validate(full_class_path)
                        except:
                            continue

                        for error_tuple in result['errors']:
                            if error_tuple[0] not in ignore:
                                if full_class_path not in failures:
                                    failures[full_class_path] = []
                                msg = f"{error_tuple[0]}: {error_tuple[1]}"
                                failures[full_class_path].append(msg)

                        clss = getattr(mod, class_name)

                        methods = [x for x in dir(clss)
                                   if (inspect.ismethod(getattr(clss, x)) or
                                       inspect.isfunction(getattr(clss, x))) and
                                   x in clss.__dict__]

                        # Loop over class methods.
                        for method_name in methods:

                            if not method_name.startswith('_'):
                                full_method_path = f'{module_name}.{class_name}.{method_name}'
                                try:
                                    result = validate.validate(full_method_path)
                                except:
                                    continue

                                for error_tuple in result['errors']:
                                    if error_tuple[0] not in ignore:
                                        if full_method_path not in failures:
                                            failures[full_method_path] = []
                                        msg = f"{error_tuple[0]}: {error_tuple[1]}"
                                        failures[full_method_path].append(msg)

                    tree = ast.parse(inspect.getsource(mod))

                    if hasattr(tree, 'body'):
                        funcs = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
                    else:
                        funcs = []

                    # Loop over standalone functions.
                    for func_name in funcs:
                        if not func_name.startswith('_'):
                            full_function_path = f'{module_name}.{func_name}'
                            try:
                                result = validate.validate(full_function_path)
                            except:
                                continue

                            for error_tuple in result['errors']:
                                if error_tuple[0] not in ignore:
                                    if full_function_path not in failures:
                                        failures[full_function_path] = []
                                    msg = f"{error_tuple[0]}: {error_tuple[1]}"
                                    failures[full_function_path].append(msg)

        if failures:
            msg = '\n'
            count = 0
            for key in failures:
                msg += f'{key}\n'
                count += len(failures[key])
                for failure in failures[key]:
                    msg += f'    {failure}\n'
            msg += f'Found {count} issues in docstrings'
            self.fail(msg)


if __name__ == '__main__':
    unittest.main()
