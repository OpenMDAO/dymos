import base64
import importlib
import inspect
import io
import re
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
import unittest

from numpydoc.docscrape import NumpyDocString, FunctionDoc, ClassDoc


def define_env(env):

    @env.macro
    def api_doc(reference, members=True):
        obj = get_object_from_reference(reference)

    @env.macro
    def inline_source(reference, include_def=True, include_docstring=True, indent_level=0):
        obj = get_object_from_reference(reference)

        source = ''.join(inspect.getsourcelines(obj)[0])

        re_declaration = re.compile(r'^(.+?):', flags=(re.DOTALL | re.MULTILINE))
        re_docstring = re.compile(r'(""".+?""")', flags=(re.DOTALL | re.MULTILINE))

        if not include_def:
            source = re_declaration.sub('', source, count=1)
        if not include_docstring:
            source = re_docstring.sub('', source, count=1)

        source = textwrap.dedent(source)
        source = source.strip()

        source = textwrap.indent(source, indent_level * '    ')

        indent = indent_level * '    '

        return f'{indent}```python\n{source}\n{indent}```'

    @env.macro
    def inline_plot(source, alt_text='', width=640, height=480):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')

        d = dict(locals(), **globals())

        exec(source, d, d)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        data = base64.b64encode(buf.getbuffer()).decode('ascii')
        return f'<img alt="{alt_text}" width="{width}" height="{height}" src="data:image/png;base64,{data}"/>'

    @env.macro
    def embed_plot_from_script(script_path, alt_text='', width=640, height=480):
        import matplotlib.pyplot as plt

        plt.switch_backend('Agg')
        d = dict(locals(), **globals())

        dir_path = get_parent_dir(env)
        path_to_script = dir_path.joinpath(script_path)

        exec(open(path_to_script).read(), d, d)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        data = base64.b64encode(buf.getbuffer()).decode('ascii')
        return f'<img alt="{alt_text}" width="{width}" height="{height}" src="data:image/png;base64,{data}"/>'

    @env.macro
    def embed_test_output(reference):
        test_case, test_method = reference.split('.')[-2:]
        testcase_obj = get_object_from_reference('.'.join(reference.split('.')[:-1]))
        test_dir = Path(inspect.getfile(testcase_obj)).parent
        output_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}.out')
        with open(output_file) as f:
            text = f.read()
        return f'```\n{text}\n```'

    @env.macro
    def embed_test_plot(reference, index=1, alt_text='', width=640, height=480):
        test_case, test_method = reference.split('.')[-2:]
        testcase_obj = get_object_from_reference('.'.join(reference.split('.')[:-1]))
        test_dir = Path(inspect.getfile(testcase_obj)).parent
        plot_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}_{index}.png')

        with open(plot_file, 'rb') as f:
            buf = io.BytesIO(f.read())

        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f'<img alt="{alt_text}" width="{width}" height="{height}" src="data:image/png;base64,{data}"/>'

    @env.macro
    def doc_env():
        "Document the environment"
        return {name:getattr(env, name) for name in dir(env) if not name.startswith('_')}

    @env.macro
    def api_doc(reference, members=True):

        module = '.'.join(reference.split('.')[:-1])
        item = reference.split('.')[-1]

        obj = getattr(get_object_from_reference(module), item)

        from numpydoc.docscrape import FunctionDoc, NumpyDocString, ClassDoc

        ss = io.StringIO()

        if inspect.isfunction(obj):
            _function_doc_markdown(obj, reference, outstream=ss)
            return ss.getvalue()
        elif inspect.isclass(obj):
            _class_doc_markdown(obj, reference, outstream=ss)

        # for key in doc.keys():
        #     print(key)

        return obj

def get_object_from_reference(reference):
    split = reference.split('.')
    right = []
    module = None
    while split:
        try:
            module = importlib.import_module('.'.join(split))
            break
        except ModuleNotFoundError:
            right.append(split.pop())
    if module:
        for entry in reversed(right):
            module = getattr(module, entry)
    return module


def get_parent_dir(env):
    page_path = Path(env.variables.page.url)
    full_path = Path(env.conf['docs_dir']).joinpath(page_path)
    dir_path = full_path.parents[0]
    return dir_path


def _function_doc_markdown(func, reference, outstream=sys.stdout):
    """
    Generate markdown documentation for the given function object.

    Parameters
    ----------
    func : function
        The function object to be documented.
    reference : str
        The dotted path to the function in the API.

    Returns
    -------
    str
        The markdown representation of the function documentation.
    """
    from numpydoc.docscrape import FunctionDoc

    indent = '    '

    doc = FunctionDoc(func)

    print(f'!!! abstract "{reference}"\n', file=outstream)

    if doc['Summary']:
        print(indent + ' '.join(doc['Summary']), file=outstream)

    if doc['Extended Summary']:
        print(indent + ' '.join(doc['Extended Summary'] + '\n'), file=outstream)

    print('', file=outstream)
    print(f"{indent}**{doc['Signature']}**\n", file=outstream)

    print(f'{indent}**Arguments:**\n', file=outstream)

    for p in doc['Parameters']:
        print(f'{indent}**{p.name}**: {" ".join(p.desc)}', file=outstream)
        print('', file=outstream)

    if doc['Raises']:
        print('{indent}**Raises:**\n', file=outstream)

        for p in doc['Raises']:
            print(f'{indent}**{p.name}**: {" ".join(p.desc)}', file=outstream)
            print('', file=outstream)

    for key in doc.keys():
        print(key, file=outstream)

def _class_doc_markdown(cls, reference, methods=None, outstream=sys.stdout):
    """

    Parameters
    ----------
    cls
    reference
    methods
    outstream

    Returns
    -------

    """
    from numpydoc.docscrape import ClassDoc, NumpyDocString

    indent = '    '

    doc = ClassDoc(cls)

    print(f'!!! abstract "{reference}"\n', file=outstream)

    if doc['Summary']:
        print(indent + ' '.join(doc['Summary']), file=outstream)

    if doc['Extended Summary']:
        print(indent + ' '.join(doc['Extended Summary'] + '\n'), file=outstream)

    print('', file=outstream)
    print(f"{indent}**{doc['Signature']}**\n", file=outstream)

    print(f'{indent}**Methods:**\n', file=outstream)

    for p in doc['Methods']:
        if p.name in methods:
            print(p)

    for key in doc.keys():
        print(key, file=outstream)

def _api_doc(reference, members=True):

    module = '.'.join(reference.split('.')[:-1])
    item = reference.split('.')[-1]

    obj = getattr(get_object_from_reference(module), item)

    from numpydoc.docscrape import FunctionDoc, NumpyDocString, ClassDoc


    if inspect.isfunction(obj):
        return _function_doc_markdown(obj, reference)
    elif inspect.isclass(obj):
        return _class_doc_markdown(obj, members)

    # for key in doc.keys():
    #     print(key)

    return obj


if __name__ == '__main__':
    reference = 'dymos.run_problem'
    module = '.'.join(reference.split('.')[:-1])
    item = reference.split('.')[-1]

    obj = getattr(get_object_from_reference(module), item)

    _function_doc_markdown(obj, reference)

    reference = 'dymos.Trajectory'
    module = '.'.join(reference.split('.')[:-1])
    item = reference.split('.')[-1]

    obj = getattr(get_object_from_reference(module), item)

    _class_doc_markdown(obj, reference, methods=['add_phase', 'link_phases'])

    obj2 = getattr(get_object_from_reference('dymos.Trajectory'), 'add_phase')

    print(obj2)



    # print(module.run_problem)

