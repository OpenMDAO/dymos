import base64
import importlib
import inspect
import io
import re
import shutil
import tempfile
import textwrap
from pathlib import Path
import unittest


def define_env(env):
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


if __name__ == '__main__':
    # embed_test_output('dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochrone.test_brachistochrone')
    embed_test_output('dymos.examples.finite_burn_orbit_raise.doc.test_doc_finite_burn_orbit_raise.TestDocFiniteBurnOrbitRaise.test_doc_finite_burn_orbit_raise')
