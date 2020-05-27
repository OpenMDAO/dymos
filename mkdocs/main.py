import importlib
import inspect
import re
import shutil
import tempfile
import textwrap
from pathlib import Path


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
    def inline_plot(source, figname='figure'):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        filename = str(Path(tempfile.gettempdir()).joinpath(f'{figname}.png'))
        d = dict(locals(), **globals())
        print(source)
        exec(source, d, d)
        plt.savefig(filename)
        return f'```{filename}```'

    @env.macro
    def embed_plot_from_script(script_path, figname='figure'):
        import matplotlib.pyplot as plt

        plt.switch_backend('Agg')
        filename = str(Path(tempfile.gettempdir()).joinpath(f'{figname}.png'))
        d = dict(locals(), **globals())

        dir_path = get_parent_dir(env)
        path_to_script = dir_path.joinpath(script_path)

        exec(open(path_to_script).read(), d, d)

        output_path = dir_path.joinpath(f'figures/{figname}.png')
        plt.savefig(str(output_path))
        return f'![Screenshot](figures/{figname}.png)'

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
    project_path = Path(env.project_dir)
    page_path = Path(env.variables.page.url)

    full_path = project_path.joinpath('docs').joinpath(page_path)
    dir_path = full_path.parents[0]
    return dir_path


if __name__ == '__main__':
    figname = 'figure'
    filename = str(Path(tempfile.gettempdir()).joinpath(f'{figname}.png'))
    print(filename)