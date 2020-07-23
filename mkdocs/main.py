import base64
import importlib
import inspect
import io
import re
import sys
import textwrap
from pathlib import Path

try:
    from numpydoc.docscrape import FunctionDoc, ClassDoc
except ImportError as e:
    print('Please install the numpydoc package to build the documentation for Dymos')
    raise e


def define_env(env):

    @env.macro
    def api_doc(reference, members=True):
        obj = get_object_from_reference(reference)

    @env.macro
    def inline_source(reference, include_def=True, include_docstring=True, indent_level=0, show_line_numbers=True, highlight_lines=None):
        obj = get_object_from_reference(reference)

        obj = inspect.unwrap(obj)

        source = ''.join(inspect.getsourcelines(obj)[0])

        re_declaration = re.compile(r'^(.+?):', flags=(re.DOTALL | re.MULTILINE))
        re_docstring = re.compile(r'(""".+?""")', flags=(re.DOTALL | re.MULTILINE))

        if not include_def:
            source = re_declaration.sub('', source, count=1)
        if not include_docstring:
            source = re_docstring.sub('', source, count=1)

        source = textwrap.dedent(source)
        source = source.strip()

        indent = indent_level * '    '

        source = textwrap.indent(source, indent)

        line_numbers = ' linenums="1"' if show_line_numbers else ''

        if highlight_lines is not None:
            hl_lines = ' hl_lines="' + ' '.join([str(i) for i in highlight_lines]) + '"'
        else:
            hl_lines = ''

        result = f'```python{line_numbers}{hl_lines}\n{source}\n{indent}```'

        return textwrap.indent(result, indent)

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
    def embed_test(reference, script_name='script', plot_alt_text='', plots=(1,), plot_size=(640, 480)):

        # First tab for the source
        src = textwrap.indent(_get_test_source(reference), '    ')
        ss = io.StringIO()
        print(f'=== "{script_name}"', file=ss)
        print('    ```python3', file=ss)
        print(src, file=ss)
        print('    ```', file=ss)

        # Second tab for the output
        test_case, test_method = reference.split('.')[-2:]
        testcase_obj = get_object_from_reference('.'.join(reference.split('.')[:-1]))
        test_dir = Path(inspect.getfile(testcase_obj)).parent
        output_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}.out')
        with open(output_file) as f:
            text = f.read()

        print(f'=== "output"', file=ss)
        print('    ```', file=ss)
        print(textwrap.indent(text, '    '), file=ss)
        print('    ```', file=ss)

        # Third tab for the plot
        index = plots[0]
        plot_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}_{index}.png')

        with open(plot_file, 'rb') as f:
            buf = io.BytesIO(f.read())

        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        width, height = plot_size
        print(f'=== "plot"\n', file=ss)
        print(f'    <img alt="{plot_alt_text}" width="{width}" height="{height}" src="data:image/png;base64,{data}"/>', file=ss)

        return ss.getvalue()

    @env.macro
    def embed_options(reference, max_width=100):
        from openmdao.api import OptionsDictionary
        options_dict = get_object_from_reference(reference)

        if isinstance(options_dict, OptionsDictionary):
            # instance of an OptionsDictionary given
            print('od instance')
            od = options_dict
        elif issubclass(options_dict, OptionsDictionary):
            print('od class')
            od = options_dict()
        else:
            return f'Invalid OptionsDictionary: {reference}'

        return _options_dict_to_markdown(od)

    @env.macro
    def doc_env():
        "Document the environment"
        return {name:getattr(env, name) for name in dir(env) if not name.startswith('_')}

    @env.macro
    def api_doc(reference, members=None):

        module = '.'.join(reference.split('.')[:-1])
        item = reference.split('.')[-1]

        obj = getattr(get_object_from_reference(module), item)

        ss = io.StringIO()

        if inspect.isfunction(obj):
            _function_doc_markdown(obj, reference, outstream=ss)
        elif inspect.isclass(obj):
            _class_doc_markdown(obj, reference, members=members, outstream=ss)
        elif inspect.ismethod(obj):
            _function_doc_markdown(obj, reference, outstream=ss)

        return ss.getvalue()


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


def _get_test_source(reference):
    obj = get_object_from_reference(reference)
    try:
        method = obj._method
    except AttributeError:
        raise RuntimeError(f'To embed test {reference} in documentation, it must be wrapped with the dymos.utils.doc_utils.save_for_docs decorator')

    func_name = reference.split('.')[-1]

    source = ''.join(inspect.getsourcelines(method)[0])

    re_declaration = re.compile(rf'(def {func_name}\(.+?\):)', flags=(re.DOTALL | re.MULTILINE))

    match_dec = re_declaration.search(source)

    start = match_dec.span()[1]

    # Strip off decorators and the test method declaration
    source = source[start:]

    source = textwrap.dedent(source)
    source = source.strip()

    # source = textwrap.indent(source)

    return source


def _function_doc_markdown(func, reference, outstream=sys.stdout, indent='', method=False):
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
    doc = FunctionDoc(func)

    # # print(f'{indent}!!! abstract "{reference}"\n', file=outstream)
    # print(f'{indent}=== "{reference}"\n', file=outstream)

    if not method:
        print(f'{indent}!!! abstract "{reference}"\n', file=outstream)
    else:
        print(f'{indent}!!! abstract ""\n', file=outstream)
    indent = indent + '    '

    print(f"{indent}**{doc['Signature']}**\n", file=outstream)
    print('', file=outstream)

    if doc['Summary']:
        print(indent + ' '.join(doc['Summary']), file=outstream)

    if doc['Extended Summary']:
        print(indent + ' '.join(doc['Extended Summary']) + '\n', file=outstream)

    print('', file=outstream)

    print(f'{indent}**Arguments:**\n', file=outstream)

    for p in doc['Parameters']:
        print(f'{indent}**{p.name}**: {" ".join(p.desc)}', file=outstream)
        print('', file=outstream)

    if doc['Raises']:
        print('{indent}**Raises:**\n', file=outstream)

        for p in doc['Raises']:
            print(f'{indent}**{p.name}**: {" ".join(p.desc)}', file=outstream)
            print('', file=outstream)

    # for key in doc.keys():
    #     print(key, file=outstream)

def _class_doc_markdown(cls, reference, members=None, outstream=sys.stdout, indent=''):
    """

    Parameters
    ----------
    cls
    reference
    members
    outstream

    Returns
    -------

    """
    doc = ClassDoc(cls)

    print(f'{indent}### class {reference}\n', file=outstream)

    indent = indent + ''

    if doc['Summary']:
        print(indent + ' '.join(doc['Summary']), file=outstream)

    if doc['Extended Summary']:
        print(indent + ' '.join(doc['Extended Summary']) + '\n', file=outstream)

    print('', file=outstream)
    print(f"{indent}**{doc['Signature']}**\n", file=outstream)

    print(f'{indent}**Public API Methods:**\n', file=outstream)

    for p in doc['Methods']:
        if members is not None and p.name in members:
            ref = '.'.join((reference, p.name))
            print(f'{indent}=== "{p.name}"\n', file=outstream)
            _function_doc_markdown(getattr(cls, p.name), ref, outstream=outstream, indent=indent + "    ", method=True)

    for key in doc.keys():
        print(key, file=outstream)

def _api_doc(reference, members=True):

    module = '.'.join(reference.split('.')[:-1])
    item = reference.split('.')[-1]

    obj = getattr(get_object_from_reference(module), item)

    if inspect.isfunction(obj):
        return _function_doc_markdown(obj, reference)
    elif inspect.isclass(obj):
        return _class_doc_markdown(obj, members)

    # for key in doc.keys():
    #     print(key)

    return obj

def _options_dict_to_markdown(od):
    """
    Generate reStructuredText view of the options table.

    Returns
    -------
    list of str
        A rendition of the options as an rST table.
    """
    lines = od.__rst__()

    # Now the lines are in rst format, convert to markdown
    # First change = to - and space to | in the header rows
    for i in range(len(lines)):
        if set(lines[i]) == {'=', ' '}:
            lines[i] = lines[i].replace('=', '-')
            lines[i] = '|'.join(lines[i].split())
            lines[i] = '|' + lines[i] + '|'
        else:
            lines[i] = '|' + lines[i]
            lines[i] = lines[i][:-1] + '|'

    # Now find the location of | in the first row, and make sure | are in that location in each row.
    locs = [m.start() for m in re.finditer(r'[|]', lines[0])]

    for i in range(len(lines)):
        lstline = list(lines[i])
        for loc in locs:
            lstline[loc] = '|'
        lines[i] = ''.join(lstline)

    md = '\n'.join(lines[1:-1])
    print(md)

    return md


if __name__ == '__main__':
    obj = get_object_from_reference('dymos.phase.options.TimeOptionsDictionary')

    od = obj()

    print(od)

    md = _options_dict_to_markdown(od)

    for line in md:
        print(line)

    # print(s)

    # reference = 'dymos.run_problem'
    # module = '.'.join(reference.split('.')[:-1])
    # item = reference.split('.')[-1]
    #
    # obj = getattr(get_object_from_reference(module), item)
    #
    # _function_doc_markdown(obj, reference)
    #


    # reference = 'dymos.Trajectory'
    # module = '.'.join(reference.split('.')[:-1])
    # item = reference.split('.')[-1]
    #
    # obj = getattr(get_object_from_reference(module), item)
    #
    # _class_doc_markdown(obj, reference, members=['add_phase', 'link_phases'])
    #
    # obj2 = getattr(get_object_from_reference('dymos.Trajectory'), 'add_phase')
    #
    # print(obj2)



    # print(module.run_problem)

