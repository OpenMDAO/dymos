import base64
import importlib
import inspect
import io
import os
import re
import sys
import textwrap
from pathlib import Path
from redbaron import RedBaron

try:
    from numpydoc.docscrape import FunctionDoc, ClassDoc
except ImportError as e:
    print('Please install the numpydoc package to build the documentation for Dymos')
    raise e


def define_env(env):

    @env.macro
    def inline_source(reference, include_def=True, include_docstring=True, indent_level=0, show_line_numbers=True, highlight_lines=None):
        """
        Macro to embed the source code of the given reference into mkdocs.

        Parameters
        ----------
        reference : str
            The dotted path to the object whose source is to be displayed.
        include_def : bool
            If True, include the definition of the class, function, or method.
        include_docstring
            If True, include the docstring of the class, function, or method.
        indent_level : int
            The baseline indentation for the source.
        show_line_numbers : bool
            If True, display the line numbers of the source code.
        highlight_lines : sequence or None
            If provided, line numbers to be highlighted in the displayed source.

        Returns
        -------
        str
            Markdown-formatted source-code for the given reference.

        """
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

        line_numbers = ' linenums="1"' if show_line_numbers else ''

        if highlight_lines is not None:
            hl_lines = ' hl_lines="' + ' '.join([str(i) for i in highlight_lines]) + '"'
        else:
            hl_lines = ''

        result = f'```python{line_numbers}{hl_lines}\n{source}\n```'

        return textwrap.indent(result, indent)

    @env.macro
    def embed_plot_from_script(script_path, alt_text='', width=640, height=480):
        """
        Macro to embed a plot figure obtained by executing a script at the given path.

        Currently this only saves the last plot produced by the script.

        Parameters
        ----------
        script_path : str
            Path to the plot-generating script.
        alt_text : str
            Alternative text for the plots, for 508 compliance.
        width : int
            Width of the embedded plot, in pixels.
        height : int
            Height of the embedded plot, in pixels.

        Returns
        -------
        str
            An html image tag with the encoded plot data, used to directly include the given plot
            in mkdocs.

        """
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
        """
        Macro to embed a unittest.TestCase method in mkdocs.

        Return a markdown-formatted output from a test given by reference.

        The embedded test **must be decorated with the dymos.utils.doc_utils.save_for_docs** decorator!

        Parameters
        ----------
        reference : str
            The path to the embedded test method.

        Returns
        -------
        str
            Markdown-formatted output of the given test method.

        """
        test_case, test_method = reference.split('.')[-2:]
        testcase_obj = get_object_from_reference('.'.join(reference.split('.')[:-1]))
        test_dir = Path(inspect.getfile(testcase_obj)).parent
        output_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}.out')
        with open(output_file) as f:
            text = f.read()
        return f'```\n{text}\n```'

    @env.macro
    def embed_test_plot(reference, index=1, alt_text='', width=640, height=480):
        """
        Macro to embed a unittest.TestCase method plot in mkdocs.

        The test method is embedded as a set of tabs.
        The embedded test **must be decorated with the dymos.utils.doc_utils.save_for_docs** decorator!

        The first tab contains the test source.
        The second tab contains the test standard output.
        The remaining tabs display any matplotlib plots generated in the test, which are selected using the plots argument.

        Parameters
        ----------
        reference : str
            The path to the embedded test method.
        index : int
            The index of the plot to be embedded.
        alt_text : str
            Alternative text for the plots, for 508 compliance.
        width : int
            Width of the embedded plot, in pixels.
        height : int
            Height of the embedded plot, in pixels.

        Returns
        -------
        str
            The markdown source that provides the embedded plot file, encoded directly as html
        """
        test_case, test_method = reference.split('.')[-2:]
        testcase_obj = get_object_from_reference('.'.join(reference.split('.')[:-1]))
        test_dir = Path(inspect.getfile(testcase_obj)).parent
        plot_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}_{index}.png')

        with open(plot_file, 'rb') as f:
            buf = io.BytesIO(f.read())

        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f'<img alt="{alt_text}" width="{width}" height="{height}" src="data:image/png;base64,{data}"/>'

    @env.macro
    def embed_test(reference, script_name='script', plot_alt_text='', plots=None, plot_size=(640, 480),
                   show_output=True, show_script=True, plot_names=None):
        """
        Macro to embed a unittest.TestCase method source, output, and plots in mkdocs.

        The test method is embedded as a set of tabs.
        The embedded test **must be decorated with the dymos.utils.doc_utils.save_for_docs** decorator!

        The first tab contains the test source.
        The second tab contains the test standard output.
        The remaining tabs display any matplotlib plots generated in the test, which are selected using the plots argument.

        Parameters
        ----------
        reference : str
            The path to the embedded test method.
        script_name : str
            A heading for the test being run.
        plot_alt_text : str
            Alternative text for the plots, for 508 compliance.
        plots : tuple of ints or None
            The plots being requested (indexed at 1) or None for all plots.
        plot_size : tuple of width, height in pixels.
            The size of the plot figures being embedded in the markdown.
        show_output : bool
            Show output tab if True.
        show_script : bool
            Show script tab if True.
        plot_names : sequence of str or None
            If given, a sequence of tab titles to be assigned to the plots.

        Returns
        -------
        str
            The markdown source that provides a set of tabs for the test source, the test
            output, and the requested plots produced by the test.
        """
        ss = io.StringIO()
        # First tab for the source
        if show_script:
            src = textwrap.indent(_get_test_source(reference), '    ')
            print(f'=== "{script_name}"', file=ss)
            print('    ```python3', file=ss)
            print(src, file=ss)
            print('    ```', file=ss)

        # Second tab for the output
        test_case, test_method = reference.split('.')[-2:]
        testcase_obj = get_object_from_reference('.'.join(reference.split('.')[:-1]))
        test_dir = Path(inspect.getfile(testcase_obj)).parent

        if show_output:
            output_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}.out')
            with open(output_file) as f:
                text = f.read()

            print(f'=== "output"', file=ss)
            print('    ```', file=ss)
            print(textwrap.indent(text, '    '), file=ss)
            print('    ```', file=ss)

        # Remaining tabs are for plots

        for index in range(1, 100):
            if plots is None or index in plots:
                plot_file = test_dir.joinpath('_output').joinpath(f'{test_case}.{test_method}_{index}.png')
                if not os.path.exists(plot_file):
                    break

                with open(plot_file, 'rb') as f:
                    buf = io.BytesIO(f.read())

                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                width, height = plot_size
                try:
                    plot_name = plot_names[index-1] if plot_names else f'{index}'
                except:
                    plot_name = f'{index}'
                print(f'=== "{plot_name}"\n', file=ss)
                print(f'    <img alt="{plot_alt_text}" width="{width}" height="{height}" src="data:image/png;base64,{data}"/>\n', file=ss)

        return ss.getvalue()

    @env.macro
    def embed_options(reference, title=''):
        """
        Macro to embed an OpenMDAO OptionsDictionary in mkdocs.

        Parameters
        ----------
        reference : str
            The dotted path to the options dictionary class or instance being documented.
        title : str
            The title provided above the options table.

        Returns
        -------
        str
            A markdown-formatted table for the entries in the options dictionary.

        """
        from openmdao.api import OptionsDictionary
        options_dict = get_object_from_reference(reference)

        if isinstance(options_dict, OptionsDictionary):
            od = options_dict
        elif issubclass(options_dict, OptionsDictionary):
            od = options_dict()
        else:
            return f'Invalid OptionsDictionary: {reference}'

        return f'{title}\n{_options_dict_to_markdown(od)}'

    @env.macro
    def doc_env():
        "Document the environment"
        return {name:getattr(env, name) for name in dir(env) if not name.startswith('_')}

    @env.macro
    def api_doc(reference, members=None):
        """
        Embed API documentation of the given function, class, or method.

        Parameters
        ----------
        reference : str
            The path to the API function, class, or method being documented.
        members : sequence or None
            If reference is a class, the members of the class to be documented.
            Each will be displayed on a separate tab of the output.

        Returns
        -------
        str
            The markdown representation of the API documentation.

        """

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

    @env.macro
    def upgrade_doc(reference, feature, old_label='previous', new_label='updated'):
        """
        This macro embeds an "upgrade guide" from the given reference string.

        In this case, the reference should point to a test method.
        The test method should document the new/preferred way of doing something.

        In the body of the test method, this macro searches for the
        strings '# upgrade_doc: begin {name}' and '# upgrade_doc: end {name}'
        where {name} is some feature to be demonstrated.  The code between those comments will
        be documented in a tab-enclosed code-block with the label as given by 'new_label'.

        If old/deprecated behavior is to be shown for comparison, this should be placed in the
        test method's doc string between the '# upgrade_doc: begin {name}' and
        '# upgrade_doc: end {name}' strings.  If present, this code will be shown in a separate
        tab-enclosed code-block with the label given by 'old_label'.

        Parameters
        ----------
        reference : str
            The path to the upgrade test for the given feature.
        feature : str
            The name of the feature whose use is to be documented.
        old_label : str
            The tab label for the tab showing the old/deprecated behavior (if present).
        new_label : str
            The tab label for the tab showing the new/preferred behavior.

        Returns
        -------
        str
            The markdown representation of the API documentation.

        """
        ss = io.StringIO()
        _upgrade_doc_markdown(reference, feature, outstream=ss,
                              old_label=old_label, new_label=new_label)
        return ss.getvalue()


def _sub_unspecified_in_signature(signature):
    """
    Returns the function signature with all instances of '<object object at (hexcode)>' removed and
    replaced with 'unspecified'.

    Parameters
    ----------
    signature : str
        The default function signature.

    Returns
    -------
    str
        The signature with object instances replaced with the 'unspecified' keyword.

    """
    return signature
    return re.sub(r'\<object.+?\>', 'unspecified', signature)


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
    """
    Return the source code from the test specified by the gien reference.

    Parameters
    ----------
    reference : str
        A dotted path to the test method whose source is desired.

    Returns
    -------
    str
        The returned source, dedented and having had assert method calls removed.

    """
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

    # Remove the assert method calls from documentation.
    source = _strip_asserts(source)

    return source

def _strip_asserts(source):
    """
    Remove assert method calls from source code.

    Using RedBaron, replace some assert calls with print statements that print the actual
    value given in the asserts. Depending on the calls, the actual value can be the first or second
    argument.

    Parameters
    ----------
    source : str
        String containing source lines.

    Returns
    -------
    str
        Source with asserts removed.
    """
    rb = RedBaron(source)  # convert to RedBaron internal structure

    # findAll is slow, so only check the ones that are present.
    asserts = ['assertAlmostEqual', 'assertLess', 'assertGreater', 'assertEqual',
               'assert_equal_arrays', 'assertTrue', 'assertFalse', 'assert_near_equal',
               'assert_rel_error', 'assert_almost_equal', 'assert_allclose']

    for assert_type in asserts:
        assert_nodes = rb.findAll("NameNode", value=assert_type)
        for i in reversed(range(len(assert_nodes))):
            parent = assert_nodes[i].parent
            for j in reversed(range(len(parent.value))):
                assert_nodes[i].parent.remove(parent.value[j])

    return rb.dumps()

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
    sig = _sub_unspecified_in_signature(doc['Signature'])

    if not method:
        print(f'{indent}!!! abstract "{reference}"\n', file=outstream)
    else:
        print(f'{indent}!!! abstract ""\n', file=outstream)
    indent = indent + '    '

    print(f"{indent}**{sig}**\n", file=outstream)
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


def _options_dict_to_markdown(od):
    """
    Generate a markdown-formatted table to document the options dictionary.

    Returns
    -------
    str
        A markdown table representation of the options dictionay.
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

    return md

def _split_docstring(obj):
    docstring = inspect.getdoc(obj)
    lines = inspect.getsourcelines(obj)[0]
    docstring_start = docstring_end = -1
    for i, line in enumerate(lines):
        # print(line)
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            if docstring_start >= 0:
                docstring_end = i
                break
            else:
                docstring_start = i
    func_body = ''.join(lines[:docstring_start]) + ''.join(lines[docstring_end + 1:])
    return docstring, func_body

def _upgrade_doc_markdown(test_reference, feature, outstream=sys.stdout,
                          old_label='previous', new_label='updated'):
    obj = get_object_from_reference(test_reference)
    docstring, func_body = _split_docstring(obj)

    re_feature = re.compile(rf'# upgrade_doc: begin {feature}(.+?)# upgrade_doc: end {feature}',
                            flags=re.MULTILINE|re.DOTALL)

    doc_match = re_feature.search(docstring) if docstring else None
    body_match = re_feature.search(func_body)

    try:
        old_way = textwrap.dedent(doc_match.groups()[0].strip())
    except AttributeError:
        print(f'Unable to find feature label {feature} in the doc string of {test_reference}')
        old_way = None
    try:
        new_way = textwrap.dedent(body_match.groups()[0].strip())
    except AttributeError:
        print(func_body)
        raise ValueError(f'Unable to find feature label {feature} in the body of {test_reference}')

    indent = '    '

    print(f'=== "{new_label}"', file=outstream)
    print(f'{indent}```', file=outstream)
    print(f'{textwrap.indent(new_way, indent)}', file=outstream)
    print(f'{indent}```', file=outstream)

    if old_way:
        print(f'=== "{old_label}"', file=outstream)
        print(f'{indent}```', file=outstream)
        print(f'{textwrap.indent(old_way, indent)}', file=outstream)
        print(f'{indent}```', file=outstream)


if __name__ == '__main__':
    obj =_upgrade_doc_markdown('dymos.test.test_upgrade_guide.TestUpgrade_0_16_0.test_glob_timeseries_outputs',
                               'glob_timeseries_outputs')
