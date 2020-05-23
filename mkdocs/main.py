import importlib
import inspect
import re
import textwrap


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
    def inline_plot(source):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        exec(source)

        return f'```python\n{source}\n```'


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


if __name__ == '__main__':
    obj = get_object_from_reference('dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochrone.test_brachistochrone_partials')

    src = ''.join(inspect.getsourcelines(obj)[0])
    print(src)