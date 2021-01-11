# Contributing to Dymos

Dymos is open-source software and the developers welcome collaboration with the community on finding and fixing bugs or requesting and implementing new features.

## Found a bug in Dymos?

If you believe you've found a bug in Dymos, [submit a new issue](https://github.com/OpenMDAO/dymos/issues).
If at all possible, please include a functional code example which demonstrates the issue (the expected behavior vs. the actual behavior).

## Fixed a bug in Dymos?

If you believe you have a fix for an existing bug in Dymos, please submit the fix as [pull request](https://github.com/OpenMDAO/dymos/pulls).
Under the "related issues" section of the pull request template, include the issue resolved by the pull request using Github's [referencing syntax](https://help.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue).
When submitting a bug-fix pull request, please include a [unit test](https://docs.python.org/2/library/unittest.html) that demonstrates the corrected behavior.
This will prevent regressions in the future.

## Need new functionality in Dymos?

If you would like to have new functionality that currently doesn't exist in Dymos, please submit your request via [the Dymos issues on Github](https://github.com/OpenMDAO/dymos/issues).
The Dymos development team is small and we can't promise that we'll add every requested capability, but we'll happily have a discussion and try to accommodate reasonable requests that fit within the goals of the library.

## Adding new examples

Adding new examples are a great way to contribute to Dymos.
They're a great introduction to the Dymos development process, and examples provide a great way for users to learn to apply Dymos in new applications.
Submit new examples via [the Dymos issues on Github](https://github.com/OpenMDAO/dymos/issues).
New examples should do the following:

- Include a new directory under the `dymos/examples` directory.
- A unittest should be included in a `doc` subfolder within the example directory.
- The unittest method should be self-contained (it should include all imports necessary to run the example).
- If you want to include output and/or plots from the example in the documentation (highly recommended), decorate the test with the `@dymos.utils.doc_utils.save_for_docs` decorator.  This will save the text and plot outputs from the test for inclusion in the Dymos documentation.
- A new markdown file should be added under `mkdocs/docs/examples/<example name>` within the Dymos repository.

The following [mkdocs macros](https://mkdocs-macros-plugin.readthedocs.io/en/latest/) have been developed to make it easier to document examples.

**inline_source**

Includes the text from the test method as a code block in the documentation.  For example, the brachistochrone example source code is included using the
following in the markdown documented.

```text
{{ "{{ inline_source('dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochrone.test_brachistochrone',
include_def=False,
include_docstring=False,
indent_level=0)
}}" }}
```

Arguments `include_def` and `include_docstring` dictate whether the method declaration and docstring are included in the markdown documentation, respectively.
The argument `indent_level` can be used to control the level of indentation added to the example, if necessary.

**embed_test_output**

Using the `@save_for_docs` decorator on a test method will save its output and any plots generated to subdirectory named `_outputs` under the test's `doc` directory.
Using `embed_test_output` will grab the output from the test and include it in the documentation as a block.

For example, from the brachistochrone documentation:

```text
{{ "{{ embed_test_output('dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochrone.test_brachistochrone') }}" }}
```

**embed_test_plot**

Similarly, the plots from the example can be included using the `embed_test_plot` macro:

```text
{{ "{{ embed_test_plot('dymos.examples.brachistochrone.doc.test_doc_brachistochrone.TestBrachistochrone.test_brachistochrone',
alt_text='The solution to the Brachistochrone problem',
index=1) }}" }}
```

Here `alt_text` is the text displayed when cursor is over the image, and `index` is the index of the plot figure from the test (default is 1, this index starts at 1).

## Running Tests

Dymos tests can be run with any test runner such as [nosetests](https://nose.readthedocs.io/en/latest/) or [pytest](https://docs.pytest.org/en/stable/).
However, due to some MPI-specific tests in our examples, we prefer our [testflo](https://github.com/OpenMDAO/testflo) package.
The testflo utility can be installed using

```
python -m pip install testflo
```

Testflo can be invoked from the top-level Dymos directory with:

```
testflo .
```

With pyoptsparse correctly installed and things working correctly, the tests should conclude after several minutes with a message like the following:
The lack of MPI capability or pyoptsparse will cause additional tests to be skipped.

```
The following tests were skipped:
test_command_line.py:TestCommandLine.test_ex_brachistochrone_reset_grid


OK


Passed:  450
Failed:  0
Skipped: 1

Ran 451 tests using 2 processes
```

