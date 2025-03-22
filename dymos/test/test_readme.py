import pathlib
import re
import subprocess
import tempfile
import unittest
from openmdao.utils.testing_utils import use_tempdirs

def extract_python_code(filename):
    """Extracts Python code blocks from a Markdown file and save them to a temp file"""
    with open(filename, "r") as f:
        content = f.read()

    # Find triple-backtick Python code blocks
    code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    # Create a temporary Python file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w")
    temp_file.write('\n'.join(code_blocks))
    temp_file.close()

    return temp_file.name  # Return the filename


@use_tempdirs
class TestReadme(unittest.TestCase):

    def test_readme_code(self):
        readme_path = pathlib.Path(__file__).parent.parent.parent / 'readme.md'
        try:
            script = extract_python_code(readme_path)
        except FileNotFoundError:
            self.skipTest('Test cannot be run from site-packages.')

        # Run the extracted Python file as a subprocess
        result = subprocess.run(["python", script], capture_output=True, text=True)

        # Print output for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Ensure the script runs without errors
        assert result.returncode == 0, f'Python script failed with errors:\n{result.stderr}'