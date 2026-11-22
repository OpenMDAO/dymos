import importlib.util
import json
import re
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

DOCS_DIR = Path(__file__).resolve().parents[2]
BOOK_DIR = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    'copy_build_artifacts', DOCS_DIR / 'copy_build_artifacts.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
copy_build_artifacts = _module.copy_build_artifacts

WATER_ROCKET_SVGS = (
    'water_rocket_overview.svg',
    'water_rocket_waterengine.svg',
)
WATER_ROCKET_FIGURES = BOOK_DIR / 'examples' / 'water_rocket' / 'figures'
WATER_ROCKET_NOTEBOOK = BOOK_DIR / 'examples' / 'water_rocket' / 'water_rocket.ipynb'


class TestCopyBuildArtifacts(unittest.TestCase):
    """
    Check that static SVGs referenced from raw HTML are copied into the HTML tree.
    """

    def test_svg_is_copied_preserving_relative_path(self):
        with TemporaryDirectory() as tmp:
            book = Path(tmp) / '_executed_book'
            src = book / 'examples' / 'water_rocket' / 'figures'
            src.mkdir(parents=True)
            (src / 'water_rocket_overview.svg').write_text('<svg id="overview"/>')
            (src / 'water_rocket_waterengine.svg').write_text('<svg id="engine"/>')
            (src / 'not_an_svg.png').write_text('png')
            (book / 'notes.md').write_text('ignore me')

            n_copied = copy_build_artifacts(book)

            html_fig = book / '_build' / 'html' / 'examples' / 'water_rocket' / 'figures'
            self.assertEqual(n_copied, 2)
            self.assertEqual(
                (html_fig / 'water_rocket_overview.svg').read_text(),
                '<svg id="overview"/>',
            )
            self.assertEqual(
                (html_fig / 'water_rocket_waterengine.svg').read_text(),
                '<svg id="engine"/>',
            )
            self.assertFalse((html_fig / 'not_an_svg.png').exists())
            self.assertFalse((book / '_build' / 'html' / 'notes.md').exists())

    def test_files_under_build_dir_are_not_recopied(self):
        with TemporaryDirectory() as tmp:
            book = Path(tmp) / '_executed_book'
            nested_build = book / '_build' / 'html' / '_build' / 'html'
            nested_build.mkdir(parents=True)
            (book / 'keep.svg').write_text('<svg id="keep"/>')
            (book / '_build' / 'stale.svg').write_text('<svg id="stale"/>')

            copy_build_artifacts(book)

            self.assertTrue((book / '_build' / 'html' / 'keep.svg').is_file())
            self.assertFalse((nested_build / 'stale.svg').exists())

    def test_water_rocket_source_svgs_copy_into_html_tree(self):
        with TemporaryDirectory() as tmp:
            book = Path(tmp) / '_executed_book'
            dest_src = book / 'examples' / 'water_rocket' / 'figures'
            dest_src.mkdir(parents=True)
            for name in WATER_ROCKET_SVGS:
                src = WATER_ROCKET_FIGURES / name
                self.assertTrue(src.is_file(), f'Missing source SVG: {src}')
                shutil.copy2(src, dest_src / name)

            copy_build_artifacts(book)

            for name in WATER_ROCKET_SVGS:
                copied = book / '_build' / 'html' / 'examples' / 'water_rocket' / 'figures' / name
                self.assertTrue(copied.is_file(), f'SVG was not copied to {copied}')
                self.assertGreater(copied.stat().st_size, 0)
                self.assertEqual(copied.read_bytes(), (WATER_ROCKET_FIGURES / name).read_bytes())

    def test_notebook_img_srcs_match_copied_svg_paths(self):
        with WATER_ROCKET_NOTEBOOK.open(encoding='utf-8') as f:
            notebook = json.load(f)

        img_srcs = []
        for cell in notebook['cells']:
            if cell.get('cell_type') != 'markdown':
                continue
            src = ''.join(cell.get('source', []))
            img_srcs.extend(re.findall(r'<img\s+src="([^"]+)"', src))

        self.assertEqual(
            img_srcs,
            [f'figures/{name}' for name in WATER_ROCKET_SVGS],
        )

        with TemporaryDirectory() as tmp:
            book = Path(tmp) / '_executed_book'
            dest_src = book / 'examples' / 'water_rocket' / 'figures'
            dest_src.mkdir(parents=True)
            for name in WATER_ROCKET_SVGS:
                shutil.copy2(WATER_ROCKET_FIGURES / name, dest_src / name)

            copy_build_artifacts(book)

            html_page_dir = book / '_build' / 'html' / 'examples' / 'water_rocket'
            for rel_src in img_srcs:
                resolved = (html_page_dir / rel_src).resolve()
                self.assertTrue(resolved.is_file(), f'Notebook img src {rel_src} did not resolve')


if __name__ == '__main__':
    unittest.main()
