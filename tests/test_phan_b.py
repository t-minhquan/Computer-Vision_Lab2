"""Run with .venv/Scripts/python.exe -m unittest discover -s tests -v."""
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'notebooks/phan_b_phan_vung_anh.ipynb'
DOCUMENT = json.loads(PATH.read_text(encoding='utf-8'))
NS = {}
for index, cell in enumerate(DOCUMENT['cells']):
    if cell['cell_type'] == 'code' and cell.get('metadata', {}).get('section') == 'implementation':
        exec(compile(''.join(cell['source']), f'{PATH.name}:cell{index}', 'exec'), NS)


class PartBTests(unittest.TestCase):
    def test_cleanup_and_shared_group_info(self):
        for path in ROOT.glob('notebooks/*.ipynb'):
            notebook = json.loads(path.read_text(encoding='utf-8'))
            markdown = '\n'.join(''.join(c['source']) for c in notebook['cells']
                                 if c['cell_type'] == 'markdown')
            self.assertNotIn('Các prompt yêu cầu dùng để tạo template', markdown)
            self.assertNotIn('| Thành viên |', markdown)
            self.assertIn('../README.md#nhom-va-phan-cong', markdown)
            for index, cell in enumerate(notebook['cells']):
                declared_prompt = cell.get('metadata', {}).get('ai_prompt')
                if cell['cell_type'] == 'code' and declared_prompt:
                    self.assertGreater(index, 0)
                    self.assertEqual(notebook['cells'][index - 1]['cell_type'], 'markdown')
                    self.assertIn(declared_prompt, ''.join(notebook['cells'][index - 1]['source']))
                if cell['cell_type'] == 'code' and cell.get('metadata', {}).get('section') == 'data_preparation':
                    self.assertGreater(index, 0)
                    previous = notebook['cells'][index - 1]
                    self.assertEqual(previous['cell_type'], 'markdown')
                    self.assertIn('tạo ảnh thử đi xong tự ném prompt ở trước cell, lưu ảnh vô image B nha',
                                  ''.join(previous['source']))
        self.assertIn('| Thành viên | MSSV | Công việc | Trạng thái |',
                      (ROOT / 'README.md').read_text(encoding='utf-8'))

    def test_classic_hand_calculated(self):
        image = np.array([[0., 1., 2., 10., 10.]])
        for order in ('stack', 'queue'):
            labels = NS['region_growing_classic'](image, [(0, 0), (0, 4)], 1., order)
            np.testing.assert_array_equal(labels, [[0, 0, -1, 1, 1]])
            self.assertEqual(labels.dtype, np.int32)
        # The changing mean allows the third pixel at T=1.5.
        np.testing.assert_array_equal(
            NS['region_growing_classic'](image, [(0, 0), (0, 4)], 1.5), [[0, 0, 0, 1, 1]])

    def test_classic_accepts_neighbors_before_pushing_pdf_page6(self):
        # Seed accepts intensity 1, so mean becomes .5 and then accepts 2.
        # A candidate-pop stack would reject 2 prematurely using mean 0.
        image = np.array([[0., 1.], [2., 3.]])
        for order in ('stack', 'queue'):
            with self.subTest(order=order):
                np.testing.assert_array_equal(
                    NS['region_growing_classic'](image, [(0, 0)], 1.5, order),
                    [[0, 0], [0, -1]])

    def test_implementation_uses_only_permitted_libraries(self):
        imports = set()
        calls = []
        for cell in DOCUMENT['cells']:
            if cell['cell_type'] != 'code' or cell.get('metadata', {}).get('section') != 'implementation':
                continue
            tree = ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module)
                elif isinstance(node, ast.Call):
                    calls.append(ast.unparse(node.func))
        self.assertEqual(imports, {'numpy', 'heapq'})
        self.assertFalse(any(call.startswith(('cv2.', 'skimage.', 'scipy.')) for call in calls))
        self.assertFalse(any(call in ('eval', 'exec', '__import__') for call in calls))

    def test_full_partition_and_seed_preservation(self):
        image = np.arange(63, dtype=float).reshape(7, 9)
        seeds = np.array([[1, 1], [5, 7], [3, 4]])
        for connectivity in (4, 8):
            for stale in ('lazy', 'frozen'):
                for tie in ('insertion', 'raster', 'random'):
                    with self.subTest(connectivity=connectivity, stale=stale, tie=tie):
                        np.random.seed(7)
                        labels, boundary = NS['srg'](image, seeds, connectivity, stale, tie)
                        self.assertEqual(labels.dtype, np.int32)
                        self.assertEqual(boundary.dtype, np.bool_)
                        self.assertEqual(labels.shape, image.shape)
                        self.assertEqual(boundary.shape, image.shape)
                        self.assertTrue(np.all(labels >= 0))
                        np.testing.assert_array_equal(np.unique(labels), [0, 1, 2])
                        np.testing.assert_array_equal(labels[seeds[:, 0], seeds[:, 1]], [0, 1, 2])
                        self.assertFalse(boundary[seeds[:, 0], seeds[:, 1]].any())

    def test_boundary_event_and_small_images(self):
        labels, boundary = NS['srg'](np.array([[0., 0., 10.]]), [(0, 0), (0, 2)])
        np.testing.assert_array_equal(labels, [[0, 0, 1]])
        np.testing.assert_array_equal(boundary, [[False, True, False]])
        for shape in ((1, 1), (1, 5), (5, 1)):
            image = np.ones(shape)
            labels, boundary = NS['srg'](image, [(0, 0)])
            self.assertFalse(labels.any())
            self.assertFalse(boundary.any())

    def test_tie_rules_and_random_reproducibility(self):
        image = np.zeros((7, 7))
        seeds = [(5, 5), (1, 1)]
        insertion = NS['srg'](image, seeds, tie='insertion')[0]
        raster = NS['srg'](image, seeds, tie='raster')[0]
        self.assertTrue(np.any(insertion != raster))
        np.random.seed(42)
        first = NS['srg'](image, seeds, tie='random')
        np.random.seed(42)
        second = NS['srg'](image, seeds, tie='random')
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        reverse = NS['srg'](image, seeds[::-1], tie='raster')[0]
        np.testing.assert_array_equal(raster, 1 - reverse)

    def test_lazy_and_frozen_have_distinct_references(self):
        # lazy means adapt toward the gradient; frozen compares with seed intensities.
        image = np.array([[0., 1., 2., 3., 4., 7.]])
        seeds = [(0, 0), (0, 5)]
        lazy = NS['srg'](image, seeds, stale='lazy')[0]
        frozen = NS['srg'](image, seeds, stale='frozen')[0]
        np.testing.assert_array_equal(lazy, [[0, 0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(frozen, [[0, 0, 0, 0, 1, 1]])

    def test_frames_are_independent_and_match_final(self):
        image = np.arange(35, dtype=float).reshape(5, 7)
        seeds = [(0, 0), (4, 6)]
        for kwargs in ({}, {'stale': 'frozen', 'connectivity': 8, 'tie': 'random'}):
            np.random.seed(19)
            frames = NS['srg_step_frames'](image, seeds, 8, **kwargs)
            self.assertEqual(len(frames), 8)
            self.assertEqual([np.count_nonzero(frame >= 0) for frame in frames],
                             [5, 9, 14, 18, 22, 27, 31, 35])
            self.assertTrue(all(not np.shares_memory(a, b)
                                for i, a in enumerate(frames) for b in frames[i + 1:]))
            np.random.seed(19)
            np.testing.assert_array_equal(frames[-1], NS['srg'](image, seeds, **kwargs)[0])
        tiny = NS['srg_step_frames'](np.zeros((1, 1)), [(0, 0)], 8)
        self.assertEqual(len(tiny), 8)
        tiny[0][0, 0] = 99
        self.assertEqual(tiny[-1][0, 0], 0)
        self.assertTrue(np.all(NS['srg_step_frames'](image, seeds, 1)[0] >= 0))

    def test_frames_pdf_percentage_milestones(self):
        frames = NS['srg_step_frames'](np.zeros((10, 10)), [(0, 0), (9, 9)])
        self.assertEqual([int(np.count_nonzero(frame >= 0)) for frame in frames],
                         [13, 25, 38, 50, 63, 75, 88, 100])
        # If seeds already reach a target, capture the initialized partition.
        seeds = np.array([[0, 0], [0, 1], [1, 0]])
        dense = NS['srg_step_frames'](np.zeros((2, 2)), seeds, 4)
        self.assertEqual([int(np.count_nonzero(frame >= 0)) for frame in dense], [3, 3, 3, 4])
        self.assertTrue(all(not np.shares_memory(a, b)
                            for i, a in enumerate(dense) for b in dense[i + 1:]))

    def test_sensitivity_label_alignment_and_jitter(self):
        image = np.arange(35, dtype=float).reshape(5, 7)
        seeds = [(0, 0), (4, 6)]
        calls = [0]
        def permuted_labels(img, points):
            labels = NS['srg'](img, points, stale='frozen')[0]
            calls[0] += 1
            return labels * (-1 if calls[0] % 2 else 1) + 100
        result = NS['seed_sensitivity'](image, seeds, permuted_labels, 3, 0)
        np.testing.assert_array_equal(result, np.zeros(image.shape))
        np.random.seed(12)
        first = NS['seed_sensitivity'](image, seeds, NS['srg'], 6, 10)
        np.random.seed(12)
        second = NS['seed_sensitivity'](image, seeds, NS['srg'], 6, 10)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float64)
        self.assertTrue(np.all((first >= 0) & (first <= 1)))
        full_seeds = np.argwhere(np.ones((2, 2), dtype=bool))
        np.testing.assert_array_equal(
            NS['seed_sensitivity'](np.zeros((2, 2)), full_seeds, NS['srg'], 2, 3), np.zeros((2, 2)))

    def test_sensitivity_zero_jitter_with_random_segmentation(self):
        image = np.zeros((7, 7))
        seeds = [(1, 1), (5, 5)]
        random_segment = lambda img, pts: NS['srg'](img, pts, tie='random')
        np.random.seed(42)
        result = NS['seed_sensitivity'](image, seeds, random_segment, 20, 0)
        np.testing.assert_array_equal(result, np.zeros(image.shape))
        np.random.seed(8)
        first = NS['seed_sensitivity'](image, seeds, random_segment, 5, 2)
        np.random.seed(8)
        second = NS['seed_sensitivity'](image, seeds, random_segment, 5, 2)
        np.testing.assert_array_equal(first, second)

    def test_sensitivity_segment_rng_does_not_change_seed_jitter(self):
        image = np.zeros((5, 5))
        seeds = [(0, 0), (4, 4)]
        paths, states = [], []
        for draws in (0, 50):
            seen = []
            def segment(img, pts):
                seen.append(pts.copy())
                np.random.random(draws)
                return NS['srg'](img, pts, stale='frozen')
            np.random.seed(27)
            NS['seed_sensitivity'](image, seeds, segment, 5, 2)
            paths.append(seen)
            states.append(np.random.get_state())
        np.testing.assert_array_equal(paths[0], paths[1])
        self.assertEqual(states[0][0], states[1][0])
        np.testing.assert_array_equal(states[0][1], states[1][1])
        self.assertEqual(states[0][2:], states[1][2:])

    def test_sensitivity_restores_rng_when_segment_raises(self):
        for failing_call in (1, 2):
            calls = [0]
            def failing(img, pts):
                calls[0] += 1
                np.random.random(10)
                if calls[0] == failing_call:
                    raise RuntimeError('segment failed')
                return NS['srg'](img, pts)
            np.random.seed(19)
            before = np.random.get_state()
            with self.assertRaisesRegex(RuntimeError, 'segment failed'):
                NS['seed_sensitivity'](np.zeros((3, 3)), [(0, 0)], failing, 1, 0)
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])

    def test_invalid_inputs(self):
        image = np.zeros((3, 3))
        for seeds in ([], [(0, 0), (0, 0)], [(-1, 0)], [(3, 0)], [(0.5, 1)]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                NS['srg'](image, seeds)
        for options in ({'connectivity': 6}, {'stale': 'bad'}, {'tie': 'bad'}):
            with self.assertRaises(ValueError):
                NS['srg'](image, [(0, 0)], **options)
        with self.assertRaises(ValueError):
            NS['region_growing_classic'](image, [(0, 0)], -1)
        with self.assertRaises(ValueError):
            NS['srg_step_frames'](image, [(0, 0)], 0)
        with self.assertRaises(ValueError):
            NS['seed_sensitivity'](image, [(0, 0)], NS['srg'], n_trials=0)
        with self.assertRaises(ValueError):
            NS['seed_sensitivity'](image, [(0, 0)], NS['srg'], jitter=-1)


class RealImagePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cv2
        cls.plt = plt
        cls.real_ns = dict(NS, plt=plt, cv2=cv2, csv=csv, Path=Path,
                           b_rng_seed=42, b_run_stamp='synthetic-unit-test')
        # Only load function definitions: never execute notebook result-writing cells.
        for ident in ('b-verification-setup', 'b-real-images-helpers'):
            cell = next(c for c in DOCUMENT['cells'] if c.get('id') == ident)
            tree = ast.parse(''.join(cell['source']))
            functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
                                   type_ignores=[])
            exec(compile(functions, ident, 'exec'), cls.real_ns)

    def test_two_missing_real_images_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory) / 'results'
            cases = [{'name': name, 'path': None, 'seeds': []} for name in ('real_01', 'real_02')]
            reports = self.real_ns['b_run_real_images'](cases, image_dir=directory, output_dir=output)
            self.assertEqual([r['status'] for r in reports], ['missing_image', 'missing_image'])
            self.assertFalse(output.exists())

    def test_two_configured_images_load_resize_and_compare(self):
        # Fixtures are synthetic; no claims about real photo segmentation.
        with tempfile.TemporaryDirectory() as directory, patch('matplotlib.pyplot.show'), patch('builtins.print'):
            folder = Path(directory)
            for name, shape in [('one', (64, 96)), ('two', (32, 24))]:
                image = np.tile(np.linspace(0, 255, shape[1], dtype=np.uint8), (shape[0], 1))
                self.plt.imsave(folder / (name + '.png'), np.repeat(image[..., None], 3, axis=2))
            cases = [
                {'name': 'real_01', 'path': 'one.png', 'source': 'synthetic fixture',
                 'seeds': [(3, 3), (16, 25)], 'T': 30},
                {'name': 'real_02', 'path': 'two.png', 'source': 'synthetic fixture',
                 'seeds': [(3, 3), (25, 20)], 'T': 30},
            ]
            output = folder / 'results'
            reports = self.real_ns['b_run_real_images'](cases, max_side=32, image_dir=folder, output_dir=output)
            self.assertEqual([r['status'] for r in reports], ['completed', 'completed'])
            self.assertEqual([r['working_shape'] for r in reports], [(21, 32), (32, 24)])
            for report in reports:
                self.assertEqual(report['image'].ndim, 2)
                labels = report['result']['labels']
                self.assertTrue(np.all(labels['srg'] >= 0))
                self.assertTrue(all(value.shape == report['working_shape'] for value in labels.values()))
                self.assertEqual(len(report['result']['pairwise']), 3)
                for suffix in ('_input.png', '_methods.png', '_coverage.csv', '_pairwise.csv'):
                    self.assertTrue((output / (report['name'] + suffix)).is_file())
            with (output / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row['source'] == 'synthetic fixture' for row in rows))

    def test_missing_and_invalid_seeds_do_not_run_segmentation(self):
        with tempfile.TemporaryDirectory() as directory, patch('matplotlib.pyplot.show'), patch('builtins.print'):
            folder = Path(directory)
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            self.plt.imsave(folder / 'preview.png', image)
            cases = [
                {'name': 'real_01', 'path': 'preview.png', 'seeds': []},
                {'name': 'real_02', 'path': 'preview.png', 'seeds': [(99, 99)]},
            ]
            output = folder / 'results'
            reports = self.real_ns['b_run_real_images'](cases, image_dir=folder, output_dir=output)
            self.assertEqual([r['status'] for r in reports], ['missing_seeds', 'invalid_seeds'])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
