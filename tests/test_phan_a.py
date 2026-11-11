"""Run: .venv/Scripts/python.exe -m unittest discover -s tests -v"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/phan_a_harris_va_hai_bien_the.ipynb'
NS = {}
DOCUMENT = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
for index, cell in enumerate(DOCUMENT['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'{NOTEBOOK.name}:cell{index}', 'exec'), NS)


class PartATests(unittest.TestCase):
    def test_notebook_cleanup(self):
        self.assertFalse((NOTEBOOK.parent / 'phan_a_dac_trung_goc.ipynb').exists())
        self.assertTrue(''.join(DOCUMENT['cells'][0]['source']).startswith(
            '# Phần A — Harris và hai biến thể'))
        self.assertNotIn('prompt', NOTEBOOK.read_text(encoding='utf-8').lower())
        names = ('gaussian_blur structure_tensor harris_response nms_2d harris_classic '
                 'scale_pyramid log_normalized harris_laplace anms top_n repeatability').split()
        self.assertTrue(all(callable(NS[name]) for name in names))

    def test_constant_image(self):
        image = np.ones((25, 31), dtype=np.uint8) * 40
        np.testing.assert_allclose(NS['gaussian_blur'](image, 1.4), image)
        for component in NS['structure_tensor'](image, 2., 1.4):
            np.testing.assert_allclose(component, 0, atol=1e-20)
        self.assertEqual(NS['harris_classic'](image).shape, (0, 3))
        self.assertEqual(NS['harris_laplace'](image, NS['scale_pyramid']()).shape, (0, 4))
        self.assertEqual(NS['harris_response'](image, 2, 1.4).dtype, np.float64)

    def test_checkerboard(self):
        row, col = np.indices((128, 128))
        image = ((row // 16 + col // 16) % 2).astype(float)
        points = NS['harris_classic'](image)
        expected = np.array([(r - .5, c - .5) for r in range(16, 128, 16)
                             for c in range(16, 128, 16)])
        distances = np.sqrt(((expected[:, None] - points[None, :, :2]) ** 2).sum(axis=2))
        self.assertTrue(np.all(distances.min(axis=1) <= 1.0))
        self.assertTrue(np.all(np.diff(points[:, 2]) <= 0))

    def test_nms_boundaries_and_ties(self):
        response = np.array([[5., 1, 1], [0, 0, 0], [1, 1, 4]])
        mask = NS['nms_2d'](response)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[0, 0] and mask[2, 2])
        self.assertTrue(NS['nms_2d'](np.ones((2, 2))).all())
        with self.assertRaises(ValueError):
            NS['nms_2d'](response, 2)

    def test_tensor_normalization_and_log(self):
        image = np.tile(np.arange(81, dtype=float), (81, 1))
        mxx, mxy, myy = NS['structure_tensor'](image, 2., 1.4)
        np.testing.assert_allclose(mxx[20:-20, 20:-20], 1.4 ** 2, atol=1e-8)
        np.testing.assert_allclose(mxy, 0, atol=1e-15)
        np.testing.assert_allclose(myy, 0, atol=1e-15)
        x = np.arange(81, dtype=float) - 40
        quadratic = x[:, None] ** 2 + x[None, :] ** 2
        self.assertAlmostEqual(NS['log_normalized'](quadratic, 2.)[40, 40], 16., places=8)

    def test_scale_extrema(self):
        np.testing.assert_allclose(NS['scale_pyramid'](1., 2., 4), [1, 2, 4, 8])
        image = np.zeros((5, 5))
        candidates = np.array([[2., 2., 10.]])
        def log_for(values):
            return lambda img, sigma: np.full(img.shape, values[int(sigma) - 1])
        with patch.dict(NS, harris_classic=lambda *args: candidates,
                        log_normalized=log_for([1, 3, 2])):
            points = NS['harris_laplace'](image, [1, 2, 3])
            np.testing.assert_array_equal(points, [[2, 2, 2, 10]])
        for values in ([3, 2, 1], [1, 3, 3]):
            with patch.dict(NS, harris_classic=lambda *args: candidates,
                            log_normalized=log_for(values)):
                self.assertEqual(NS['harris_laplace'](image, [1, 2, 3]).shape, (0, 4))
        self.assertEqual(NS['harris_laplace'](image, [1, 2]).shape, (0, 4))

    def test_anms_and_top_n(self):
        points = np.array([[0., 0, 100], [0, 1, 80], [0, 10, 70], [0, 11, 60]])
        np.testing.assert_array_equal(NS['anms'](points, 3), points[[0, 2, 1]])
        np.testing.assert_array_equal(NS['top_n'](points[::-1], 2), points[:2])
        for name in ('anms', 'top_n'):
            self.assertEqual(NS[name](points, 0).shape, (0, 3))
            self.assertEqual(NS[name](points, 100).shape, (4, 3))
            self.assertEqual(NS[name](np.empty((0, 3)), 3).shape, (0, 3))
            np.testing.assert_array_equal(NS[name](points, 3), NS[name](points[::-1], 3))

    def test_repeatability(self):
        a = np.array([[1., 1], [3, 3], [9, 9]])
        identity = lambda xy: xy
        translate = lambda xy: xy + 2
        self.assertEqual(NS['repeatability'](a, a, identity, (10, 10), eps=0), 1.)
        self.assertEqual(NS['repeatability'](a, a[:1], identity, (10, 10), eps=0), 1 / 3)
        self.assertEqual(NS['repeatability'](a, a[:2] + 2, translate, (10, 10), eps=0), 1.)
        self.assertEqual(NS['repeatability'](a, a, lambda xy: xy + 100, (10, 10)), 0.)
        self.assertEqual(NS['repeatability'](a[:0], a, identity, (10, 10)), 0.)
        self.assertEqual(NS['repeatability'](a, a[:0], identity, (10, 10)), 0.)
        self.assertEqual(NS['repeatability'](a, a, lambda xy: xy * np.nan, (10, 10)), 0.)


if __name__ == '__main__':
    unittest.main()
