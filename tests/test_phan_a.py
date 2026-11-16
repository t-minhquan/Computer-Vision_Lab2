"""Test phần cài đặt A; không chạy các cell kiểm chứng/thí nghiệm."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


# Nạp riêng các cell chứa hàm cài đặt.
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/phan_a_harris_va_hai_bien_the.ipynb"

document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
NS = {}

for index, cell in enumerate(document["cells"]):
    role = cell.get("metadata", {}).get("template_role")

    if cell["cell_type"] == "code" and role in {
        "student_setup",
        "student_implementation",
    }:
        source = "".join(cell["source"])
        exec(
            compile(source, f"{NOTEBOOK.name}:cell{index}", "exec"),
            NS,
        )


class PartATests(unittest.TestCase):
    def test_required_functions_and_loader(self):
        """Đủ 11 hàm; không nạp các cell thí nghiệm."""
        names = (
            "gaussian_blur structure_tensor harris_response nms_2d "
            "harris_classic scale_pyramid log_normalized harris_laplace "
            "anms top_n repeatability"
        ).split()

        for name in names:
            with self.subTest(function=name):
                self.assertTrue(callable(NS.get(name)))

        for name in ("a_board", "a1_table", "a2_candidates"):
            self.assertNotIn(name, NS)

    def test_constant_image(self):
        """Ảnh hằng không tạo gradient hoặc góc."""
        image = np.full((25, 31), 40, dtype=np.uint8)
        original = image.copy()

        blurred = NS["gaussian_blur"](image, 1.4)
        np.testing.assert_allclose(blurred, image)
        self.assertEqual(blurred.dtype, np.float64)

        for component in NS["structure_tensor"](image, 2.0, 1.4):
            self.assertEqual(component.shape, image.shape)
            np.testing.assert_allclose(component, 0, atol=1e-20)

        self.assertEqual(
            NS["harris_classic"](image).shape,
            (0, 3),
        )
        self.assertEqual(
            NS["harris_laplace"](
                image,
                NS["scale_pyramid"](),
            ).shape,
            (0, 4),
        )

        np.testing.assert_array_equal(image, original)

    def test_checkerboard_positions(self):
        """Các giao điểm chuẩn và điểm phát hiện cách nhau không quá 1 px."""
        row, col = np.indices((128, 128))
        image = (
            ((row // 16 + col // 16) % 2) * 255
        ).astype(np.uint8)

        points = NS["harris_classic"](image)

        # Biên giữa hai pixel nằm tại tọa độ nửa nguyên.
        expected = np.array([
            (r - 0.5, c - 0.5)
            for r in range(16, 128, 16)
            for c in range(16, 128, 16)
        ])

        self.assertGreater(len(points), 0)

        distances = np.sqrt(
            (
                (expected[:, None] - points[None, :, :2]) ** 2
            ).sum(axis=2)
        )

        # Mọi giao điểm chuẩn đều có điểm phát hiện gần đó.
        self.assertTrue(np.all(distances.min(axis=1) <= 1.0))

        # Mọi điểm phát hiện đều nằm gần một giao điểm chuẩn.
        self.assertTrue(np.all(distances.min(axis=0) <= 1.0))

        # Đầu ra được sắp theo response giảm dần.
        self.assertTrue(np.all(np.diff(points[:, 2]) <= 0))

    def test_nms_boundaries_and_ties(self):
        """NMS xử lý biên và giữ các cực đại bằng nhau."""
        response = np.array([
            [5.0, 1, 0],
            [0,   0, 0],
            [0,   1, 4],
        ])

        expected = np.zeros((3, 3), dtype=bool)
        expected[0, 0] = True
        expected[2, 2] = True

        actual = NS["nms_2d"](response)

        self.assertEqual(actual.dtype, np.bool_)
        np.testing.assert_array_equal(actual, expected)

        # Theo quy ước hiện tại, NMS giữ điểm hòa.
        self.assertTrue(
            NS["nms_2d"](np.ones((2, 2))).all()
        )

        # Cửa sổ 1×1 giữ mọi điểm.
        self.assertTrue(
            NS["nms_2d"](response, 1).all()
        )

        with self.assertRaises(ValueError):
            NS["nms_2d"](response, 2)

    def test_tensor_normalization_and_log(self):
        """Đối chiếu tensor và LoG bằng các hàm có đạo hàm biết trước."""
        image = np.tile(
            np.arange(81, dtype=float),
            (81, 1),
        )

        mxx, mxy, myy = NS["structure_tensor"](
            image,
            2.0,
            1.4,
        )

        # I(row, col) = col:
        # Ix = 1, Iy = 0; tensor có hệ số sigma_d².
        np.testing.assert_allclose(
            mxx[20:-20, 20:-20],
            1.4 ** 2,
            atol=1e-8,
        )
        np.testing.assert_allclose(mxy, 0, atol=1e-15)
        np.testing.assert_allclose(myy, 0, atol=1e-15)

        # I(x, y) = x² + y² có Laplacian bằng 4.
        # sigma = 2 => sigma² * |Laplacian| = 16.
        x = np.arange(81, dtype=float) - 40
        quadratic = x[:, None] ** 2 + x[None, :] ** 2

        self.assertAlmostEqual(
            NS["log_normalized"](quadratic, 2.0)[40, 40],
            16.0,
            places=8,
        )

    def test_harris_formula_by_hand(self):
        """Kiểm tra R = det(M) - k * trace(M)² bằng số tính tay."""
        image = np.zeros((5, 5))

        # Mỗi bộ là (Mxx, Mxy, Myy), với k mặc định 0.04.
        cases = [
            ((0.0, 0.0, 0.0), 0.0),
            ((4.0, 0.0, 0.0), -0.64),
            ((4.0, 0.0, 4.0), 13.44),
            ((4.0, 2.0, 9.0), 25.24),
        ]

        for values, expected in cases:
            components = tuple(
                np.full(image.shape, value)
                for value in values
            )

            # Cố định tensor để kiểm tra riêng công thức Harris.
            with self.subTest(values=values), patch.dict(
                NS,
                structure_tensor=(
                    lambda *args, result=components: result
                ),
            ):
                actual = NS["harris_response"](
                    image,
                    2.0,
                    1.4,
                )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-12,
                )

    def test_scale_pyramid(self):
        """Các mức sigma tạo thành cấp số nhân."""
        np.testing.assert_allclose(
            NS["scale_pyramid"](1.0, 2.0, 4),
            [1, 2, 4, 8],
        )

        sigmas = NS["scale_pyramid"]()

        self.assertEqual(sigmas.shape, (10,))
        self.assertTrue(np.all(np.diff(sigmas) > 0))

        np.testing.assert_allclose(
            sigmas[1:] / sigmas[:-1],
            1.2,
        )

    def test_laplace_strict_scale_extrema(self):
        """Chỉ giữ cực đại LoG nghiêm ngặt ở tầng trong."""
        image = np.zeros((5, 5))
        candidates = np.array([[2.0, 2.0, 10.0]])

        cases = [
            ([1, 3, 2], True),
            ([3, 2, 1], False),
            ([1, 2, 3], False),
            ([1, 3, 3], False),
            ([3, 3, 1], False),
        ]

        for values, should_keep in cases:
            # Cố định ứng viên và LoG để kiểm tra riêng chọn scale.
            with self.subTest(values=values), patch.dict(
                NS,
                harris_classic=lambda *args: candidates.copy(),
                log_normalized=(
                    lambda img, sigma, v=values:
                    np.full(img.shape, v[int(sigma) - 1])
                ),
            ):
                actual = NS["harris_laplace"](
                    image,
                    [1, 2, 3],
                )

                if should_keep:
                    np.testing.assert_array_equal(
                        actual,
                        [[2, 2, 2, 10]],
                    )

                    # Bằng ngưỡng chưa đủ vì điều kiện là >.
                    self.assertEqual(
                        NS["harris_laplace"](
                            image,
                            [1, 2, 3],
                            log_thresh=3,
                        ).shape,
                        (0, 4),
                    )
                else:
                    self.assertEqual(actual.shape, (0, 4))

        # Hai tầng không có tầng giữa để xét cực đại scale.
        self.assertEqual(
            NS["harris_laplace"](image, [1, 2]).shape,
            (0, 4),
        )

    def test_anms_and_top_n(self):
        """Đối chiếu lựa chọn điểm bằng ví dụ tính tay."""
        points = np.array([
            [0.0, 0, 100],
            [0,   1, 80],
            [0,  10, 70],
            [0,  11, 60],
        ])
        original = points.copy()

        np.testing.assert_array_equal(
            NS["anms"](points, 3),
            points[[0, 2, 1]],
        )

        np.testing.assert_array_equal(
            NS["top_n"](points[::-1], 2),
            points[:2],
        )

        for name in ("anms", "top_n"):
            with self.subTest(function=name):
                self.assertEqual(
                    NS[name](points, 0).shape,
                    (0, 3),
                )
                self.assertEqual(
                    NS[name](points, 100).shape,
                    (4, 3),
                )
                self.assertEqual(
                    NS[name](np.empty((0, 3)), 3).shape,
                    (0, 3),
                )

                # Đảo thứ tự đầu vào không đổi kết quả.
                np.testing.assert_array_equal(
                    NS[name](points, 3),
                    NS[name](points[::-1], 3),
                )

        np.testing.assert_array_equal(points, original)

    def test_anms_robust_boundary(self):
        """Kiểm tra dấu > trong điều kiện c * Rj > Ri."""
        points = np.array([
            [0.0, 0, 100],
            [0,   1, 90],
            [0,  10, 80],
        ])

        # 90 == 0.9 * 100:
        # hai điểm đầu đều không có điểm mạnh hơn theo điều kiện >.
        np.testing.assert_array_equal(
            NS["anms"](points, 3, 0.9),
            points,
        )

        # Khi bán kính và response hòa, ưu tiên row rồi col.
        tied = np.array([
            [4.0, 4, 10],
            [1,   1, 10],
            [2,   2, 10],
        ])

        np.testing.assert_array_equal(
            NS["anms"](tied, 2, 1.0),
            tied[[1, 2]],
        )

    def test_repeatability(self):
        """Kiểm tra mẫu số, miền hợp lệ và bán kính ghép."""
        repeat = NS["repeatability"]

        a = np.array([
            [1.0, 1],
            [3,   3],
            [9,   9],
        ])
        identity = lambda points: points

        self.assertEqual(
            repeat(a, a, identity, (10, 10), 0),
            1.0,
        )
        self.assertEqual(
            repeat(a, a[:1], identity, (10, 10), 0),
            1 / 3,
        )

        # Điểm (9, 9) dịch thành (11, 11), nằm ngoài ảnh B.
        # Mẫu số chỉ còn hai điểm hợp lệ.
        self.assertEqual(
            repeat(
                a,
                a[:2] + 2,
                lambda points: points + 2,
                (10, 10),
                0,
            ),
            1.0,
        )

        self.assertEqual(
            repeat(a[:0], a, identity, (10, 10)),
            0.0,
        )
        self.assertEqual(
            repeat(a, a[:0], identity, (10, 10)),
            0.0,
        )
        self.assertEqual(
            repeat(
                a,
                a,
                lambda points: points * np.nan,
                (10, 10),
            ),
            0.0,
        )

        border = np.array([
            [0.0, 0],
            [9,   9],
            [10,  0],
            [-1,  0],
        ])
        self.assertEqual(
            repeat(border, border[:2], identity, (10, 10), 0),
            1.0,
        )

        # Theo quy ước hiện tại, nhiều điểm A được dùng chung điểm B.
        near = np.array([[1.0, 1], [1, 2]])
        self.assertEqual(
            repeat(near, near[:1], identity, (10, 10), 1),
            1.0,
        )

        # Khoảng cách đúng bằng eps vẫn được ghép.
        self.assertEqual(
            repeat(
                np.array([[0.0, 0]]),
                np.array([[3.0, 4]]),
                identity,
                (10, 10),
                5,
            ),
            1.0,
        )

    def test_invalid_inputs(self):
        """Dữ liệu hoặc tham số sai phải được báo lỗi."""
        image = np.zeros((5, 5))
        points = np.zeros((2, 3))

        calls = [
            lambda: NS["gaussian_blur"](image, 0),
            lambda: NS["gaussian_blur"]([[np.nan]], 1),
            lambda: NS["gaussian_blur"](np.zeros((2, 2, 3)), 1),
            lambda: NS["harris_classic"](image, thresh_rel=-0.1),
            lambda: NS["harris_laplace"](image, [1, 1, 2]),
            lambda: NS["anms"](points, 1, c_robust=0),
            lambda: NS["top_n"](points, -1),
            lambda: NS["repeatability"](
                points,
                points,
                lambda p: p[:, 0],
                (5, 5),
            ),
        ]

        for index, call in enumerate(calls):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main(verbosity=2)