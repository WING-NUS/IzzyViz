import os
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from izzyviz import (
    check_stability_heatmap_with_gradient_color,
    compare_two_attentions_with_circles,
    visualize_attention_evolution_sparklines,
    visualize_attention_matrix,
    visualize_attention_overview,
)
from izzyviz.visualization import _find_top_cells


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(123)
        self.labels = ["[CLS]", "a", "b", "[SEP]"]

    def tearDown(self):
        plt.close("all")

    def test_find_top_cells_includes_ties(self):
        data = np.ones((3, 3))
        cells = _find_top_cells(data, 4)
        self.assertEqual(len(cells), 9)

    def test_visualize_attention_matrix_accepts_deprecated_aliases(self):
        matrix = self.rng.random((4, 4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ax, _ = visualize_attention_matrix(
                matrix,
                x_labels=self.labels,
                y_labels=self.labels,
                if_interval=True,
                if_top_cells=True,
                lean_more=True,
                save_path=None,
            )

        self.assertIsNotNone(ax)
        self.assertGreaterEqual(len(caught), 3)

    def test_save_path_none_does_not_create_default_files(self):
        matrix = self.rng.random((4, 4))
        stability = self.rng.random((3, 4, 4))
        attention_time = self.rng.random((3, 2, 2, 4, 4))
        attention_time = attention_time / attention_time.sum(axis=-1, keepdims=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                compare_two_attentions_with_circles(matrix, matrix, self.labels)
                check_stability_heatmap_with_gradient_color(
                    stability,
                    x_labels=self.labels,
                    y_labels=self.labels,
                )
                visualize_attention_evolution_sparklines(
                    attention_time,
                    tokens=self.labels,
                    layer=1,
                    head=0,
                )

                self.assertFalse(Path("attention_comparison_circles.pdf").exists())
                self.assertFalse(
                    Path("check_stability_heatmap_with_gradient_color.pdf").exists()
                )
                self.assertFalse(Path("attention_evolution_sparklines.pdf").exists())
            finally:
                os.chdir(old_cwd)

    def test_sparkline_requires_tokens(self):
        attention_time = self.rng.random((3, 2, 2, 4, 4))
        with self.assertRaises(ValueError):
            visualize_attention_evolution_sparklines(
                attention_time,
                tokens=None,
                layer=1,
                head=0,
            )

    def test_compare_validates_shape_and_tokens(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            compare_two_attentions_with_circles(
                np.ones((4, 4)),
                np.ones((3, 4)),
                self.labels,
            )

        with self.assertRaisesRegex(ValueError, "len\\(tokens\\)"):
            compare_two_attentions_with_circles(
                np.ones((4, 4)),
                np.ones((4, 4)),
                ["a"],
            )

    def test_stability_validates_labels_and_radial_resolution(self):
        matrices = self.rng.random((3, 4, 4))

        with self.assertRaisesRegex(ValueError, "len\\(x_labels\\)"):
            check_stability_heatmap_with_gradient_color(
                matrices,
                x_labels=["a"],
                y_labels=self.labels,
            )

        with self.assertRaisesRegex(ValueError, "radial_resolution"):
            check_stability_heatmap_with_gradient_color(
                matrices,
                x_labels=self.labels,
                y_labels=self.labels,
                radial_resolution=1,
            )

    def test_stability_accepts_list_of_torch_tensors(self):
        matrices = [torch.ones(4, 4), torch.ones(4, 4)]
        ax = check_stability_heatmap_with_gradient_color(
            matrices,
            x_labels=self.labels,
            y_labels=self.labels,
            save_path=None,
        )
        self.assertIsNotNone(ax)

    def test_sparkline_validates_layer_head_and_square_shape(self):
        attention_time = self.rng.random((3, 2, 2, 4, 4))

        with self.assertRaisesRegex(ValueError, "layer"):
            visualize_attention_evolution_sparklines(
                attention_time,
                tokens=self.labels,
                layer=2,
                head=0,
            )

        with self.assertRaisesRegex(ValueError, "square"):
            visualize_attention_evolution_sparklines(
                self.rng.random((3, 1, 1, 4, 3)),
                tokens=self.labels,
                layer=0,
                head=0,
            )

    def test_sparkline_auto_color_is_independent_from_normalization(self):
        attention_time = np.full((3, 1, 1, 2, 2), 0.1)
        attention_time[:, 0, 0, 0, 0] = 0.9

        ax = visualize_attention_evolution_sparklines(
            attention_time,
            tokens=["a", "b"],
            layer=0,
            head=0,
            normalize_sparklines=True,
            sparkline_color_dark="navy",
            sparkline_color_light="white",
            sparkline_color_mode="auto",
            save_path=None,
        )

        self.assertEqual(ax.lines[0].get_color(), "white")

    def test_overview_smoke(self):
        fig, axes = visualize_attention_overview(
            self.rng.random((2, 2, 4, 4)),
            shared_color_scale=True,
            shared_cbar=True,
            save_path=None,
        )
        self.assertEqual(axes.shape, (2, 2))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
