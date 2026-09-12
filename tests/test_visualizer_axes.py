import unittest
from unittest.mock import Mock

import torch
from rnntoolkit.flow_visualizer.axes_panel import AxesPreferencesPanel
from rnntoolkit.flow_visualizer.visualizer import FlowFieldVisualizer


class TestVisualizerAxes(unittest.TestCase):
    def test_default_labels_and_first_two_axes_are_selected(self):
        visualizer = object.__new__(FlowFieldVisualizer)
        axes = torch.eye(4)

        selected = visualizer._initialize_axes(axes, None)

        self.assertEqual(visualizer.axis_labels, ["axis_1", "axis_2", "axis_3", "axis_4"])
        self.assertEqual(visualizer.selected_axis_indices, [0, 1])
        self.assertTrue(torch.equal(selected, axes[:2]))

    def test_custom_labels_are_preserved(self):
        visualizer = object.__new__(FlowFieldVisualizer)
        axes = torch.eye(3)

        visualizer._initialize_axes(axes, ["choice", "memory", "output"])

        self.assertEqual(visualizer.axis_labels, ["choice", "memory", "output"])

    def test_axis_bank_and_labels_are_validated(self):
        cases = [
            (torch.ones(3), None, ValueError),
            (torch.ones(1, 3), None, ValueError),
            (torch.ones(3, 3), ["one", "two"], ValueError),
            (torch.ones(3, 3), ["one", 2, "three"], TypeError),
            (None, ["one", "two"], ValueError),
        ]

        for axes, labels, error in cases:
            with self.subTest(axes=axes, labels=labels), self.assertRaises(error):
                visualizer = object.__new__(FlowFieldVisualizer)
                visualizer._initialize_axes(axes, labels)

    def test_completing_a_new_pair_rebuilds_finder_and_marks_dirty(self):
        visualizer = object.__new__(FlowFieldVisualizer)
        axes = torch.eye(4)
        visualizer._initialize_axes(axes, None)
        old_finder = object()
        new_finder = object()
        visualizer.axes = axes[:2]
        visualizer.pages = [old_finder]
        visualizer.current_page = 1
        visualizer._flow_cache = {"cached": True}
        visualizer._flow_dirty = False
        visualizer.build_finder = Mock(return_value=new_finder)

        visualizer.toggle_axis(0)

        self.assertEqual(visualizer.selected_axis_indices, [1])
        self.assertEqual(visualizer.pages, [old_finder])

        visualizer.toggle_axis(2)

        self.assertEqual(visualizer.selected_axis_indices, [1, 2])
        self.assertTrue(torch.equal(visualizer.axes, axes[[1, 2]]))
        self.assertEqual(visualizer.pages, [new_finder])
        self.assertIsNone(visualizer._flow_cache)
        self.assertTrue(visualizer._flow_dirty)
        visualizer.build_finder.assert_called_once_with()

    def test_more_than_two_axes_can_be_staged_before_rebuilding(self):
        visualizer = object.__new__(FlowFieldVisualizer)
        axes = torch.eye(4)
        visualizer._initialize_axes(axes, None)
        old_finder = object()
        new_finder = object()
        visualizer.axes = axes[:2]
        visualizer.pages = [old_finder]
        visualizer.current_page = 1
        visualizer._flow_cache = {"cached": True}
        visualizer._flow_dirty = False
        visualizer.build_finder = Mock(return_value=new_finder)

        visualizer.toggle_axis(2)
        visualizer.toggle_axis(3)

        self.assertEqual(visualizer.selected_axis_indices, [0, 1, 2, 3])
        self.assertEqual(visualizer.pages, [old_finder])
        self.assertTrue(torch.equal(visualizer.axes, axes[:2]))
        visualizer.build_finder.assert_not_called()

        visualizer.toggle_axis(0)

        self.assertEqual(visualizer.selected_axis_indices, [1, 2, 3])
        self.assertEqual(visualizer.pages, [old_finder])
        visualizer.build_finder.assert_not_called()

        visualizer.toggle_axis(1)

        self.assertEqual(visualizer.selected_axis_indices, [2, 3])
        self.assertTrue(torch.equal(visualizer.axes, axes[[2, 3]]))
        self.assertEqual(visualizer.pages, [new_finder])
        self.assertIsNone(visualizer._flow_cache)
        self.assertTrue(visualizer._flow_dirty)
        visualizer.build_finder.assert_called_once_with()

    def test_constructor_installs_axis_preferences_panel(self):
        rnn = torch.nn.RNN(2, 3, batch_first=True)
        axes = torch.eye(3)

        visualizer = FlowFieldVisualizer(rnn, axes=axes, axis_labels=["x", "y", "z"])

        self.assertIsInstance(visualizer.preferences_panel, AxesPreferencesPanel)
        self.assertTrue(torch.equal(visualizer.current_field().axes, axes[:2]))


if __name__ == "__main__":
    unittest.main()
