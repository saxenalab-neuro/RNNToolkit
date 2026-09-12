import unittest
from unittest.mock import Mock, patch

from rnntoolkit.flow_visualizer.visualizer_base import FlowFieldVisualizerBase


class TestVisualizerResize(unittest.TestCase):
    def make_visualizer(self):
        visualizer = object.__new__(FlowFieldVisualizerBase)
        visualizer.screen = Mock()
        visualizer._layout_size = (900, 650)
        visualizer._calculate_grid_area = Mock()
        visualizer._position_toolbar_controls = Mock()
        return visualizer

    def test_sync_uses_auto_resized_surface_without_resetting_display(self):
        visualizer = self.make_visualizer()
        resized_surface = Mock()
        resized_surface.get_size.return_value = (1200, 800)

        with (
            patch("pygame.display.get_surface", return_value=resized_surface),
            patch("pygame.display.set_mode") as set_mode,
        ):
            visualizer._sync_window_layout()

        self.assertIs(visualizer.screen, resized_surface)
        self.assertEqual(visualizer._layout_size, (1200, 800))
        visualizer._calculate_grid_area.assert_called_once_with()
        visualizer._position_toolbar_controls.assert_called_once_with()
        set_mode.assert_not_called()

    def test_sync_does_not_relayout_when_size_is_unchanged(self):
        visualizer = self.make_visualizer()
        current_surface = Mock()
        current_surface.get_size.return_value = visualizer._layout_size

        with patch("pygame.display.get_surface", return_value=current_surface):
            visualizer._sync_window_layout()

        self.assertIs(visualizer.screen, current_surface)
        visualizer._calculate_grid_area.assert_not_called()
        visualizer._position_toolbar_controls.assert_not_called()

    def test_sync_ignores_an_uninitialized_display(self):
        visualizer = self.make_visualizer()
        original_surface = visualizer.screen

        with patch("pygame.display.get_surface", return_value=None):
            visualizer._sync_window_layout()

        self.assertIs(visualizer.screen, original_surface)
        visualizer._calculate_grid_area.assert_not_called()
        visualizer._position_toolbar_controls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
