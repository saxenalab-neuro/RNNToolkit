from typing import Any

import pygame
from rnntoolkit.flow_visualizer.button import Button

pygame.init()

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 650
TOP_BAR_HEIGHT = 35
TOOLBAR_HEIGHT = 40

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
MEDIUM_GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
CANVAS_BG = (245, 245, 250)

PREFERENCE = {
    "grid_points": {
        "min": 5,
        "max": 21,
        "step": 1,
        "fmt": lambda v: str(int(v)),
        "is_int": True,
    },
    "scroll_speed": {
        "min": 0.2,
        "max": 3.0,
        "step": 0.2,
        "fmt": lambda v: f"{v:.1f}x",
        "is_int": False,
    },
    "zoom_speed": {
        "min": 0.2,
        "max": 3.0,
        "step": 0.2,
        "fmt": lambda v: f"{v:.1f}x",
        "is_int": False,
    },
    "arrow_size_mult": {
        "min": 0.5,
        "max": 3.5,
        "step": 0.1,
        "fmt": lambda v: f"{v:.1f}x",
        "is_int": False,
    },
    "arrow_length_mode": {
        "choices": ("speed", "equal"),
        "fmt": str,
    },
    "colormap": {
        "choices": (
            "viridis",
            "cividis",
            "plasma",
            "magma",
            "inferno",
            "YlGnBu",
            "PuBuGn",
            "cubehelix",
            "Purples",
            "coolwarm",
        ),
        "fmt": str,
    },
    "vector_colors": {
        "choices": (
            "black",
            "viridis",
            "cividis",
            "plasma",
            "magma",
            "inferno",
            "YlGnBu",
            "PuBuGn",
            "cubehelix",
            "Purples",
            "coolwarm",
        ),
        "fmt": str,
    },
    "show_heatmap": {"choices": ("off", "on"), "fmt": str},
    "show_contours": {"choices": ("off", "on"), "fmt": str},
    "show_vectors": {"choices": ("on", "off"), "fmt": str},
    "state_trajectory": {"choices": ("off", "on"), "fmt": str},
    "trajectory_color": {
        "choices": ("black", "red", "blue", "green", "white"),
        "fmt": str,
    },
    "trajectory_thickness": {
        "min": 1,
        "max": 8,
        "step": 1,
        "fmt": lambda v: str(int(v)),
        "is_int": True,
    },
    "state_marker": {
        "choices": ("star", "dot", "x"),
        "fmt": str,
    },
    "state_marker_size": {
        "min": 0.5,
        "max": 3.5,
        "step": 0.1,
        "fmt": lambda v: f"{v:.1f}x",
        "is_int": False,
    },
}

PREF_LABELS = [
    ("grid_points", "Grid Points"),
    ("scroll_speed", "Scroll Speed"),
    ("zoom_speed", "Zoom Speed"),
    ("arrow_size_mult", "Arrow Size"),
    ("arrow_length_mode", "Arrow Length"),
    ("colormap", "Energy Colors"),
    ("vector_colors", "Vector Colors"),
    ("show_heatmap", "Energy Heatmap"),
    ("show_contours", "Energy Contours"),
    ("show_vectors", "Show Vectors"),
    ("state_trajectory", "State Trajectory"),
    ("trajectory_color", "Trajectory Color"),
    ("trajectory_thickness", "Trajectory Thickness"),
    ("state_marker", "State Marker"),
    ("state_marker_size", "Marker Size"),
]

FIELD_HIDDEN_SIZE = 8
FIELD_INPUT_SIZE = 4
FIELD_TRAJ_LEN = 200


class PreferencesPanel:
    """Popup controls for changing visualizer preferences.

    Every row has minus and plus buttons. The visualizer's ``adjust_pref``
    method owns value changes and range checking; this class handles the UI.
    """

    def __init__(self, options_button: Button, app: Any) -> None:
        """Initialize the panel anchored to ``options_button`` for ``app``."""
        self.file = options_button
        self.app = app
        self.visible = False
        self.label_font = pygame.font.SysFont("Arial", 14)
        self.value_font = pygame.font.SysFont("Arial", 13)
        self.btn_font = pygame.font.SysFont("Arial", 15)
        self.width = 340
        self.row_height = 36
        self.padding = 5

    def show(self) -> None:
        """Open the preferences panel."""
        self.visible = True

    def hide(self) -> None:
        """Close the preferences panel."""
        self.visible = False

    def _row_rect(self, i: int) -> pygame.Rect:
        """Return the rectangle occupied by preference row ``i``."""
        start_y = self.file.rect.bottom + 4 + self.padding
        return pygame.Rect(
            self.file.rect.left + self.padding,
            start_y + i * self.row_height,
            self.width - self.padding * 2,
            self.row_height,
        )

    def _menu_rect(self) -> pygame.Rect:
        """Return the rectangle enclosing the full preferences panel."""
        return pygame.Rect(
            self.file.rect.left,
            self.file.rect.bottom + 4,
            self.width,
            self.row_height * len(PREF_LABELS) + self.padding * 2,
        )

    def _stepper_rects(self, i: int) -> tuple[pygame.Rect, pygame.Rect]:
        """Return the minus and plus button rectangles for row ``i``."""
        row = self._row_rect(i)
        minus_rect = pygame.Rect(row.right - 92, row.top + 4, 26, 26)
        plus_rect = pygame.Rect(row.right - 34, row.top + 4, 26, 26)
        return minus_rect, plus_rect

    def _is_toggle(self, key: str) -> bool:
        """Return whether a choice is a simple two-state on/off setting."""
        return key in {
            "show_heatmap",
            "show_contours",
            "show_vectors",
            "state_trajectory",
        }

    def _value_rect(self, i: int) -> pygame.Rect:
        """Return the clickable value area for a toggle preference row."""
        row = self._row_rect(i)
        minus_rect, plus_rect = self._stepper_rects(i)
        return pygame.Rect(
            minus_rect.left,
            row.top + 4,
            plus_rect.right - minus_rect.left,
            26,
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        """Apply stepper clicks and close the panel on an outside click."""
        if not self.visible:
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, (key, label) in enumerate(PREF_LABELS):
                minus_rect, plus_rect = self._stepper_rects(i)
                if self._is_toggle(key):
                    if self._value_rect(i).collidepoint(event.pos):
                        self.app.adjust_pref(key, 1)
                        return
                elif minus_rect.collidepoint(event.pos):
                    self.app.adjust_pref(key, -1)
                    return
                elif plus_rect.collidepoint(event.pos):
                    self.app.adjust_pref(key, 1)
                    return
            if not self._menu_rect().collidepoint(event.pos):
                self.hide()

    def draw(self, screen: pygame.Surface) -> None:
        """Draw labels, current values, and stepper buttons."""
        if not self.visible:
            return
        menu_rect = self._menu_rect()
        pygame.draw.rect(screen, (205, 205, 210), menu_rect.move(0, 3), border_radius=8)
        pygame.draw.rect(screen, WHITE, menu_rect, border_radius=8)
        pygame.draw.rect(screen, (218, 218, 223), menu_rect, 1, border_radius=8)

        for i, (key, label) in enumerate(PREF_LABELS):
            row = self._row_rect(i)
            if i > 0:
                pygame.draw.line(
                    screen,
                    (235, 235, 238),
                    (row.left + 8, row.top),
                    (row.right - 8, row.top),
                    1,
                )

            label_surface = self.label_font.render(label, True, BLACK)
            screen.blit(
                label_surface,
                (row.left + 10, row.centery - label_surface.get_height() // 2),
            )

            minus_rect, plus_rect = self._stepper_rects(i)
            spec = PREFERENCE[key]
            value_str = spec["fmt"](self.app.preferences[key])

            # Controls use the same background as the panel and have no border.
            if self._is_toggle(key):
                value_rect = self._value_rect(i)
                value_surface = self.value_font.render(value_str, True, DARK_GRAY)
                screen.blit(
                    value_surface, value_surface.get_rect(center=value_rect.center)
                )
            else:
                pygame.draw.rect(screen, WHITE, minus_rect, border_radius=6)
                pygame.draw.rect(screen, WHITE, plus_rect, border_radius=6)
                step_labels = ("<", ">") if "choices" in spec else ("-", "+")
                left_text = self.btn_font.render(step_labels[0], True, BLACK)
                right_text = self.btn_font.render(step_labels[1], True, BLACK)
                screen.blit(left_text, left_text.get_rect(center=minus_rect.center))
                screen.blit(right_text, right_text.get_rect(center=plus_rect.center))
                value_rect = pygame.Rect(
                    minus_rect.right,
                    row.top,
                    plus_rect.left - minus_rect.right,
                    row.height,
                )
                value_surface = self.value_font.render(value_str, True, DARK_GRAY)
                screen.blit(
                    value_surface, value_surface.get_rect(center=value_rect.center)
                )
