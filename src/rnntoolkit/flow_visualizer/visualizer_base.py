from typing import Any

import pygame
import pygame.gfxdraw as gfxdraw
import numpy as np
import math
import torch
from matplotlib import colormaps
from rnntoolkit.flow_visualizer.button import Button
from rnntoolkit.flow_visualizer.preferences import PreferencesPanel
from rnntoolkit.flow_fields.flow_field import FlowField

Point = tuple[float, float] | tuple[int, int]
Color = tuple[int, int, int]
FlowCache = dict[str, np.ndarray]

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


def draw_aa_line(
    surface: pygame.Surface,
    start: Point,
    end: Point,
    color: Color,
) -> None:
    """Draw an anti-aliased line from two point tuples with gfxdraw."""
    gfxdraw.line(
        surface,
        round(start[0]),
        round(start[1]),
        round(end[0]),
        round(end[1]),
        color,
    )


def fmt_num(v: float) -> str:
    """Format a tick value compactly for axis and bound labels."""
    if abs(v) < 1e-9:
        return "0"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    if s == "-0":
        return "0"
    return s


PREFERENCE = {
    "grid_points": {
        "min": 5,
        "max": 35,
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


class FlowFieldVisualizerBase:
    """Interactive two-dimensional viewer for an RNN's flow field.

    This class asks ``FlowFieldFinder`` to project hidden states and calculate
    motion, maintains the coordinate view during pan/zoom operations, and draws
    both the vector field and its controls. Expensive flow results are cached
    until navigation or a setting marks them as dirty.
    """

    def __init__(
        self,
        rnn: Any,
        num_points: int = 10,
        x_offset: int = 5,
        y_offset: int = 5,
        x_center: float = 0.0,
        y_center: float = 0.0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        flow_type: str = "nonlinear",
    ) -> None:
        """Create the window, prepare the supplied data, and build the UI."""
        # Pygame window and frame-rate controller.
        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE
        )
        self._layout_size = self.screen.get_size()
        pygame.display.set_caption("Flow Field Visualizer")
        self.clock = pygame.time.Clock()

        # User-adjustable rendering and navigation settings.
        self.preferences = {
            "grid_points": num_points,
            "scroll_speed": 1.0,
            "zoom_speed": 1.0,
            "arrow_size_mult": 1.8,
            "arrow_length_mode": "speed",
            "colormap": "viridis",
            "vector_colors": "black",
            "show_heatmap": "off",
            "show_contours": "off",
            "show_vectors": "on",
            "state_trajectory": "off",
            "trajectory_color": "black",
            "trajectory_thickness": 2,
            "state_marker": "star",
            "state_marker_size": 1.0,
        }
        self.small_grid = 10

        # Initial projection and grid configuration.
        self.num_points = num_points
        self.x_center = x_center
        self.y_center = y_center
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.view_span = 10.0

        # Establish the square canvas and its initial data-coordinate bounds.
        self._calculate_grid_area()
        self.x_bounds = [0.0, 10.0]
        self.y_bounds = [0.0, 10.0]
        self._update_bounds()

        # Model data consumed by FlowFieldFinder.
        self.rnn = rnn
        self.fit_states = fit_states
        self.axes = axes
        self.current_element_idx = 0

        # Each page stores a finder and the input width it expects.
        # Subclasses provide the finder implementation.
        self.pages = [self.build_finder()]

        self.current_page = 1
        self.total_pages = 1
        self.flow_type = flow_type

        if self.flow_type != "nonlinear" and self.flow_type != "linear":
            raise ValueError("Must be nonlinear or linear")

        # The vector field is computed lazily the first time it is drawn.
        self._flow_cache = None
        self._flow_dirty = True

        self._create_ui()
        self.running = True

        # State retained while the user pans with a left-mouse drag.
        self.dragging = False
        self.drag_last_pos = None

        # Page arrows advance once immediately, then repeat while held.
        self._page_hold_direction = 0
        self._page_hold_started_at = 0
        self._page_last_repeat_at = 0
        self._page_hold_delay_ms = 350
        self._page_repeat_interval_ms = 100

    # ----------- Must be overrided by user -----------------------

    def build_finder(self, *args: Any, **kwargs: Any) -> Any:
        """Construct and return the finder used by this visualizer."""
        raise NotImplementedError

    def prepare_data(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Convert user inputs/states into page-aligned visualizer data."""
        raise NotImplementedError

    def compute_flow_field(self, *args: Any, **kwargs: Any) -> FlowField:
        """Compute and cache the flow field for the current page."""
        raise NotImplementedError

    def current_field(self) -> Any:
        """Return metadata for the currently visible field page."""
        return self.pages[self.current_page - 1]

    def _mark_dirty(self) -> None:
        """Require the cached flow field to be recomputed before drawing."""
        self._flow_dirty = True

    def _step_page(self, direction: int) -> None:
        """Move one page in ``direction`` while respecting page bounds."""
        new_index = max(
            0,
            min(self.n_pages - 1, self.current_element_idx + direction),
        )
        if new_index != self.current_element_idx:
            self.current_element_idx = new_index
            self._mark_dirty()

    def _start_page_hold(self, direction: int) -> None:
        """Start delayed page repetition after an initial arrow press."""
        now = pygame.time.get_ticks()
        self._page_hold_direction = direction
        self._page_hold_started_at = now
        self._page_last_repeat_at = now

    def _stop_page_hold(self) -> None:
        """Stop page repetition for a released or abandoned arrow press."""
        self._page_hold_direction = 0

    def _repeat_held_page_arrow(self) -> None:
        """Advance pages at a fixed rate while an arrow remains held."""
        direction = self._page_hold_direction
        if not direction:
            return

        button = self.left_arrow_btn if direction < 0 else self.right_arrow_btn
        if not button.is_pressed or not button.rect.collidepoint(pygame.mouse.get_pos()):
            self._stop_page_hold()
            return

        now = pygame.time.get_ticks()
        if now - self._page_hold_started_at < self._page_hold_delay_ms:
            return
        if now - self._page_last_repeat_at < self._page_repeat_interval_ms:
            return

        self._step_page(direction)
        self._page_last_repeat_at = now

    # ----------- base class functionality -----------------------

    def _calculate_grid_area(self) -> None:
        """Center the square plot in the space between the two toolbars."""
        window_width, window_height = self.screen.get_size()
        side_margin = 80
        vertical_margin = 10

        left_bound = side_margin
        right_bound = window_width - side_margin
        top_bound = TOP_BAR_HEIGHT + vertical_margin
        bottom_bound = window_height - TOOLBAR_HEIGHT - vertical_margin

        available_width = max(1, right_bound - left_bound)
        available_height = max(1, bottom_bound - top_bound)
        size = min(available_width, available_height)
        left = (window_width - size) // 2
        top = top_bound + (available_height - size) // 2
        self.grid_area = pygame.Rect(left, top, size, size)

    def _sync_window_layout(self) -> None:
        """Update layout after SDL or the window manager changes the size.

        Pygame 2 resizes a ``RESIZABLE`` display surface automatically. The
        application must not call ``set_mode`` again from a resize event,
        because doing so can recreate the native window while a window manager
        owns the interactive-resize pointer grab. Polling the current display
        surface also covers programmatic window-manager changes that different
        platforms report through different SDL window events.
        """
        display_surface = pygame.display.get_surface()
        if display_surface is None:
            return

        self.screen = display_surface
        current_size = display_surface.get_size()
        if current_size == self._layout_size:
            return

        self._layout_size = current_size
        self._calculate_grid_area()
        self._position_toolbar_controls()

    def _create_ui(self) -> None:
        """Create all buttons and popup controls."""
        self.pref_btn = Button(14, 5, 105, 25, "Options", style="menu")
        self.preferences_panel = PreferencesPanel(self.pref_btn, self)

        toolbar_y = WINDOW_HEIGHT - TOOLBAR_HEIGHT
        self.mode_btn = Button(14, toolbar_y + 7, 100, 25, "Nonlinear", style="menu")

        right_x = WINDOW_WIDTH - 180
        self.left_arrow_btn = Button(right_x, toolbar_y + 7, 40, 25, "<", style="menu")
        self.page_label_rect = pygame.Rect(right_x + 50, toolbar_y + 7, 50, 25)
        self.right_arrow_btn = Button(
            right_x + 110, toolbar_y + 7, 40, 25, ">", style="menu"
        )

        self._position_toolbar_controls()

        self.all_buttons = [
            self.pref_btn,
            self.mode_btn,
            self.left_arrow_btn,
            self.right_arrow_btn,
        ]

    def _position_toolbar_controls(self) -> None:
        """Keep toolbar controls fixed-size while moving them with the window."""
        _, window_height = self.screen.get_size()
        toolbar_y = window_height - TOOLBAR_HEIGHT
        self.mode_btn.rect.topleft = (14, toolbar_y + 7)
        right_x = max(230, self.screen.get_width() - 180)
        self.left_arrow_btn.rect.topleft = (right_x, toolbar_y + 7)
        self.page_label_rect.topleft = (right_x + 50, toolbar_y + 7)
        self.right_arrow_btn.rect.topleft = (right_x + 110, toolbar_y + 7)

    def adjust_pref(self, key: str, direction: int) -> None:
        """Move a preference one step and clamp it to its allowed range."""
        spec = PREFERENCE[key]
        if "choices" in spec:
            choices = spec["choices"]
            current_index = choices.index(self.preferences[key])
            self.preferences[key] = choices[(current_index + direction) % len(choices)]
            return

        val = self.preferences[key] + direction * spec["step"]
        val = max(spec["min"], min(spec["max"], val))
        val = round(val) if spec["is_int"] else round(val, 3)
        self.preferences[key] = val
        if key == "grid_points":
            self._mark_dirty()

    def get_grid_step(self) -> float:
        """Choose a readable major-grid interval for the current zoom."""
        target_divisions = max(1, int(self.num_points - 1))
        rough_step = self.view_span / target_divisions

        # Snap the desired interval to a small set of visually useful steps.
        mag = 10.0 ** math.floor(math.log10(rough_step))
        rel_step = rough_step / mag

        if rel_step < 1.5:
            return 1.0 * mag
        elif rel_step < 2.75:
            return 2.0 * mag
        elif rel_step < 5.0:
            return 3.5 * mag
        elif rel_step < 8.0:
            return 6.5 * mag
        else:
            return 10.0 * mag

    def _update_bounds(self) -> None:
        """Derive visible x/y bounds from the center and square span."""
        self.x_bounds = (
            self.x_center - self.view_span / 2,
            self.x_center + self.view_span / 2,
        )
        self.y_bounds = (
            self.y_center - self.view_span / 2,
            self.y_center + self.view_span / 2,
        )

    def data_to_px(self, x_val: float, y_val: float) -> tuple[float, float]:
        """Convert data coordinates to Pygame pixel coordinates."""
        x_range = self.x_bounds[1] - self.x_bounds[0]
        y_range = self.y_bounds[1] - self.y_bounds[0]
        px = (
            self.grid_area.left
            + (x_val - self.x_bounds[0]) / x_range * self.grid_area.width
        )
        # Pixel y increases downward, so data-space y must be inverted.
        py = (
            self.grid_area.top
            + (self.y_bounds[1] - y_val) / y_range * self.grid_area.height
        )
        return px, py

    def px_to_data(self, px: float, py: float) -> tuple[float, float]:
        """Convert Pygame pixel coordinates back to data coordinates."""
        fx = (px - self.grid_area.left) / self.grid_area.width
        fy = (py - self.grid_area.top) / self.grid_area.height
        x_val = self.x_bounds[0] + fx * (self.x_bounds[1] - self.x_bounds[0])
        y_val = self.y_bounds[1] - fy * (self.y_bounds[1] - self.y_bounds[0])
        return x_val, y_val

    def zoom(self, amount: int, mouse_pos: tuple[int, int]) -> None:
        """Zoom while keeping the point beneath the cursor stationary."""
        if amount == 0:
            return

        mx, my = self.px_to_data(*mouse_pos)
        speed = self.preferences["zoom_speed"]

        factor = 1.0 + (0.10 * speed * abs(amount))

        if amount > 0:
            new_span = self.view_span / factor
        else:
            new_span = self.view_span * factor

        if new_span < 0.001 or new_span > 100000:
            return

        self.view_span = new_span

        fx = (mouse_pos[0] - self.grid_area.left) / self.grid_area.width
        fy = (mouse_pos[1] - self.grid_area.top) / self.grid_area.height

        # Recenter so the cursor still corresponds to the original data point.
        self.x_center = mx + new_span / 2 - fx * new_span
        self.y_center = my - new_span / 2 + fy * new_span
        self._mark_dirty()

    def handle_events(self) -> None:
        """Process one frame of window, control, zoom, and pan events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            pref_button_toggled = False
            any_button_consumed = False

            # Buttons get the event before it can start a canvas interaction.
            for button in self.all_buttons:
                if button.handle_event(event):
                    any_button_consumed = True
                    if button == self.pref_btn:
                        if self.preferences_panel.visible:
                            self.preferences_panel.hide()
                        else:
                            self.preferences_panel.show()
                        pref_button_toggled = True
                    elif button == self.mode_btn:
                        self.flow_type = (
                            "linear" if self.flow_type == "nonlinear" else "nonlinear"
                        )
                        self.mode_btn.text = (
                            "Linear" if self.flow_type == "linear" else "Nonlinear"
                        )
                        self._mark_dirty()
                    elif button == self.left_arrow_btn:
                        self._step_page(-1)
                        self._start_page_hold(-1)
                    elif button == self.right_arrow_btn:
                        self._step_page(1)
                        self._start_page_hold(1)

            if not pref_button_toggled:
                self.preferences_panel.handle_event(event)

            # Zoom only when a popup is not covering the canvas.
            if event.type == pygame.MOUSEWHEEL:
                if not self.preferences_panel.visible:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.grid_area.collidepoint(mouse_pos):
                        self.zoom(event.y, mouse_pos)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if (
                    not any_button_consumed
                    and not self.preferences_panel.visible
                    and self.grid_area.collidepoint(event.pos)
                ):
                    self.dragging = True
                    self.drag_last_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._stop_page_hold()
                self.dragging = False
                self.drag_last_pos = None

            elif event.type == pygame.MOUSEMOTION and self.dragging:
                # Convert pixel motion to a zoom-independent data-space offset.
                dx_px = event.pos[0] - self.drag_last_pos[0]
                dy_px = event.pos[1] - self.drag_last_pos[1]
                self.drag_last_pos = event.pos

                dx_data = (
                    -dx_px
                    * (self.view_span / self.grid_area.width)
                    * self.preferences["scroll_speed"]
                )
                dy_data = (
                    dy_px
                    * (self.view_span / self.grid_area.height)
                    * self.preferences["scroll_speed"]
                )

                self.x_center += dx_data
                self.y_center += dy_data
                self._mark_dirty()

        self._sync_window_layout()
        self._repeat_held_page_arrow()

        self._update_bounds()

    def draw_grid(self, inputs: torch.Tensor, states: torch.Tensor) -> None:
        """Draw grid lines, flow arrows, and arrowheads."""
        pygame.draw.rect(self.screen, WHITE, self.grid_area)

        # Prevent plot contents from spilling into the surrounding UI.
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(self.grid_area)

        step = self.get_grid_step()
        x_start_mult = math.floor(self.x_bounds[0] / step)
        x_end_mult = math.ceil(self.x_bounds[1] / step)
        y_start_mult = math.floor(self.y_bounds[0] / step)
        y_end_mult = math.ceil(self.y_bounds[1] / step)

        sub = self.small_grid
        sub_step = step / sub

        for mx in range(x_start_mult - 1, x_end_mult + 2):
            gx = mx * step
            px, _ = self.data_to_px(gx, 0)
            pygame.draw.line(
                self.screen,
                LIGHT_GRAY,
                (px, self.grid_area.top),
                (px, self.grid_area.bottom),
                1,
            )
            if mx <= x_end_mult:
                for k in range(1, sub):
                    px_minor, _ = self.data_to_px(gx + k * sub_step, 0)
                    pygame.draw.line(
                        self.screen,
                        (240, 240, 240),
                        (px_minor, self.grid_area.top),
                        (px_minor, self.grid_area.bottom),
                        1,
                    )

        for my in range(y_start_mult - 1, y_end_mult + 2):
            gy = my * step
            _, py = self.data_to_px(0, gy)
            pygame.draw.line(
                self.screen,
                LIGHT_GRAY,
                (self.grid_area.left, py),
                (self.grid_area.right, py),
                1,
            )
            if my <= y_end_mult:
                for k in range(1, sub):
                    _, py_minor = self.data_to_px(0, gy + k * sub_step)
                    pygame.draw.line(
                        self.screen,
                        (240, 240, 240),
                        (self.grid_area.left, py_minor),
                        (self.grid_area.right, py_minor),
                        1,
                    )

        # Flow computation is performed by the visualization loop before
        # rendering, so this method remains independent of RNN data formats.

        if (
            self.preferences["show_heatmap"] == "on"
            or self.preferences["show_contours"] == "on"
        ):
            self._draw_energy_landscape(
                self._flow_cache,
                draw_heatmap=self.preferences["show_heatmap"] == "on",
                draw_contours=self.preferences["show_contours"] == "on",
            )

        if self.preferences["show_vectors"] != "on":
            self._draw_state_trajectory(states)
            self._draw_state_marker(states)
            self.screen.set_clip(prev_clip)
            return

        unit_px = (
            abs(
                self.data_to_px(self.x_bounds[0] + step, 0)[0]
                - self.data_to_px(self.x_bounds[0], 0)[0]
            )
            or 1
        )
        arrow_scale = unit_px * 0.35 * self.preferences["arrow_size_mult"]

        vector_colors = self.preferences["vector_colors"]
        arrow_colormap = colormaps[vector_colors] if vector_colors != "black" else None

        cache = self._flow_cache
        grid = cache["grid"]
        x_vel = cache["x_vel"]
        y_vel = cache["y_vel"]
        speed = cache["speed"]

        # Normalize arrow lengths relative to the fastest visible vector.
        max_speed = float(np.max(speed)) if np.max(speed) > 0 else 1.0

        n_i, n_j = grid.shape[0], grid.shape[1]
        for i in range(n_i):
            for j in range(n_j):
                gx, gy = float(grid[i, j, 0]), float(grid[i, j, 1])
                px, py = self.data_to_px(gx, gy)

                if not (
                    -50 <= px - self.grid_area.left <= self.grid_area.width + 50
                    and -50 <= py - self.grid_area.top <= self.grid_area.height + 50
                ):
                    continue

                vx, vy = float(x_vel[i, j]), float(y_vel[i, j])
                mag = math.sqrt(vx * vx + vy * vy)
                s = float(speed[i, j])

                # Preserve direction while normalized speed controls length.
                if mag > 1e-9:
                    unit_dx, unit_dy = vx / mag, vy / mag
                else:
                    unit_dx, unit_dy = 0.0, 0.0

                length_frac = s / max_speed if max_speed > 0 else 0.0
                length_frac = max(0.0, min(1.0, length_frac))
                if self.preferences["arrow_length_mode"] == "equal":
                    arrow_len = arrow_scale if mag > 1e-9 else 0.0
                else:
                    arrow_len = arrow_scale * (0.3 + 0.7 * length_frac)
                if arrow_colormap is None:
                    arrow_color = (0, 0, 0)
                else:
                    rgba = arrow_colormap(length_frac)
                    arrow_color = tuple(round(channel * 255) for channel in rgba[:3])

                end_x = px + unit_dx * arrow_len
                end_y = py - unit_dy * arrow_len

                draw_aa_line(
                    self.screen,
                    (round(px), round(py)),
                    (round(end_x), round(end_y)),
                    arrow_color,
                )

                if mag > 1e-9:
                    angle = math.atan2(-unit_dy, unit_dx)
                    head_len = min(8.0, max(3.0, arrow_len * 0.55))
                    p1 = (
                        end_x - head_len * math.cos(angle - math.pi / 6),
                        end_y - head_len * math.sin(angle - math.pi / 6),
                    )
                    p2 = (
                        end_x - head_len * math.cos(angle + math.pi / 6),
                        end_y - head_len * math.sin(angle + math.pi / 6),
                    )
                    head = [
                        (round(p1[0]), round(p1[1])),
                        (round(end_x), round(end_y)),
                        (round(p2[0]), round(p2[1])),
                    ]
                    gfxdraw.filled_polygon(self.screen, head, arrow_color)
                    gfxdraw.aapolygon(self.screen, head, arrow_color)

        self._draw_state_trajectory(states)
        self._draw_state_marker(states)
        self.screen.set_clip(prev_clip)

    def _draw_state_trajectory(self, states_nxd: torch.Tensor | None) -> None:
        """Connect prior states when the input is a time-ordered trajectory."""
        if self.preferences["state_trajectory"] != "on":
            return
        if states_nxd is None or self.current_element_idx < 1:
            return

        finder = self.current_field()
        states = states_nxd[: self.current_element_idx + 1]
        reduced_states = finder._reduce_traj(states)
        points = [
            self.data_to_px(float(state[0]), float(state[1]))
            for state in reduced_states
        ]
        if len(points) < 2:
            return

        colors = {
            "black": (0, 0, 0),
            "red": (190, 40, 40),
            "blue": (40, 80, 190),
            "green": (40, 140, 70),
            "white": (255, 255, 255),
        }
        trajectory_color = colors[self.preferences["trajectory_color"]]
        thickness = self.preferences["trajectory_thickness"]
        for start, end in zip(points, points[1:]):
            start_px = (round(start[0]), round(start[1]))
            end_px = (round(end[0]), round(end[1]))
            pygame.draw.line(self.screen, trajectory_color, start_px, end_px, thickness)
            draw_aa_line(self.screen, start_px, end_px, trajectory_color)

    def _draw_state_marker(self, states_nxd: torch.Tensor | None) -> None:
        """Draw the state used for the current flow field over the plot."""
        if states_nxd is None or len(states_nxd) == 0:
            return

        state = states_nxd[self.current_element_idx]
        finder = self.current_field()
        reduced_state = finder._reduce_traj(state.unsqueeze(0))[0]
        px, py = self.data_to_px(float(reduced_state[0]), float(reduced_state[1]))
        marker = self.preferences["state_marker"]
        marker_scale = self.preferences["state_marker_size"]
        color = (20, 20, 20)
        highlight = (255, 255, 255)

        if not self.grid_area.inflate(16, 16).collidepoint(px, py):
            return

        if marker == "dot":
            pygame.draw.circle(
                self.screen, highlight, (round(px), round(py)), round(8 * marker_scale)
            )
            pygame.draw.circle(
                self.screen, color, (round(px), round(py)), round(5 * marker_scale)
            )
        elif marker == "x":
            pygame.draw.line(
                self.screen,
                highlight,
                (px - 8 * marker_scale, py - 8 * marker_scale),
                (px + 8 * marker_scale, py + 8 * marker_scale),
                max(1, round(4 * marker_scale)),
            )
            pygame.draw.line(
                self.screen,
                highlight,
                (px - 8 * marker_scale, py + 8 * marker_scale),
                (px + 8 * marker_scale, py - 8 * marker_scale),
                max(1, round(4 * marker_scale)),
            )
            pygame.draw.line(
                self.screen,
                color,
                (px - 7 * marker_scale, py - 7 * marker_scale),
                (px + 7 * marker_scale, py + 7 * marker_scale),
                max(1, round(2 * marker_scale)),
            )
            pygame.draw.line(
                self.screen,
                color,
                (px - 7 * marker_scale, py + 7 * marker_scale),
                (px + 7 * marker_scale, py - 7 * marker_scale),
                max(1, round(2 * marker_scale)),
            )
        else:
            points = []
            for i in range(10):
                angle = -math.pi / 2 + i * math.pi / 5
                radius = marker_scale * (10 if i % 2 == 0 else 4)
                points.append(
                    (px + radius * math.cos(angle), py + radius * math.sin(angle))
                )
            star_points = [(round(x), round(y)) for x, y in points]
            gfxdraw.filled_polygon(self.screen, star_points, highlight)
            gfxdraw.aapolygon(self.screen, star_points, highlight)
            inner_points = []
            for i in range(10):
                angle = -math.pi / 2 + i * math.pi / 5
                radius = marker_scale * (8 if i % 2 == 0 else 3)
                inner_points.append(
                    (px + radius * math.cos(angle), py + radius * math.sin(angle))
                )
            star_inner = [(round(x), round(y)) for x, y in inner_points]
            gfxdraw.filled_polygon(self.screen, star_inner, color)
            gfxdraw.aapolygon(self.screen, star_inner, color)

    def _draw_energy_landscape(
        self,
        cache: FlowCache,
        draw_heatmap: bool = True,
        draw_contours: bool = True,
    ) -> None:
        """Draw optional speed heatmap and equal-speed contours."""
        grid, speed = cache["grid"], cache["speed"]
        cmap = colormaps[self.preferences["colormap"]]
        # Match flow-field arrow coloring: zero maps to the bottom of the
        # colormap and the fastest visible vector maps to the top.
        max_speed = float(np.max(speed))
        rows, cols = speed.shape
        if draw_heatmap:
            for i in range(rows - 1):
                for j in range(cols - 1):
                    value = float(np.mean(speed[i : i + 2, j : j + 2]))
                    norm = value / max_speed if max_speed > 1e-12 else 0.0
                    norm = max(0.0, min(1.0, norm))
                    color = tuple(round(c * 255) for c in cmap(norm)[:3])
                    x0, y0 = self.data_to_px(grid[i, j, 0], grid[i, j, 1])
                    x1, y1 = self.data_to_px(
                        grid[i + 1, j + 1, 0], grid[i + 1, j + 1, 1]
                    )
                    rect = pygame.Rect(
                        round(min(x0, x1)),
                        round(min(y0, y1)),
                        # Slight overlap prevents rounded cell boundaries from
                        # exposing the background grid between heatmap cells.
                        max(1, round(abs(x1 - x0)) + 2),
                        max(1, round(abs(y1 - y0)) + 2),
                    )
                    pygame.draw.rect(self.screen, color, rect)

        if draw_contours and max_speed > 1e-12:
            import matplotlib.pyplot as plt

            contour = plt.contour(
                grid[:, :, 0],
                grid[:, :, 1],
                speed,
                levels=np.linspace(0.0, max_speed, 8),
            )
            # Newer Matplotlib versions expose all contour paths directly
            # on QuadContourSet rather than through ``collections``. A single
            # path can contain several disconnected subpaths, so split on
            # MOVETO/CLOSEPOLY to avoid drawing spurious straight connectors.
            from matplotlib.path import Path as MatplotlibPath

            for path in contour.get_paths():
                points = []
                for vertex, code in path.iter_segments():
                    if code == MatplotlibPath.MOVETO:
                        if len(points) > 1:
                            for start, end in zip(points, points[1:]):
                                draw_aa_line(
                                    self.screen,
                                    (round(start[0]), round(start[1])),
                                    (round(end[0]), round(end[1])),
                                    (35, 35, 35),
                                )
                        points = [self.data_to_px(float(vertex[0]), float(vertex[1]))]
                    elif code == MatplotlibPath.CLOSEPOLY:
                        if len(points) > 1:
                            for start, end in zip(points, points[1:]):
                                draw_aa_line(
                                    self.screen,
                                    (round(start[0]), round(start[1])),
                                    (round(end[0]), round(end[1])),
                                    (35, 35, 35),
                                )
                        points = []
                    else:
                        points.append(
                            self.data_to_px(float(vertex[0]), float(vertex[1]))
                        )
                if len(points) > 1:
                    for start, end in zip(points, points[1:]):
                        draw_aa_line(
                            self.screen,
                            (round(start[0]), round(start[1])),
                            (round(end[0]), round(end[1])),
                            (35, 35, 35),
                        )
            plt.close(contour.figure)

    def _snap_to_grid(self, value: float) -> float:
        """Round a coordinate to the nearest major-grid line."""
        step = self.get_grid_step()
        return round(value / step) * step

    def draw_axes(self) -> None:
        """Draw border axes, tick marks, and numeric labels."""
        font = pygame.font.SysFont("Arial", 15)
        x_axis_y = self.grid_area.bottom
        y_axis_x = self.grid_area.left

        pygame.draw.line(
            self.screen,
            BLACK,
            (self.grid_area.left, x_axis_y),
            (self.grid_area.right, x_axis_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            BLACK,
            (y_axis_x, self.grid_area.top),
            (y_axis_x, self.grid_area.bottom),
            2,
        )

        step = self.get_grid_step()
        x_start_mult = math.floor(self.x_bounds[0] / step)
        x_end_mult = math.ceil(self.x_bounds[1] / step)
        y_start_mult = math.floor(self.y_bounds[0] / step)
        y_end_mult = math.ceil(self.y_bounds[1] / step)

        for mx in range(x_start_mult, x_end_mult + 1):
            gx = mx * step
            px, _ = self.data_to_px(gx, 0)
            if self.grid_area.left - 1 <= px <= self.grid_area.right + 1:
                pygame.draw.line(
                    self.screen, BLACK, (px, x_axis_y), (px, x_axis_y + 8), 2
                )
                label = font.render(fmt_num(gx), True, BLACK)
                self.screen.blit(label, label.get_rect(center=(px, x_axis_y + 20)))

        for my in range(y_start_mult, y_end_mult + 1):
            gy = my * step
            _, py = self.data_to_px(0, gy)
            if self.grid_area.top - 1 <= py <= self.grid_area.bottom + 1:
                pygame.draw.line(
                    self.screen, BLACK, (y_axis_x, py), (y_axis_x - 8, py), 2
                )
                label = font.render(fmt_num(gy), True, BLACK)
                self.screen.blit(label, label.get_rect(center=(y_axis_x - 30, py)))

    def draw_bounds_text(self) -> None:
        """Show the visible coordinate bounds in the top bar."""
        label_font = pygame.font.SysFont("Arial", 15)
        bounds_text = (
            f"x: ( {fmt_num(self.x_bounds[0])} , {fmt_num(self.x_bounds[1])} )      \
        y: ( {fmt_num(self.y_bounds[0])} , {fmt_num(self.y_bounds[1])} )"
        )
        surface = label_font.render(bounds_text, True, DARK_GRAY)
        x = self.screen.get_width() - surface.get_width() - 20
        y = (TOP_BAR_HEIGHT - surface.get_height()) // 2
        self.screen.blit(surface, (x, y))

    def draw_top_bar(self) -> None:
        """Draw the top bar and its menus over the plot."""
        top_bar_rect = pygame.Rect(0, 0, self.screen.get_width(), TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, top_bar_rect)
        pygame.draw.line(
            self.screen,
            (218, 218, 223),
            (0, top_bar_rect.bottom),
            (self.screen.get_width(), top_bar_rect.bottom),
            1,
        )

        self.pref_btn.is_active = self.preferences_panel.visible
        self.pref_btn.draw(self.screen)

        self.draw_bounds_text()

        self.preferences_panel.draw(self.screen)

    def draw_toolbar(self) -> None:
        """Draw bottom controls and the page or sample indicator."""
        toolbar_rect = pygame.Rect(
            0,
            self.screen.get_height() - TOOLBAR_HEIGHT,
            self.screen.get_width(),
            TOOLBAR_HEIGHT,
        )
        pygame.draw.rect(self.screen, WHITE, toolbar_rect)
        pygame.draw.line(
            self.screen,
            (218, 218, 223),
            (0, toolbar_rect.top),
            (self.screen.get_width(), toolbar_rect.top),
            1,
        )

        self.mode_btn.draw(self.screen)
        self.left_arrow_btn.draw(self.screen)
        self.right_arrow_btn.draw(self.screen)

        page_font = pygame.font.SysFont("Arial", 14)
        page_text = page_font.render(
            f"{self.current_element_idx + 1}/{self.n_pages}",
            True,
            BLACK,
        )
        self.screen.blit(
            page_text, page_text.get_rect(center=self.page_label_rect.center)
        )

    # ----------- may likely be overrided by user -----------------------

    def visualize(self, *args: Any, **kwargs: Any) -> None:
        """Prepare arbitrary data and run the shared visualization loop.

        Subclasses with a different lifecycle or rendering data contract may
        override this method entirely. Otherwise, ``prepare_data`` adapts
        their inputs into the two page-aligned arrays consumed by the common
        drawing and interaction code.
        """
        prepared_data = self.prepare_data(*args, **kwargs)
        if not isinstance(prepared_data, tuple) or len(prepared_data) < 2:
            raise ValueError("prepare_data must return at least inputs and states")
        inp_nxd, states_nxd = prepared_data[:2]
        self.n_pages = inp_nxd.shape[0]
        if states_nxd.shape[0] != self.n_pages:
            raise ValueError("inputs and states must contain the same number of pages")

        while self.running:
            self.handle_events()
            if self._flow_dirty or self._flow_cache is None:
                self.compute_flow_field(*prepared_data)
            self.screen.fill(CANVAS_BG)
            self.draw_grid(inp_nxd, states_nxd)
            self.draw_axes()
            self.draw_top_bar()
            self.draw_toolbar()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        return
