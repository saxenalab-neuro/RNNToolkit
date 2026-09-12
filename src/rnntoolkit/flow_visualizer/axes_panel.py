"""Axis-selection controls for the interactive flow-field visualizer."""

from typing import Any

import pygame
from rnntoolkit.flow_visualizer.preferences import (
    BLACK,
    DARK_GRAY,
    PREF_LABELS,
    WHITE,
    PreferencesPanel,
)

AXES_LABEL = "Axes"


class AxesPreferencesPanel(PreferencesPanel):
    """Preferences panel with a nested submenu for selecting two axes."""

    def __init__(self, options_button: Any, app: Any) -> None:
        """Create an axis-aware preferences panel for ``app``."""
        super().__init__(options_button, app)
        self.axes_visible = False
        self.axes_width = 240

    def hide(self) -> None:
        """Close both the preferences menu and nested axes submenu."""
        super().hide()
        self.axes_visible = False

    def _row_rect(self, i: int) -> pygame.Rect:
        """Shift standard preference rows below the custom Axes row."""
        return super()._row_rect(i + 1)

    def _menu_rect(self) -> pygame.Rect:
        """Return the options rectangle including the custom Axes row."""
        return pygame.Rect(
            self.file.rect.left,
            self.file.rect.bottom + 4,
            self.width,
            self.row_height * (len(PREF_LABELS) + 1) + self.padding * 2,
        )

    def _axes_row_rect(self) -> pygame.Rect:
        """Return the top-level row that opens the axes submenu."""
        return super()._row_rect(0)

    def _axes_menu_rect(self) -> pygame.Rect:
        """Return the side-menu rectangle containing all available axes."""
        row = self._axes_row_rect()
        height = self.row_height * len(self.app.axis_labels) + self.padding * 2
        return pygame.Rect(row.right + 4, row.top, self.axes_width, height)

    def _axis_row_rect(self, index: int) -> pygame.Rect:
        """Return the row rectangle for one available axis."""
        menu = self._axes_menu_rect()
        return pygame.Rect(
            menu.left + self.padding,
            menu.top + self.padding + index * self.row_height,
            menu.width - self.padding * 2,
            self.row_height,
        )

    def _axis_toggle_rect(self, index: int) -> pygame.Rect:
        """Return the clickable toggle rectangle for one axis."""
        row = self._axis_row_rect(index)
        return pygame.Rect(row.right - 58, row.top + 5, 48, 26)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle axis submenu clicks before standard preference controls."""
        if not self.visible:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.axes_visible:
                for index in range(len(self.app.axis_labels)):
                    if self._axis_toggle_rect(index).collidepoint(event.pos):
                        self.app.toggle_axis(index)
                        return

            if self._axes_row_rect().collidepoint(event.pos):
                if self.app.axis_labels:
                    self.axes_visible = not self.axes_visible
                return

            axes_rect = self._axes_menu_rect() if self.axes_visible else None
            if axes_rect is not None and axes_rect.collidepoint(event.pos):
                return
            if not self._menu_rect().collidepoint(event.pos):
                self.hide()
                return

        super().handle_event(event)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the standard options, Axes row, and nested toggle list."""
        if not self.visible:
            return

        super().draw(screen)
        row = self._axes_row_rect()
        pygame.draw.line(
            screen,
            (235, 235, 238),
            (row.left + 8, row.top),
            (row.right - 8, row.top),
            1,
        )
        label_surface = self.label_font.render(AXES_LABEL, True, BLACK)
        screen.blit(
            label_surface,
            (row.left + 10, row.centery - label_surface.get_height() // 2),
        )

        if self.app.axis_labels:
            summary = (
                f"{len(self.app.selected_axis_indices)}/"
                f"{len(self.app.axis_labels)}"
            )
            arrow = ">"
        else:
            summary = "PCA"
            arrow = ""
        value_surface = self.value_font.render(summary, True, DARK_GRAY)
        screen.blit(
            value_surface, value_surface.get_rect(center=(row.right - 62, row.centery))
        )
        arrow_surface = self.btn_font.render(arrow, True, BLACK)
        screen.blit(
            arrow_surface, arrow_surface.get_rect(center=(row.right - 20, row.centery))
        )

        if not self.axes_visible:
            return

        menu = self._axes_menu_rect()
        pygame.draw.rect(screen, (205, 205, 210), menu.move(0, 3), border_radius=8)
        pygame.draw.rect(screen, WHITE, menu, border_radius=8)
        pygame.draw.rect(screen, (218, 218, 223), menu, 1, border_radius=8)

        for index, axis_label in enumerate(self.app.axis_labels):
            axis_row = self._axis_row_rect(index)
            if index > 0:
                pygame.draw.line(
                    screen,
                    (235, 235, 238),
                    (axis_row.left + 8, axis_row.top),
                    (axis_row.right - 8, axis_row.top),
                    1,
                )
            text = self.label_font.render(axis_label, True, BLACK)
            screen.blit(
                text,
                (axis_row.left + 10, axis_row.centery - text.get_height() // 2),
            )

            toggle = self._axis_toggle_rect(index)
            is_on = index in self.app.selected_axis_indices
            fill = (92, 150, 96) if is_on else (190, 190, 195)
            knob_x = toggle.right - 13 if is_on else toggle.left + 13
            pygame.draw.rect(screen, fill, toggle, border_radius=13)
            pygame.draw.circle(screen, WHITE, (knob_x, toggle.centery), 10)
            state = self.value_font.render("on" if is_on else "off", True, DARK_GRAY)
            screen.blit(state, state.get_rect(midright=(toggle.left - 8, toggle.centery)))
