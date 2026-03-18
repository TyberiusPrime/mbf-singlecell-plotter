"""Custom plotnine colorbar guide for mbf-singlecell-plotter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import plotnine as p9
from plotnine.guides.guide_colorbar import guide_colorbar


@dataclass
class sc_guide_colorbar(guide_colorbar):
    """
    Subclass of plotnine's guide_colorbar.

    - Uppercases the title.
    - Renders the title vertically (rotated 90°) to the left of the colorbar.
    - Draws rectangular extension boxes for zero (below) and/or clipped (above)
      values when the corresponding color is supplied.
    """

    zero_color: Optional[str] = None
    zero_label: str = "0"
    upper_clip_color: Optional[str] = None
    clip_label: str = ""

    def draw(self):
        if self.title:
            self.title = self.title.upper()

        # theme + … drops the `targets` attr (set by setup(), not a dataclass
        # field).  Save and restore it so super().draw() can still write to it.
        saved_targets = getattr(self.theme, "targets", None)

        # Target colorbar height ≈ 70 % of figure height (in pt).
        # Setting legend_key_height explicitly disables the inherited 7.25×
        # scale factor, so the value is used directly as the bar height in pt.
        fig_h_pt = self.theme.getp("figure_size")[1] * 72
        target_key_h = round(fig_h_pt * 0.70)

        self.theme = self.theme + p9.theme(
            legend_title_position="left",
            legend_title=p9.element_text(rotation=90, ha="center", va="center"),
            legend_key_height=target_key_h,
        )
        if saved_targets is not None:
            self.theme.targets = saved_targets

        # Recreate elements so cached properties pick up the updated theme values.
        self.elements = self._elements_cls(self.theme, self)

        # Drop boundary tick/label that would duplicate the extension box label.
        if self.zero_color is not None and len(self.key) > 1:
            self.key = self.key.iloc[1:].reset_index(drop=True)
        if self.upper_clip_color is not None and len(self.key) > 1:
            self.key = self.key.iloc[:-1].reset_index(drop=True)

        box = super().draw()

        if self.zero_color is not None or self.upper_clip_color is not None:
            self._add_extensions(box)

        return box

    def _add_extensions(self, box) -> None:
        """Append zero / clip rectangles + labels directly to the auxbox."""
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text

        # auxbox is always the last child (title precedes it for "left" position)
        auxbox = box.get_children()[-1]

        elements = self.elements
        bar_h = elements.key_height
        bar_w = elements.key_width + elements.frame.linewidth
        ext_h = bar_h * 0.10
        text_x = bar_w + elements.text.margin
        fontsize = elements.text.fontsize

        if self.zero_color is not None:
            auxbox.add_artist(
                Rectangle(
                    (0, -ext_h), bar_w, ext_h,
                    facecolor=self.zero_color, linewidth=0,
                )
            )
            auxbox.add_artist(
                Text(text_x, -ext_h / 2, self.zero_label,
                     fontsize=fontsize, va="center", ha="left")
            )

        if self.upper_clip_color is not None:
            auxbox.add_artist(
                Rectangle(
                    (0, bar_h), bar_w, ext_h,
                    facecolor=self.upper_clip_color, linewidth=0,
                )
            )
            auxbox.add_artist(
                Text(text_x, bar_h + ext_h / 2, self.clip_label,
                     fontsize=fontsize, va="center", ha="left")
            )
