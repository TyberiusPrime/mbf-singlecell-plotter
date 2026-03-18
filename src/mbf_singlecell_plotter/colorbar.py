"""Custom plotnine colorbar guide for mbf-singlecell-plotter."""

from __future__ import annotations

from plotnine.guides.guide_colorbar import guide_colorbar


class sc_guide_colorbar(guide_colorbar):
    """
    Subclass of plotnine's guide_colorbar for custom rendering.

    Starts as a minimal override: uppercases the title before drawing,
    then delegates entirely to the parent.
    """

    def draw(self):
        if self.title:
            self.title = self.title.upper()
        return super().draw()
