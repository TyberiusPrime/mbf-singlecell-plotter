import plotnine as p9

DEFAULT_COLORS = [
    "#1C86EE",
    "#008B00",
    "#FF7F00",  # orange
    "#4D4D4D",
    "#FFD700",
    "#7EC0EE",
    "#FB9A99",  # lt pink
    "#FDBF6F",  # lt orange
    "#B03060",
    "#FF83FA",
    "#36648B",
    "#00CED1",
    "#00FF00",
    "#8B8B00",
    "#CDCD00",
    "#F52A2A",
]


def embedding_theme(show_spines: bool = True, bg_color: str = "#FFFFFF") -> p9.theme:
    """Standard theme for embedding plots: no axis titles, clean ticks."""
    border = (
        p9.element_rect(color="black", fill=None)
        if show_spines
        else p9.element_blank()
    )
    return p9.theme_void() + p9.theme(
        plot_background=p9.element_rect(fill=bg_color, color=None),
        panel_background=p9.element_rect(fill=bg_color),
        panel_border=border,
        legend_background=p9.element_rect(fill=bg_color, color=None),
    )
