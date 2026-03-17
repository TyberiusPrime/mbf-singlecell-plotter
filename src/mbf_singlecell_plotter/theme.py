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


def embedding_theme(
    base_size, show_spines: bool = True, bg_color: str = "#FFFFFF"
) -> p9.theme:
    """Standard theme for embedding plots: no axis titles, clean ticks."""
    border = (
        p9.element_rect(color="#AAAAAA", size=0.5, fill=None)
        if show_spines
        else p9.element_blank()
    )
    return p9.theme_void(base_size=base_size) + p9.theme(
        axis_title_x=p9.element_blank(),
        axis_title_y=p9.element_blank(),
        axis_text=p9.element_text(
            color="#4d4d4d", margin={"t": 5, "r": 5, "units": "pt"}
        ),
        panel_border=border,
        plot_margin=0.02,
        # plot_background=p9.element_rect(fill=bg_color, color=None),
        # panel_background=p9.element_rect(fill=bg_color),
        # panel_border=border,
        # legend_background=p9.element_rect(fill=bg_color, color=bg_color, size=0.0),
        # legend_box_background=p9.element_blank(),
        # plot_margin=0.05,
    )
