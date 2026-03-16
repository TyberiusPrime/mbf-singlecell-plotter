# Refactor Plan: mbf-singlecell-plotter

## Goals

1. **100% plotnine** — remove all raw matplotlib scatter/axis/legend/colorbar
   code and express everything as plotnine ggplot objects.
2. **Builder pattern** — replace the 35+ keyword arguments on `plot_scatter`
   with a composable builder where you configure the *plotter* once and then
   call `.plot(gene)` many times with different genes/columns.
   Each call gives you back a new object!
3. **Separate concerns** — split the single 1750-line file into clear layers:
   data access, plot-data transforms, and final rendering.

---

## Current Pain Points

| Problem | Where |
|---|---|
| `plot_scatter` has 35 kwargs, `_plot_scatter_core` has 30+ positional args | `__init__.py:1226–1270`, `:705–746` |
| `plot_scatter` always creates its own figure (ignores `fig`/`ax` args) | `:1380–1386` |
| `_add_legends` is 270 lines of matplotlib plumbing with dead code & commented-out blocks | `:309–580` |
| `edge_grid` branch unreachable (guarded by `draw_grid or grid_axes` check) | `:763,793` |
| `prep_boundaries` does 120 lines of pixel-level image processing inline | `:178–294` |
| PCA tuple embedding crashes constructor (`"X_" + tuple`) | `:88` |
| `plot_cell_density` duplicates tick/spine/colorbar logic from scatter | `:582–680` |
| `default` sentinel used as default arg value — fragile, hides issues | throughout |
| Grid label / `point_to_grid` bugs (clamped as hotfix but semantics unclear) | `:682–703` |

---

## Proposed Architecture

```
src/mbf_singlecell_plotter/
├── __init__.py          # Public re-exports only
├── data.py              # Layer 1: data access (AnnData → DataFrames)
├── transforms.py        # Layer 2: plot-data transforms (boundaries, density, grid, etc.)
├── plots.py             # Layer 3: plotnine plot builders
├── theme.py             # Shared plotnine theme, color palette, constants
└── util.py              # map_to_integers, unmap, etc.
```

### Layer 1 — `data.py`: `EmbeddingData`

Wraps an AnnData + embedding choice. Pure data extraction, no plotting.

```python
class EmbeddingData:
    def __init__(self, ad, embedding, alternative_id_column=None):
        ...

    # --- data accessors (existing logic, cleaned up) ---
    def get_column(self, name) -> tuple[pd.Series, str]: ... # also looks in the alternative_id_column
    def coordinates(self) -> pd.DataFrame:              ...  # x, y cols

    # --- grid helpers (pure math, no plotting) ---
    def grid_coordinate(self, x, y) -> str:              ...
    def grid_coordinates(self) -> pd.Series:             ... # must be fast, not just a loop around grid_coordinate
    def cluster_centers(self, cluster_column) -> pd.DataFrame:   ...
    def grid_local_histogram(self, key, ...) -> tuple:   ...
```

The PCA tuple vs. string embedding resolution moves here, with the
`isinstance` check done *before* string concatenation.

### Layer 2 — `transforms/*.py`: plot-data preparation

Functions (not methods) that take DataFrames and return DataFrames ready
for plotnine.  Each one has a single, well-defined job.

```python
def prepare_scatter_df(
    data: EmbeddingData,
    gene: str,
    clip_quantile: float = 0.95,
    zero_value: float = 0.0,
) -> pd.DataFrame:
    """Returns df with columns: x, y, expression, is_zero, is_clipped.
    Categorical expression is left as-is; numerical is clipped and
    annotated with is_zero / is_clipped flags."""

def prepare_density_df(
    data: EmbeddingData,
    bins: int = 200,
) -> pd.DataFrame:
    """2D histogram → long-form df with x, y, density columns."""

def compute_boundaries(
    data: EmbeddingData,
    resolution: int = 200,
    blur: float = 1.1,
    threshold: float = 0.95,
    colors: list[str] | None = None,
) -> pd.DataFrame:
    """Returns df with x, y, color columns for boundary scatter points.
    Requires scikit-image."""

(facetting will be delegated to plotnine).
 ).
```

### Layer 3 — `plots.py`: plotnine builders

The key insight: **you configure the plotter once, then call `.plot(gene)`
many times**.  The gene/column is the thing that varies; dot size, grid,
borders, colormap are the things you set up (once) for a batch of figures.

```python
class ScatterPlotter:
    """Reusable builder for embedding scatter plots.

    Configure once, plot many genes:

        plotter = (
            ScatterPlotter()
            .set_source(ad, embedding="umap")
            .dot_size(5)
            .with_borders(column_for_borders)
            .with_grid(labels=False, coords=True, vertical_letters=True)
            .focus_on(x_min, x_max, y_min, y_max)
            .colormap(cm.Reds, max_quantile=0.95,
                      zeros_behind=True, zero_color="darkgrey",
                      title="M2 score")
            .with_border_colormap(cmap_or_dict_or_list_of_colors)
        )

        # Then plot many genes with the same settings
        plotter.plot("S100A8")
        plotter.plot("CST3")
        plotter.plot("leiden")   # categorical auto-detected
    """

    def __init__(self):
        # --- source (set via .set_source()) ---
        self._data: EmbeddingData | None = None

        # --- dot appearance ---
        self._dot_size: float = 1
        self._show_spines: bool = True
        self._bg_color: str = "#FFFFFF"
        self._anti_overplot: bool = True
        self._flip_order: bool = False
        self._categorical_outliers: bool = True
        self._categorical_outlier_quantile: float = 0.95

        # --- colormap config (applied to numerical data) ---
        self._cmap = None               # None → default blue-magenta
        self._max_quantile: float = 0.95
        self._upper_clip_color: str = "#FF0000"
        self._cbar_title: str | None = None   # None → auto from gene name

        # --- zero handling ---
        self._zeros_behind: bool = True  # True = grey underlay behind data
        self._zero_color: str = "#D0D0D0"
        self._zero_dot_size: float = 5
        self._zero_marker: str = "."

        # --- categorical colormap ---
        self._cat_colors: list[str] | None = None  # None → DEFAULT_COLORS

        # --- optional layers ---
        self._borders: BorderConfig | None = None
        self._grid: GridConfig | None = None
        self._focus: tuple[float,float,float,float] | None = None  # x_min,x_max,y_min,y_max
        self._facet_variable: str | None = None
        self._n_col: int = 2
        self._title_override: str | None = None

    # ── source ──────────────────────────────────────────────
    def set_source(self, ad, embedding="umap") -> "ScatterPlotter":
        """Attach the AnnData and embedding. Can be swapped between plots."""
        self._data = EmbeddingData(ad, embedding)
        return self

    # ── dot appearance ──────────────────────────────────────
    def dot_size(self, s: float) -> "ScatterPlotter": ...
    def no_spines(self) -> "ScatterPlotter": ...
    def spines(self) -> "ScatterPlotter": ...
    def background(self, color: str) -> "ScatterPlotter": ...
    def flip(self) -> "ScatterPlotter": ...
    def no_outlier_replot(self) -> "ScatterPlotter": ...

    # ── colormap (numerical) ────────────────────────────────
    def colormap(self, cmap=None, *,
                 max_quantile: float = 0.95,
                 upper_clip_color: str = "#FF0000",
                 title: str | None = None
                 ) -> "ScatterPlotter":
        """Configure the continuous colormap and zero handling."""
        ...

    # ── categorical colors ──────────────────────────────────
    def colormap_discrete(self, cmap_or_dict_or_list_of_colors):
        """Configure a discrete color map for categorical data"""
            ...

    # ── zero decisios ──────────────────────────────────
    def zeros(self, 

                 behind: bool = True,
                 color: str = "#D0D0D0",
                 dot_size: float | None = None,
              ,) -> "ScatterPlotter":
        """Configure how to plot zeros. As normal points, 'behind' the 
        non-zero values"""
        ...


    # ── borders ─────────────────────────────────────────────
    def with_borders(self, *,
                     size: float = 15,
                     resolution: int = 200,
                     blur: float = 1.1,
                     threshold: float = 0.95) -> "ScatterPlotter": ...
    def without_borders(self) -> "ScatterPlotter": ... # to turn it back off.

    # ── grid overlay ────────────────────────────────────────
    def with_grid(self, *,
                  labels: bool = False,
                  coords: bool = False,
                  vertical_letters: bool = False,
                  grid_size: int = 12,
                  color: str = "#777777") -> "ScatterPlotter":
        """
        labels:   show "A1", "B2" labels in each cell
        coords:   use grid coordinates as axis tick labels
        """
        ...

    def without_grid(self) -> "ScatterPlotter": ...

    # ── viewport ────────────────────────────────────────────
    def focus_on(self, x_min, x_max, y_min, y_max) -> "ScatterPlotter":
        """Zoom into a region of the embedding."""
        ...

    def unfocus(self) -> "ScatterPlotter": ...

    # ── faceting ────────────────────────────────────────────
    def facet(self, variable: str, n_col: int = 2) -> "ScatterPlotter": ...
    def unfacet(self) -> "ScatterPlotter": ...

    # ── title ───────────────────────────────────────────────
    def title(self, t: str) -> "ScatterPlotter": ...

    # ── terminal ────────────────────────────────────────────
    def plot(self, gene: str) -> plotnine.ggplot:
        """Build and return a plotnine ggplot for the given gene/column.

        Numerical vs categorical is auto-detected.
        Returns a plotnine.ggplot that can be displayed, saved, or
        further composed with + and / operators.
        """
        ...

    def render(self, gene: str, path: str, **kwargs):
        """Shortcut: plot + save."""
        self.plot(gene).save(path, **kwargs)


@dataclass(frozen=True)
class BorderConfig:
    size: float = 15
    resolution: int = 200
    blur: float = 1.1
    threshold: float = 0.95

@dataclass(frozen=True)
class GridConfig:
    labels: bool = False
    coords: bool = False
    vertical_letters: bool = False
    grid_size: int = 12
    color: str = "#777777"
```

Density and grid-histogram branch off just before .plot

```python

def plot_density(bins):
     ... # reuses the continuous color map 

def plot_grid_histogram(column):
    ... # reuses the discrete color map.
```

### `theme.py`

```python
DEFAULT_COLORS = [...]

def embedding_theme(show_spines=True) -> plotnine.theme:
    """Standard theme for embedding plots: no axis titles, clean ticks."""
    ...
```

### `util.py`

`map_to_integers`, `unmap` — unchanged, kept as pure functions.

---

## plotnine Migration Notes

### Numerical scatter (currently `ax.scatter(..., c=, cmap=, vmin=, vmax=)`)

generally, don't bother with setting limits on the axis,
cut down the data before plotnine sees it.

plotnine equivalent:

```python
(ggplot(df, aes("x", "y", color="expression"))
 + geom_point(size=dot_size)
 + scale_color_cmap(cmap_name, limits=(vmin, vmax))
)
```

For the zero-value underlay, add a second `geom_point` layer with
`data=df_zeros` *before* the main one.

For the clipped-overplot outlier redraw (categorical), add a filtered
`geom_point` layer *after* the main one.

### Boundaries

Currently rendered as a raw `ax.scatter(bdf["x"], bdf["y"], color=bdf["color"])`.
In plotnine this becomes:

```python
geom_point(data=boundary_df, mapping=aes("x", "y", color="color"),
           size=border_size, inherit_aes=False)
+ scale_color_identity()
```

Since boundary colors are pre-computed RGBA tuples, `scale_color_identity()`
tells plotnine to use them as-is.

### Grid lines and labels

Currently done with `ax.axvline`/`ax.axhline`/`ax.text`.
plotnine equivalents:

```python
geom_vline(data=grid_df, mapping=aes(xintercept="x"), ...)
geom_hline(data=grid_df, mapping=aes(yintercept="y"), ...)
geom_text(data=labels_df, mapping=aes("x", "y", label="label"), ...)
```

### Legends

plotnine handles legends automatically from scales.
- Categorical: `scale_color_manual(values=...)` produces the legend.
- Numerical: `scale_color_cmap(...)` produces the colorbar.
- Borders: `scale_color_identity(guide=False)` hides the identity scale
  from the legend.

The entire 270-line `_add_legends` method disappears.

### Cell density

Currently `ax.hist2d` → replace with `prepare_density_df` +
`geom_tile(aes(fill="density"))` + `scale_fill_cmap(...)`.

### Faceting

`facet_wrap("~facet_column", ncol=n_col)` replaces the manual
subplot loop entirely (lines 1362–1470).

---

## Migration Strategy

### Phase 1 — Extract and test data layer
1. Create `data.py` with `EmbeddingData` class.
2. Move all data accessor methods from `ScanpyPlotter`.
3. Fix the PCA tuple bug in the constructor.
4. Update existing tests to import from new location.
5. Keep `ScanpyPlotter` as a thin wrapper that delegates to `EmbeddingData`
   (backwards compat during transition).
6. Run existing test suite — everything must still pass.

### Phase 2 — Extract transforms
1. Create `transforms.py` with `prepare_scatter_df`, `prepare_density_df`,
   `compute_boundaries`.
2. These are pure functions: DataFrame in, DataFrame out.
3. Write unit tests for each transform (shape, column names, edge cases).
4. `compute_boundaries` gets its own tests gated on `pytest.importorskip("skimage")`.

### Phase 3 — Build plotnine scatter
1. Create `plots.py` with `ScatterPlotter` builder.
2. Implement `.plot()` for numerical data first (simplest case).
3. Add categorical support.
4. Add each optional layer one at a time: zeros, borders, grid, facet.
5. After each addition, generate new reference images and compare visually
   (minor rendering differences expected between matplotlib and plotnine).
6. Update `test_images.py` to use the new API.

### Phase 4 — Density and grid histogram
1. Implement `DensityPlotter` and `GridHistogramPlotter`.
2. Generate new references, compare.

### Phase 5 — Cleanup
1. Remove `ScanpyPlotter` and all raw matplotlib code.
2. Remove `matplotlib` imports.
3. Update `__init__.py` to re-export the new public API.
4. Drop `barcode_rank_plot_umis` or port it (it depends on `dppd` which is
   not in dependencies — decide whether to keep it).
5. Final full test run + visual review of all reference images.

---

## Public API After Refactor

```python
from mbf_singlecell_plotter import (
    EmbeddingData, ScatterPlotter, DensityPlotter, GridHistogramPlotter,
    DEFAULT_COLORS,
)

# Configure once
sp = (
    ScatterPlotter()
    .set_source(ad, embedding="umap", cell_type_column="leiden")
    .dot_size(5)
    .with_borders()
    .with_grid(labels=False, coords=True, vertical_letters=True)
    .colormap(cm.Reds, max_quantile=0.95,title="M2 score")
    .zeros(behind=True, color='darkgrey')
)

# Plot many genes with the same settings
sp.plot("S100A8")
sp.plot("CST3")
sp.plot("leiden")         # categorical auto-detected, uses cat_colors

# Zoom in, change colormap for a specific plot
sp.focus_on(-5, 0, -5, 0).colormap(cm.Blues).plot("CD79A")

# Override just one setting for a single plot, then restore
sp.unfocus().colormap(cm.Reds).plot("TYROBP")

# Cell density
dp = (
    sp.with_borders().plot_density(bins=100)
)
dp.plot()

# Grid local histogram
ghp = (
    sp.plot_grid(min_cell_count=10)
)
ghp.plot("leiden")

# Data queries (no plotting)
data = EmbeddingData(ad, "umap")
centers = data.cluster_centers("leiden")
coords = data.grid_coordinates()
```

---

## Risks and Decisions

1. **plotnine colorbar for numerical scatter** — plotnine's `scale_color_cmap`
   generates a colorbar legend automatically, but fine control (extend arrows,
   custom tick formatter with ">X.XX" labels) may need `guide_colorbar()`
   customization.  Needs a spike to confirm plotnine can do this; 
   A simpler labeling scheme is not acceptable, we need to highlight
   the extra quantiles.

2. **Boundary rendering performance** — the boundary DataFrame can have
   ~100k+ points.  `geom_point` with `scale_color_identity()` should handle
   this fine, but worth benchmarking.

3. **Visual fidelity** — switching rendering backends means reference images
   will ALL change.  Plan a single "regenerate all references" step at the
   end of phase 3 and do a visual review with `dev/review-image-changes.py`.

4. **`barcode_rank_plot_umis`** — drop this for now..

5. **Backwards compatibility** —  No backwards compatibility necessary.

6. **Builder mutability** — the builder is immutable by design (`.focus_on()`
    gives back a new object!).  This is intentional:
   you configure once and tweak between plots.  
