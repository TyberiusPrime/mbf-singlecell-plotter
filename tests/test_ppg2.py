"""Pipegraph-level tests for PlotBuilder: job layout and change detection.

These exercise the part that needs a real graph — that recorded configuration
becomes files on disk, and that every recorded detail is covered by an
invariant.  The recorder itself is tested in test_ppg2_core.py.

Change detection is asserted through the filesystem (an output's mtime only
moves when its job re-ran) rather than through pipegraph internals, so these
stay valid across pypipegraph2 versions.
"""

import shutil
from pathlib import Path

import anndata
import matplotlib
import numpy as np
import pandas as pd
import plotnine as p9
import pytest

matplotlib.use("Agg")

ppg = pytest.importorskip("pypipegraph2")

from conftest import CELL_TYPE_COLUMN, EXAMPLE_DATA  # noqa: E402

from mbf_singlecell_plotter.ppg2 import Plot, PlotBuilder  # noqa: E402

RESULTS = "results"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A scratch directory that persists across several graphs in one test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def h5ad(workdir):
    """A private copy of the example data, so tests may modify it."""
    target = workdir / "analysis.h5ad"
    shutil.copy(EXAMPLE_DATA, target)
    return target


def run(build):
    """Create a fresh graph, let *build* populate it, and run it."""
    ppg.new()
    result = build()
    ppg.run()
    return result


def cheap_plot(builder, column=CELL_TYPE_COLUMN):
    """A histogram — the least expensive terminal to render repeatedly."""
    plot = builder.add(Plot(column))
    plot.histogram()
    return plot


def mtimes(*paths):
    return [p.stat().st_mtime_ns for p in paths]


# ---------------------------------------------------------------------------
# job layout
# ---------------------------------------------------------------------------


class TestJobLayout:
    def test_outputs_are_written(self, h5ad, workdir):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            plot = builder.add(Plot(CELL_TYPE_COLUMN))
            plot.histogram()
            return builder.register_all(RESULTS)

        run(build)
        assert (workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png").exists()

    def test_one_job_serves_all_outputs_of_a_plot(self, h5ad):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            plot = builder.add(Plot(CELL_TYPE_COLUMN))
            plot.histogram()
            plot.grid_histogram()
            return builder.register_all(RESULTS)

        assert len(run(build)) == 1

    def test_one_job_per_plot(self, h5ad):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            cheap_plot(builder)
            cheap_plot(builder, "n_genes")
            return builder.register_all(RESULTS)

        assert len(run(build)) == 2

    def test_several_outputs_land_side_by_side(self, h5ad, workdir):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            plot = builder.add(Plot(CELL_TYPE_COLUMN))
            plot.histogram()
            plot.grid_histogram()
            return builder.register_all(RESULTS)

        run(build)
        out = workdir / RESULTS
        assert (out / f"{CELL_TYPE_COLUMN}_histogram.png").exists()
        assert (out / f"{CELL_TYPE_COLUMN}_grid_histogram.png").exists()

    def test_into_nests_builder_then_plot(self, h5ad, workdir):
        def build():
            builder = PlotBuilder(into="genes")
            builder.set_source(h5ad, embedding="umap")
            plot = builder.add(Plot(CELL_TYPE_COLUMN, into="clusters"))
            plot.histogram()
            return builder.register_all(RESULTS)

        run(build)
        target = workdir / RESULTS / "genes" / "clusters"
        assert (target / f"{CELL_TYPE_COLUMN}_histogram.png").exists()

    def test_explicit_filename_is_honoured(self, h5ad, workdir):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            plot = builder.add(Plot(CELL_TYPE_COLUMN))
            plot.histogram(filename="clusters.png")
            return builder.register_all(RESULTS)

        run(build)
        assert (workdir / RESULTS / "clusters.png").exists()

    def test_a_plot_without_outputs_is_rejected(self, h5ad):
        ppg.new()
        builder = PlotBuilder()
        builder.set_source(h5ad, embedding="umap")
        builder.add(Plot(CELL_TYPE_COLUMN))
        with pytest.raises(ValueError, match="no outputs"):
            builder.register_all(RESULTS)

    def test_output_config_reaches_the_rendered_file(self, h5ad, workdir):
        """Output-scoped configuration must not be dropped on the way in."""

        def build(size):
            def inner():
                builder = PlotBuilder()
                builder.set_source(h5ad, embedding="umap")
                plot = builder.add(Plot(CELL_TYPE_COLUMN))
                plot.histogram().panel_size(size, size)
                return builder.register_all(RESULTS)

            return inner

        run(build(3))
        small = (workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png").read_bytes()
        run(build(6))
        large = (workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png").read_bytes()
        assert small != large


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


class TestInvariants:
    """Everything recorded must be covered; nothing else may cause churn."""

    def graph(self, h5ad, *, builder_size=(4, 4), plot_size=None, dot_size=1,
              filter_fn=None, extra_plot=False):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            builder.panel_size(*builder_size)
            builder.style(dot_size=dot_size)
            if filter_fn is not None:
                builder.set_filter(filter_fn)
            plot = builder.add(Plot(CELL_TYPE_COLUMN))
            if plot_size is not None:
                plot.panel_size(*plot_size)
            plot.histogram()
            if extra_plot:
                cheap_plot(builder, "n_genes")
            return builder.register_all(RESULTS)

        return build

    @pytest.fixture
    def target(self, workdir):
        return workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png"

    def test_an_unchanged_graph_does_not_rerun(self, h5ad, target):
        run(self.graph(h5ad))
        before = mtimes(target)
        run(self.graph(h5ad))
        assert mtimes(target) == before

    def test_a_builder_level_change_reruns(self, h5ad, target):
        """The hole in the old wrapper: builder defaults were never hashed."""
        run(self.graph(h5ad, builder_size=(4, 4)))
        before = mtimes(target)
        run(self.graph(h5ad, builder_size=(5, 5)))
        assert mtimes(target) != before

    def test_a_plot_level_change_reruns(self, h5ad, target):
        run(self.graph(h5ad))
        before = mtimes(target)
        run(self.graph(h5ad, plot_size=(6, 6)))
        assert mtimes(target) != before

    def test_a_changed_argument_value_reruns(self, h5ad, target):
        run(self.graph(h5ad, dot_size=1))
        before = mtimes(target)
        run(self.graph(h5ad, dot_size=3))
        assert mtimes(target) != before

    def test_a_changed_function_body_reruns(self, h5ad, target):
        """Callables are covered by FunctionInvariants, not by the JSON."""

        def keep_all(data):
            return np.ones(data.coordinates().shape[0], dtype=bool)

        def keep_all_but_differently(data):
            count = data.coordinates().shape[0]
            return np.array([True] * count)

        run(self.graph(h5ad, filter_fn=keep_all))
        before = mtimes(target)
        run(self.graph(h5ad, filter_fn=keep_all_but_differently))
        assert mtimes(target) != before

    def test_an_unrelated_plot_does_not_rerun_this_one(self, h5ad, target):
        run(self.graph(h5ad))
        before = mtimes(target)
        run(self.graph(h5ad, extra_plot=True))
        assert mtimes(target) == before

    def test_a_changed_input_file_reruns(self, h5ad, target):
        run(self.graph(h5ad))
        before = mtimes(target)

        modified = anndata.read_h5ad(h5ad)
        modified.obs[CELL_TYPE_COLUMN] = pd.Series(
            "0", index=modified.obs.index, dtype="category"
        )
        modified.write_h5ad(h5ad)

        run(self.graph(h5ad))
        assert mtimes(target) != before

    def test_an_output_level_change_reruns(self, h5ad, target):
        def graph(vertical):
            def build():
                builder = PlotBuilder()
                builder.set_source(h5ad, embedding="umap")
                plot = builder.add(Plot(CELL_TYPE_COLUMN))
                plot.histogram()
                plot.grid_histogram(vertical=vertical)
                return builder.register_all(RESULTS)

            return build

        run(graph(False))
        before = mtimes(target)
        run(graph(True))
        assert mtimes(target) != before


class TestDependencies:
    def test_a_source_job_is_waited_for(self, workdir):
        """(path, job) sources make the plot wait for the file's producer."""
        # relative: pypipegraph2 refuses absolute paths as job ids
        source = Path("generated.h5ad")

        def make_source(output_filename):
            shutil.copy(EXAMPLE_DATA, output_filename)

        def build():
            job = ppg.FileGeneratingJob(source, make_source)
            builder = PlotBuilder()
            builder.set_source((source, job), embedding="umap")
            cheap_plot(builder)
            return builder.register_all(RESULTS)

        run(build)
        assert (workdir / source).exists()
        assert (workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png").exists()

    def test_a_derived_source_carries_its_own_dependencies(self, h5ad, workdir):
        """A column callable may advertise upstream jobs via a .deps attribute."""
        marker = Path("marker.txt")  # relative, see test_a_source_job_is_waited_for

        def score(data):
            return data.get_column("n_genes").series * 2

        def build():
            # inside build(): ppg.new() has reset the graph by now
            score.deps = [
                ppg.FileGeneratingJob(marker, lambda of: of.write_text("done"))
            ]
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            builder.add_derived_source({"score": score})
            plot = builder.add(Plot("score"))
            plot.histogram()
            return builder.register_all(RESULTS)

        run(build)
        assert (workdir / marker).exists()
        assert (workdir / RESULTS / "score_histogram.png").exists()


class TestReachability:
    """Configuration the old wrapper could not express must now go through."""

    def test_a_previously_unreachable_method_is_recordable(self, h5ad, workdir):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            builder.background(enabled=True)  # never exposed by the old wrapper
            # a plotnine object, so the invariant encoder is exercised too
            builder.theme(plot_title=p9.element_text(size=14))
            cheap_plot(builder)
            return builder.register_all(RESULTS)

        run(build)
        assert (workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png").exists()

    def test_an_unwrapped_terminal_is_reachable(self, h5ad, workdir):
        def build():
            builder = PlotBuilder()
            builder.set_source(h5ad, embedding="umap")
            plot = builder.add(Plot(name="cells"))
            plot.density(bins=8)
            return builder.register_all(RESULTS)

        run(build)
        assert (workdir / RESULTS / "cells_density.png").exists()
