"""Pipegraph-level tests for PlotBuilder: job creation, layout, invariants.

These exercise the part that needs a real graph -- that recorded configuration
becomes jobs and files on disk, and that every recorded detail is covered by an
invariant.  The recorder itself is tested in test_ppg2_core.py.

Change detection is asserted through the filesystem (an output's mtime only
moves when its job re-ran) rather than through pipegraph internals, so these
stay valid across pypipegraph2 versions.
"""

import shutil
import warnings
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

from mbf_singlecell_plotter.ppg2 import PlotBuilder  # noqa: E402

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


@pytest.fixture
def graph(workdir):
    """An empty graph, for tests that only create jobs and never run them."""
    # explicit: ppg.new() otherwise reuses the previous test's run mode
    ppg.new(run_mode=ppg.RunMode.CONSOLE)


def run(build):
    """Create a fresh graph, let *build* populate it, and run it."""
    ppg.new(run_mode=ppg.RunMode.CONSOLE)
    result = build()
    ppg.run()
    return result


def source(h5ad, **kwargs) -> PlotBuilder:
    return PlotBuilder(output_folder=RESULTS, **kwargs).set_source(
        h5ad, embedding="umap"
    )


def mtimes(*paths):
    return [p.stat().st_mtime_ns for p in paths]


def names(*plots):
    """The output file names of the jobs these plots created."""
    return [Path(job.job_id).name for plot in plots for job in plot.jobs_]


# ---------------------------------------------------------------------------
# job creation
# ---------------------------------------------------------------------------


class TestJobCreation:
    def test_a_terminal_creates_its_job_immediately(self, h5ad, graph):
        plot = source(h5ad).plot(CELL_TYPE_COLUMN).histogram()
        assert len(plot.jobs_) == 1
        assert plot.job_.job_id.endswith(f"{CELL_TYPE_COLUMN}_histogram.png")

    def test_each_terminal_gets_its_own_job(self, h5ad, graph):
        plot = source(h5ad).plot(CELL_TYPE_COLUMN).histogram().grid_histogram()
        assert names(plot) == [
            f"{CELL_TYPE_COLUMN}_histogram.png",
            f"{CELL_TYPE_COLUMN}_grid_histogram.png",
        ]

    def test_job_is_the_most_recent_one(self, h5ad, graph):
        plot = source(h5ad).plot(CELL_TYPE_COLUMN).histogram().grid_histogram()
        assert plot.job_ is plot.jobs_[-1]

    def test_a_config_call_carries_the_jobs_along(self, h5ad, graph):
        plot = source(h5ad).plot(CELL_TYPE_COLUMN).histogram()
        assert plot.panel_size(3, 3).jobs_ == plot.jobs_

    def test_builder_jobs_collect_every_plot(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        builder.plot("n_genes").histogram()
        assert len(builder.jobs_) == 2

    def test_builder_jobs_include_derived_builders(self, h5ad, graph):
        """A copy shares the session; jobs_ is the whole lineage."""
        builder = source(h5ad)
        builder.facet(CELL_TYPE_COLUMN).plot("n_genes").histogram()
        assert len(builder.jobs_) == 1

    def test_builder_jobs_reset_with_a_new_graph(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        ppg.new(run_mode=ppg.RunMode.CONSOLE)
        assert builder.jobs_ == []

    def test_default_names(self, h5ad, graph):
        builder = source(h5ad)
        plot = builder.plot("S100A8")
        assert names(
            plot.scatter(), plot.violin(CELL_TYPE_COLUMN), plot.ridgeline("coarse")
        ) == [
            "S100A8_scatter.png",
            "S100A8_violin_leiden.png",
            "S100A8_ridgeline_coarse.png",
        ]

    def test_an_explicit_column_stays_out_of_the_name(self, h5ad, graph):
        plot = source(h5ad).plot("S100A8").scatter(column="CST3")
        assert names(plot) == ["S100A8_scatter.png"]

    def test_a_generic_output_reaches_unwrapped_terminals(self, h5ad, graph):
        plot = source(h5ad).plot("S100A8").output("plot_moran_markers", min_moran=0.3)
        assert names(plot) == ["S100A8_moran_markers.png"]

    def test_a_generic_output_rejects_a_config_method(self, h5ad, graph):
        with pytest.raises(ValueError, match="not a ScatterPlotter terminal"):
            source(h5ad).plot("x").output("style", dot_size=1)

    def test_a_terminal_without_a_source_is_rejected(self, graph):
        with pytest.raises(ValueError, match="no data source"):
            PlotBuilder(output_folder=RESULTS).plot(CELL_TYPE_COLUMN).histogram()

    def test_a_terminal_without_a_graph_is_rejected(self, h5ad, monkeypatch):
        monkeypatch.setattr(ppg, "global_pipegraph", None)
        with pytest.raises(RuntimeError, match="needs an active graph"):
            source(h5ad).plot(CELL_TYPE_COLUMN).histogram()


# ---------------------------------------------------------------------------
# output paths
# ---------------------------------------------------------------------------


class TestOutputPaths:
    def test_outputs_are_written(self, h5ad, workdir):
        run(lambda: source(h5ad).plot(CELL_TYPE_COLUMN).histogram())
        assert (workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png").exists()

    def test_several_outputs_land_side_by_side(self, h5ad, workdir):
        run(
            lambda: source(h5ad)
            .plot(CELL_TYPE_COLUMN)
            .histogram()
            .grid_histogram()
        )
        out = workdir / RESULTS
        assert (out / f"{CELL_TYPE_COLUMN}_histogram.png").exists()
        assert (out / f"{CELL_TYPE_COLUMN}_grid_histogram.png").exists()

    def test_into_nests_builder_then_plot(self, h5ad, workdir):
        run(
            lambda: source(h5ad, into="genes")
            .plot(CELL_TYPE_COLUMN, into="clusters")
            .histogram()
        )
        target = workdir / RESULTS / "genes" / "clusters"
        assert (target / f"{CELL_TYPE_COLUMN}_histogram.png").exists()

    def test_explicit_filename_is_honoured(self, h5ad, workdir):
        run(
            lambda: source(h5ad)
            .plot(CELL_TYPE_COLUMN)
            .histogram(filename="clusters.png")
        )
        assert (workdir / RESULTS / "clusters.png").exists()

    def test_the_column_reaches_the_terminal(self, h5ad, workdir):
        """A violin needs (column, group_by); injection puts them in that order."""
        run(
            lambda: source(h5ad)
            .plot("S100A8")
            .violin(CELL_TYPE_COLUMN)
        )
        assert (workdir / RESULTS / "S100A8_violin_leiden.png").exists()


class TestCollisions:
    """Two different plots may not claim one file -- but see TestRedeclaration."""

    def test_two_terminals_writing_one_file_are_rejected(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        with pytest.raises(ValueError, match="already produced by"):
            builder.style(dot_size=3).plot(CELL_TYPE_COLUMN).histogram()

    def test_the_error_names_the_first_claimant(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        with pytest.raises(ValueError, match=r"Plot\('leiden'\).histogram\(\) at"):
            builder.style(dot_size=3).plot(CELL_TYPE_COLUMN).histogram()

    def test_a_filename_disambiguates(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        plot = builder.style(dot_size=3).plot(CELL_TYPE_COLUMN).histogram(
            filename="other.png"
        )
        assert names(plot) == ["other.png"]

    def test_a_name_disambiguates(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        plot = builder.style(dot_size=3).plot(CELL_TYPE_COLUMN).histogram(name="again")
        assert names(plot) == [f"{CELL_TYPE_COLUMN}_again.png"]

    def test_separate_builders_collide_too(self, h5ad, graph):
        source(h5ad).plot(CELL_TYPE_COLUMN).histogram()
        with pytest.raises(ValueError, match="already produced by"):
            source(h5ad, base_size=20).plot(CELL_TYPE_COLUMN).histogram()

    def test_a_new_graph_clears_the_claims(self, h5ad, graph):
        source(h5ad).plot(CELL_TYPE_COLUMN).histogram()
        ppg.new(run_mode=ppg.RunMode.CONSOLE)
        source(h5ad, base_size=20).plot(CELL_TYPE_COLUMN).histogram()  # must not raise


class TestRedeclaration:
    """One graph, adjusted and re-run -- the interactive workflow."""

    def test_an_identical_redeclaration_is_not_a_collision(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        builder.plot(CELL_TYPE_COLUMN).histogram()  # must not raise

    def test_an_identical_redeclaration_yields_the_same_job(self, h5ad, graph):
        builder = source(h5ad)
        first = builder.plot(CELL_TYPE_COLUMN).histogram()
        second = builder.plot(CELL_TYPE_COLUMN).histogram()
        assert first.job_ is second.job_

    def test_an_identical_redeclaration_does_not_multiply_jobs(self, h5ad, graph):
        builder = source(h5ad)
        builder.plot(CELL_TYPE_COLUMN).histogram()
        builder.plot(CELL_TYPE_COLUMN).histogram()
        assert len(builder.jobs_) == 1

    def test_a_changed_redeclaration_warns_in_an_interactive_run_mode(self, h5ad):
        ppg.new(run_mode=ppg.RunMode.NOTEBOOK)
        try:
            builder = source(h5ad)
            builder.plot(CELL_TYPE_COLUMN).histogram()
            with pytest.warns(UserWarning, match="Redefining it"):
                builder.style(dot_size=3).plot(CELL_TYPE_COLUMN).histogram()
        finally:
            ppg.new(run_mode=ppg.RunMode.CONSOLE)

    def test_an_identical_redeclaration_stays_quiet_in_a_notebook(self, h5ad):
        ppg.new(run_mode=ppg.RunMode.NOTEBOOK)
        try:
            builder = source(h5ad)
            builder.plot(CELL_TYPE_COLUMN).histogram()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                builder.plot(CELL_TYPE_COLUMN).histogram()
        finally:
            ppg.new(run_mode=ppg.RunMode.CONSOLE)

    def test_one_graph_can_be_declared_and_run_twice(self, h5ad, workdir):
        """The interactive loop: same graph, same declarations, run again."""

        def build(builder):
            builder.plot(CELL_TYPE_COLUMN).histogram()
            builder.plot("n_genes").histogram()

        ppg.new(run_mode=ppg.RunMode.CONSOLE)
        build(source(h5ad))
        ppg.run()
        targets = [
            workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png",
            workdir / RESULTS / "n_genes_histogram.png",
        ]
        before = mtimes(*targets)

        build(source(h5ad))  # same graph, re-declared
        ppg.run()
        assert mtimes(*targets) == before

    def test_an_adjusted_plot_reruns_alone(self, h5ad, workdir):
        """Adjust one plot in an interactive graph: only that file is rebuilt."""

        def build(dot_size):
            builder = source(h5ad)
            builder.plot(CELL_TYPE_COLUMN).style(dot_size=dot_size).histogram()
            builder.plot("n_genes").grid_histogram(column=CELL_TYPE_COLUMN)

        ppg.new(run_mode=ppg.RunMode.NOTEBOOK)
        try:
            build(1)
            ppg.run()
            adjusted = workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png"
            untouched = workdir / RESULTS / "n_genes_grid_histogram.png"
            before_adjusted, before_untouched = mtimes(adjusted, untouched)

            with pytest.warns(UserWarning, match="Redefining it"):
                build(3)
            ppg.run()
            assert mtimes(adjusted)[0] != before_adjusted
            assert mtimes(untouched)[0] == before_untouched
        finally:
            ppg.new(run_mode=ppg.RunMode.CONSOLE)


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------


class TestInvariants:
    """Everything recorded must be covered; nothing else may cause churn."""

    def graph(self, h5ad, *, builder_size=(4, 4), plot_size=None, dot_size=1,
              filter_fn=None, extra_plot=False, sibling_vertical=None):
        def build():
            builder = source(h5ad).panel_size(*builder_size).style(dot_size=dot_size)
            if filter_fn is not None:
                builder = builder.set_filter(filter_fn)
            plot = builder.plot(CELL_TYPE_COLUMN)
            if plot_size is not None:
                plot = plot.panel_size(*plot_size)
            plot = plot.histogram()
            if sibling_vertical is not None:
                plot.grid_histogram(vertical=sibling_vertical)
            if extra_plot:
                builder.plot("n_genes").histogram()
            return builder.jobs_

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

    def test_a_sibling_output_does_not_rerun_this_one(self, h5ad, target):
        """One job per file: a violin's settings cannot disturb the histogram."""
        run(self.graph(h5ad, sibling_vertical=False))
        before = mtimes(target)
        run(self.graph(h5ad, sibling_vertical=True))
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

    def test_divergent_builders_do_not_share_invariants(self, h5ad, workdir):
        """Two builder copies, two different callables, one graph."""

        def keep_all(data):
            return np.ones(data.coordinates().shape[0], dtype=bool)

        def keep_none_at_all(data):
            mask = np.ones(data.coordinates().shape[0], dtype=bool)
            mask[0] = False
            return mask

        def build():
            builder = source(h5ad)
            builder.set_filter(keep_all).plot(CELL_TYPE_COLUMN).histogram(
                filename="a.png"
            )
            builder.set_filter(keep_none_at_all).plot(CELL_TYPE_COLUMN).histogram(
                filename="b.png"
            )

        run(build)
        assert (workdir / RESULTS / "a.png").exists()
        assert (workdir / RESULTS / "b.png").exists()

    def test_the_same_builder_twice_shares_invariants(self, h5ad, workdir):
        """Identical scripts must not multiply FunctionInvariants either."""

        def keep_all(data):
            return np.ones(data.coordinates().shape[0], dtype=bool)

        def build():
            for stem in ("a", "b"):
                source(h5ad).set_filter(keep_all).plot(CELL_TYPE_COLUMN).histogram(
                    filename=f"{stem}.png"
                )

        run(build)
        assert (workdir / RESULTS / "b.png").exists()


class TestDependencies:
    def test_a_source_job_is_waited_for(self, workdir):
        """(path, job) sources make the plot wait for the file's producer."""
        # relative: pypipegraph2 refuses absolute paths as job ids
        path = Path("generated.h5ad")

        def make_source(output_filename, EXAMPLE_DATA=EXAMPLE_DATA):
            shutil.copy(EXAMPLE_DATA, output_filename)

        def build():
            job = ppg.FileGeneratingJob(path, make_source)
            builder = PlotBuilder(output_folder=RESULTS).set_source(
                (path, job), embedding="umap"
            )
            builder.plot(CELL_TYPE_COLUMN).histogram()

        run(build)
        assert (workdir / path).exists()
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
            source(h5ad).add_derived_source({"score": score}).plot("score").histogram()

        run(build)
        assert (workdir / marker).exists()
        assert (workdir / RESULTS / "score_histogram.png").exists()

    def test_the_created_job_can_be_depended_on(self, h5ad, workdir):
        """job_ is a real job -- downstream work can wait for the figure."""
        note = Path("note.txt")

        def build():
            plot = source(h5ad).plot(CELL_TYPE_COLUMN).histogram()
            follow_up = ppg.FileGeneratingJob(note, lambda of: of.write_text("seen"))
            follow_up.depends_on(plot.job_)

        run(build)
        assert (workdir / note).exists()


class TestReachability:
    """Configuration the old wrapper could not express must now go through."""

    def test_a_previously_unreachable_method_is_recordable(self, h5ad, workdir):
        def build():
            (
                source(h5ad)
                .background(enabled=True)  # never exposed by the old wrapper
                # a plotnine object, so the invariant encoder is exercised too
                .theme(plot_title=p9.element_text(size=14))
                .plot(CELL_TYPE_COLUMN)
                .histogram()
            )

        run(build)
        assert (workdir / RESULTS / f"{CELL_TYPE_COLUMN}_histogram.png").exists()

    def test_an_unwrapped_terminal_is_reachable(self, h5ad, workdir):
        run(lambda: source(h5ad).plot(name="cells").density(bins=8))
        assert (workdir / RESULTS / "cells_density.png").exists()
