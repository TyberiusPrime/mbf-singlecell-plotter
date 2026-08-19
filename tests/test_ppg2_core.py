"""Tests for the ppg2 call recorder that need no pipegraph.

Recording, introspection, replay and argument encoding are deliberately free of
pypipegraph2 so they can be exercised directly; the job wiring built on top of
them lives in test_ppg2.py.
"""

import functools
import inspect
import json
from pathlib import Path

import numpy as np
import plotnine as p9
import pytest

from mbf_singlecell_plotter import ppg2
from mbf_singlecell_plotter.plots import ScatterPlotter
from mbf_singlecell_plotter.ppg2 import (
    Output,
    Plot,
    PlotBuilder,
    UnencodableArgument,
    _Call,
    _replay,
    _Walker,
)

# Public ScatterPlotter methods that are neither configuration nor a plot, and
# so legitimately have no counterpart on the recorder.
NON_PLOT_METHODS = {
    "get_cluster_markers",
    "get_column",
    "get_morans_i_markers",
    "save_interactive_cluster_markers",
    "save_interactive_moran_grid",
}


def a_filter(data):
    return np.ones(1, dtype=bool)


def script(builder, plot, output, data):
    """The merged script, with an in-memory source spliced in.

    ``PlotBuilder.set_source`` only accepts trackable paths, so these
    pipegraph-free tests supply the session's EmbeddingData directly.
    """
    return (
        [_Call("set_source", (data,))] + builder._calls + plot._calls + output._calls
    )


# ---------------------------------------------------------------------------
# introspection: the recorder must keep up with ScatterPlotter by itself
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_every_public_method_is_classified(self):
        """A new ScatterPlotter method must not be silently unreachable."""
        public = {
            name
            for name, _ in inspect.getmembers(ScatterPlotter, inspect.isfunction)
            if not name.startswith("_")
        }
        classified = set(ppg2.CONFIG_METHODS) | set(ppg2.TERMINAL_METHODS)
        assert public - classified - NON_PLOT_METHODS == set()

    def test_config_and_terminal_are_disjoint(self):
        assert not set(ppg2.CONFIG_METHODS) & set(ppg2.TERMINAL_METHODS)

    @pytest.mark.parametrize("cls", [PlotBuilder, Plot, Output])
    def test_config_methods_are_installed(self, cls):
        for name in ppg2.CONFIG_METHODS:
            assert callable(getattr(cls, name)), name

    def test_output_shorthands_are_installed(self):
        for terminal in ppg2.TERMINAL_METHODS:
            assert callable(getattr(Plot, ppg2._output_name_for(terminal)))
        assert callable(Plot.scatter) and callable(Plot.violin)

    def test_source_params_still_exist(self):
        """SOURCE_PARAMS hardcodes two parameter names -- keep them honest."""
        for method, param in ppg2.SOURCE_PARAMS.items():
            signature = inspect.signature(ppg2.CONFIG_METHODS[method])
            assert list(signature.parameters)[1] == param

    def test_reserved_output_kwargs_do_not_shadow_terminals(self):
        """name=/filename=/dpi= are ours; no terminal may claim them."""
        for terminal, fn in ppg2.TERMINAL_METHODS.items():
            params = set(inspect.signature(fn).parameters)
            assert not params & set(ppg2.RESERVED_OUTPUT_KWARGS), terminal

    def test_docstrings_are_carried_over(self):
        assert "dot_size" in Plot.style.__doc__
        assert "Recorded, not executed" in Plot.style.__doc__

    def test_signature_is_carried_over(self):
        assert "dot_alpha" in inspect.signature(Plot.style).parameters


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_calls_are_recorded_not_executed(self):
        plot = Plot("S100A8").style(dot_size=3).panel_size(2, 2)
        assert [c.method for c in plot._calls] == ["style", "panel_size"]
        assert plot._calls[0].kwargs == {"dot_size": 3}
        assert plot._calls[1].args == (2, 2)

    def test_config_methods_chain(self):
        plot = Plot("x")
        assert plot.style(dot_size=1) is plot

    def test_unknown_keyword_fails_at_record_time(self):
        with pytest.raises(TypeError, match="dot_sizes"):
            Plot("x").style(dot_sizes=3)

    def test_too_many_positionals_fail_at_record_time(self):
        with pytest.raises(TypeError, match="panel_size"):
            Plot("x").panel_size(1, 2, 3)

    def test_unknown_terminal_keyword_fails_at_record_time(self):
        with pytest.raises(TypeError, match="scatter"):
            Plot("x").scatter(nonsense=1)

    def test_builder_init_kwargs_are_validated(self):
        assert PlotBuilder(base_size=20).init_kwargs == {"base_size": 20}
        with pytest.raises(TypeError, match="base_sizes"):
            PlotBuilder(base_sizes=20)

    def test_string_source_is_rejected(self):
        """A str path would silently lose its file invariant."""
        with pytest.raises(TypeError, match="must be a"):
            PlotBuilder().set_source("analysis.h5ad")

    def test_in_memory_source_is_rejected(self, ad):
        with pytest.raises(TypeError, match="AnnData"):
            PlotBuilder().set_source(ad)

    def test_string_alternative_source_is_rejected(self):
        with pytest.raises(TypeError, match="must be a"):
            PlotBuilder().add_alternative_source("genes.h5ad", name="genes")

    def test_path_source_is_accepted(self):
        builder = PlotBuilder().set_source(Path("analysis.h5ad"), embedding="umap")
        assert builder._calls[0].args == (Path("analysis.h5ad"),)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


class TestOutputs:
    def test_column_is_injected(self):
        output = Plot("S100A8").scatter()
        assert output.terminal == "plot"
        assert output.terminal_args == ("S100A8",)

    def test_column_is_injected_before_positional_args(self):
        """violin('leiden') means (column=S100A8, group_by=leiden)."""
        output = Plot("S100A8").violin("leiden")
        assert output.terminal_args == ("S100A8", "leiden")

    def test_explicit_column_wins(self):
        output = Plot("S100A8").scatter(column="CST3")
        assert output.terminal_args == ()
        assert output.terminal_kwargs == {"column": "CST3"}

    def test_terminal_without_column_gets_none(self):
        output = Plot(name="density").density(bins=12)
        assert output.terminal == "plot_density"
        assert output.terminal_args == ()
        assert output.terminal_kwargs == {"bins": 12}

    def test_missing_column_is_an_error(self):
        with pytest.raises(TypeError, match="no column"):
            Plot(name="anonymous").scatter()

    def test_default_names(self):
        plot = Plot("S100A8")
        assert plot.scatter().name == "scatter"
        assert plot.violin("leiden").name == "violin_leiden"
        assert plot.ridgeline("coarse").name == "ridgeline_coarse"

    def test_duplicate_names_are_rejected(self):
        plot = Plot("S100A8")
        plot.violin("leiden")
        with pytest.raises(ValueError, match="already has an output"):
            plot.violin("leiden")

    def test_explicit_name_disambiguates(self):
        plot = Plot("S100A8")
        plot.violin("leiden")
        assert plot.violin("leiden", name="violin_again").name == "violin_again"

    def test_file_names(self):
        plot = Plot("S100A8")
        assert plot.scatter().file_name("S100A8") == "S100A8_scatter.png"
        assert plot.violin("leiden").file_name("S100A8") == "S100A8_violin_leiden.png"

    def test_explicit_filename_is_used_verbatim(self):
        output = Plot("S100A8").scatter(filename="custom.pdf")
        assert output.file_name("ignored") == "custom.pdf"

    def test_output_takes_its_own_config(self):
        output = Plot("S100A8").ridgeline("leiden").unfacet()
        assert isinstance(output, Output)
        assert [c.method for c in output._calls] == ["unfacet"]

    def test_generic_output_reaches_unwrapped_terminals(self):
        output = Plot("S100A8").output("plot_moran_markers", min_moran=0.3)
        assert output.terminal == "plot_moran_markers"
        assert output.name == "moran_markers"

    def test_generic_output_does_not_inject_the_column(self):
        output = Plot("S100A8").output("plot_histogram", "CST3")
        assert output.terminal_args == ("CST3",)

    def test_generic_output_rejects_a_config_method(self):
        with pytest.raises(ValueError, match="not a ScatterPlotter terminal"):
            Plot("x").output("style", dot_size=1)

    def test_stem_falls_back_to_the_column(self):
        assert Plot("S100A8").stem == "S100A8"
        assert Plot(("genes", "CST3")).stem == "CST3"
        assert Plot("S100A8", name="custom").stem == "custom"

    def test_stem_without_column_or_name_is_an_error(self):
        with pytest.raises(ValueError, match="needs a name"):
            _ = Plot().stem


class TestLayout:
    def test_into_composes_builder_then_plot(self):
        builder = PlotBuilder(into="genes")
        plot = builder.add(Plot("S100A8", into="markers"))
        assert builder._out_dir(plot, Path("/out")) == Path("/out/genes/markers")

    def test_into_accepts_nested_paths(self):
        builder = PlotBuilder().into("a/b")
        assert builder._out_dir(Plot("x"), Path("/out")) == Path("/out/a/b")

    def test_no_implicit_facet_subdirectory(self):
        """Faceting is a plain recorded call; it must not move files around."""
        builder = PlotBuilder()
        plot = builder.add(Plot("S100A8").facet("leiden"))
        assert builder._out_dir(plot, Path("/out")) == Path("/out")

    def test_add_returns_the_plot(self):
        builder = PlotBuilder()
        plot = Plot("S100A8")
        assert builder.add(plot) is plot
        assert builder.plots == [plot]

    def test_add_rejects_non_plots(self):
        with pytest.raises(TypeError, match="expected a Plot"):
            PlotBuilder().add("S100A8")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


class TestReplay:
    def test_replay_produces_a_configured_plotter(self, data):
        calls = [
            _Call("set_source", (data,)),
            _Call("style", (), {"dot_size": 7}),
            _Call("panel_size", (3, 4)),
        ]
        plotter = _replay(calls, {})
        assert isinstance(plotter, ScatterPlotter)
        assert plotter._dot_size == 7
        assert plotter._fixed_panel_size == (3, 4)

    def test_init_kwargs_reach_the_constructor(self):
        assert _replay([], {"base_size": 22}).base_size == 22

    def test_later_calls_win(self):
        """Overriding is just calling again -- ordinary ScatterPlotter semantics."""
        plotter = _replay([_Call("panel_size", (1, 1)), _Call("panel_size", (5, 5))], {})
        assert plotter._fixed_panel_size == (5, 5)

    def test_style_merges_per_keyword(self):
        """A plot can override dot_size without discarding the builder's alpha."""
        plotter = _replay(
            [
                _Call("style", (), {"dot_size": 1, "dot_alpha": 0.5}),
                _Call("style", (), {"dot_size": 9}),
            ],
            {},
        )
        assert plotter._dot_size == 9
        assert plotter._dot_alpha == 0.5

    def test_undo_methods_work_in_sequence(self):
        plotter = _replay([_Call("facet", ("leiden",)), _Call("unfacet", ())], {})
        assert plotter._facet_variable is None

    def test_builder_then_plot_then_output_order(self, data):
        """The documented merge order, end to end, without a pipegraph."""
        builder = PlotBuilder().panel_size(1, 1).style(dot_size=2)
        plot = builder.add(Plot("S100A8").panel_size(2, 2))
        output = plot.scatter().panel_size(3, 3)
        plotter = _replay(script(builder, plot, output, data), {})
        assert plotter._fixed_panel_size == (3, 3)
        assert plotter._dot_size == 2

    def test_output_config_does_not_leak_to_siblings(self, data):
        builder = PlotBuilder()
        plot = builder.add(Plot("S100A8").facet("leiden"))
        scatter = plot.scatter()
        ridge = plot.ridgeline("leiden").unfacet()
        assert _replay(script(builder, plot, scatter, data), {})._facet_variable
        assert _replay(script(builder, plot, ridge, data), {})._facet_variable is None

    def test_a_recorded_script_renders(self, data):
        """Everything recorded is genuinely callable on a ScatterPlotter."""
        builder = PlotBuilder().style(dot_size=2)
        plot = builder.add(Plot("S100A8"))
        output = plot.scatter()
        plotter = _replay(script(builder, plot, output, data), {})
        figure = getattr(plotter, output.terminal)(*output.terminal_args)
        assert isinstance(figure, p9.ggplot)


# ---------------------------------------------------------------------------
# argument encoding
# ---------------------------------------------------------------------------


def encode(value):
    """Encode one value with a ppg-free walker."""
    return _Walker("test").walk(value, "test")[0]


def encode_calls(calls):
    return _Walker("test").walk_calls(calls)[0]


class TestEncoding:
    @pytest.mark.parametrize("value", [None, True, 3, 3.5, "x"])
    def test_primitives_pass_through(self, value):
        assert encode(value) == value

    def test_containers_are_encoded_recursively(self):
        assert encode([1, "a", None]) == [1, "a", None]
        assert encode({"b": 1, "a": 2}) == {"__dict__": [["a", 2], ["b", 1]]}

    def test_dictionary_order_does_not_matter(self):
        assert encode({"a": 1, "b": 2}) == encode({"b": 2, "a": 1})

    def test_tuples_and_lists_are_distinguishable(self):
        assert encode((1, 2)) != encode([1, 2])

    def test_sets_are_order_independent(self):
        assert encode({"a", "b"}) == encode({"b", "a"})

    def test_numpy_is_encoded_by_value(self):
        assert encode(np.int64(3)) == 3
        assert encode(np.array([1.0, 2.0])) == {"__array__": [1.0, 2.0]}

    def test_paths_are_encoded_by_name(self):
        assert encode(Path("a/b.h5ad")) == {"__path__": "a/b.h5ad"}

    def test_functions_are_encoded_by_qualified_name(self):
        encoded = encode(a_filter)
        assert encoded == {"__function__": f"{a_filter.__module__}.a_filter"}

    def test_functions_are_collected_for_invariants(self):
        walker = _Walker("prefix")
        walker.walk(a_filter, "where")
        assert len(walker._functions) == 1
        assert walker._functions[0][0].startswith("prefix::fn0::")

    def test_partials_record_their_binding(self):
        encoded = encode(functools.partial(a_filter, 1, keyword=2))
        assert encoded["__partial__"].endswith("a_filter")
        assert encoded["__bound__"] == {
            "__dict__": [["args", [1]], ["kwargs", {"__dict__": [["keyword", 2]]}]]
        }

    def test_classes_are_encoded_by_name(self):
        assert encode(ScatterPlotter) == {
            "__class__": "mbf_singlecell_plotter.plots.ScatterPlotter"
        }

    def test_objects_are_encoded_as_type_plus_state(self):
        """plotnine elements must hash by value, never by id()."""
        encoded = encode(p9.element_text(rotation=90))
        assert "element_text" in encoded["__object__"]
        assert encoded == encode(p9.element_text(rotation=90))

    def test_equivalent_objects_of_different_state_differ(self):
        assert encode(p9.element_text(rotation=90)) != encode(
            p9.element_text(rotation=45)
        )

    def test_unencodable_values_are_rejected_loudly(self):
        class Opaque:
            __slots__ = ()

        with pytest.raises(UnencodableArgument, match="cannot encode Opaque"):
            encode(Opaque())

    def test_the_error_names_the_offending_argument(self):
        class Opaque:
            __slots__ = ()

        walker = _Walker("job")
        with pytest.raises(UnencodableArgument, match=r"job\[0\].style\(x=\.\.\.\)"):
            walker.walk_calls([_Call("style", (), {"x": Opaque()})])

    def test_self_referential_values_are_rejected(self):
        nested = []
        nested.append(nested)
        with pytest.raises(UnencodableArgument, match="nested too deeply"):
            encode(nested)

    def test_path_specs_resolve_to_plain_paths_for_replay(self):
        """The replayed call gets a Path, not the (path, job) pair."""

        class FakeJob:
            job_id = "fake"

            def depends_on(self, *_):
                pass

        walker = _Walker("test")
        _, resolved = walker.walk((Path("a.h5ad"), FakeJob()), "where")
        assert resolved == Path("a.h5ad")


class TestEncodingIsStable:
    def build(self, dot_size, transform=a_filter):
        builder = PlotBuilder(base_size=15)
        builder.set_source(Path("a.h5ad"), transform=transform)
        builder.style(dot_size=dot_size)
        return builder

    def parameters(self, builder):
        return json.dumps(encode_calls(builder._calls), sort_keys=True)

    def test_identical_scripts_encode_identically(self):
        assert self.parameters(self.build(1)) == self.parameters(self.build(1))

    def test_a_changed_argument_changes_the_encoding(self):
        assert self.parameters(self.build(1)) != self.parameters(self.build(2))

    def test_a_changed_function_identity_changes_the_encoding(self):
        def other_filter(data):
            return data

        assert self.parameters(self.build(1)) != self.parameters(
            self.build(1, transform=other_filter)
        )

    def test_keyword_order_does_not_change_the_encoding(self):
        first = PlotBuilder().style(dot_size=1, dot_alpha=0.5)
        second = PlotBuilder().style(dot_alpha=0.5, dot_size=1)
        assert self.parameters(first) == self.parameters(second)

    def test_encoding_covers_builder_level_configuration(self):
        """The bug the old wrapper had: builder defaults must be hashed too."""
        assert self.parameters(PlotBuilder().panel_size(4, 4)) != self.parameters(
            PlotBuilder().panel_size(5, 5)
        )
