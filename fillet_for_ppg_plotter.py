from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import pypipegraph2 as ppg
from mbf_singlecell_plotter import ScatterPlotter, EmbeddingData, H5adFacade
import plotnine as p9
import numpy as np


@dataclass
class Plot:
    column: str | Tuple[str, str]  # what to colour by / plot; names the output file
    filename: Optional[str] = None  # override the output filename
    facet: Optional[str | Tuple[str, str]] = (
        None  # split into panels; names the sub-directory
    )
    facet_args: Optional[dict] = None  # extra args to facet/facet_2d
    style: Optional[dict] = None  # extra style, composed on top of dot_size=1
    histogram: bool = True  # also emit a grid histogram (off for continuous data)
    violin: Optional[List[str] | str] = None  # also emit violet plot(s)
    ridges: Optional[List[str]] = None
    facet_violin: Optional[List[Tuple[str, str]]] = (
        None  #  also emit violet plot facteded
    )
    global_histogram: bool = False  # an overall histogram
    global_relative_histogram: Optional[str] = None
    colors: Optional[List[str]] = None
    filter: Optional[Callable[["EmbeddingData"], "pd.Series | np.ndarray"]] = None
    hard_filter: Optional[Callable[["EmbeddingData"], "pd.Series | np.ndarray"]] = None
    grey_border: Optional[bool] = None
    subfolder: Optional[str] = None  # subfolder for the plot, e.g. 'genes'
    title: Optional[str | Callable[[str], str]] = None
    anti_overplot_seed: Optional[int] = None
    ascending: Optional[bool] = None
    dpi: int = 150

    def facet_name(self):
        assert self.facet is not None
        if isinstance(self.facet, str):
            return self.facet
        else:
            return f"{self.facet[0]}_vs_{self.facet[1]}"

    def get_facet_args(self):
        if self.facet_args is None:
            return {}
        else:
            return self.facet_args

    def derived_columns_needed(self):
        res = []
        if isinstance(self.column, str):
            res.append(self.column)
        else:
            res.append(self.column[1])
        if self.facet is not None:
            if isinstance(self.facet, str):
                res.append(self.facet)
            else:
                res.extend(self.facet)
        if self.do_border():
            res.append("constant")
        if self.violin is not None:
            if isinstance(self.violin, str):
                res.append(self.violin)
            else:
                res.extend(self.violin)
        if self.facet_violin is not None:
            for x_column, facet_column in self.facet_violin:
                res.append(x_column)
                res.append(facet_column)
        return res

    def do_border(self):
        if self.grey_border:
            return True
        elif self.facet is not None:
            return True
        return False


def build_plotter(
    plot: Plot,
    input_file: Path,
    input_embedding: str,
    gene_source,
    COLUMN_SOURCES,
    COLUMN_COLORS,
) -> ScatterPlotter:
    p = ScatterPlotter(
        base_size=15,
    ).set_source(
        input_file,
        input_embedding,
        alternative_id_column="gene_name",
        transform=lambda x: x / np.log(2),
        layer="X",
    )
    if gene_source is not None:
        p = p.add_alternative_source(gene_source, name="genes")
    for column in plot.derived_columns_needed():
        if column in COLUMN_SOURCES:
            if isinstance(COLUMN_SOURCES[column], Path):
                p = p.add_alternative_source(COLUMN_SOURCES[column])
            else:
                p = p.add_derived_source({column: COLUMN_SOURCES[column]})
    if plot.filter:
        p = p.set_filter(plot.filter)
    if plot.hard_filter:
        p = p.hard_filter(plot.hard_filter)
    p = p.style(dot_size=1)
    if plot.style:
        p = p.style(**plot.style)
    if plot.column == "-density":
        p = p.colormap(title="density")
    else:
        if "_" not in plot.column:
            p = p.colormap(title="log2 expression")
    colors = plot.colors
    if colors is None:
        colors = COLUMN_COLORS.get(plot.column)
    if colors is not None:
        p = p.colormap_discrete(colors, title=plot.column)
    if plot.facet:
        if isinstance(plot.facet, str):
            p = p.facet(plot.facet, **plot.get_facet_args())
        else:
            p = p.facet_2d(*plot.facet, **plot.get_facet_args())
    if plot.do_border():
        # so we can turn it off or force it on,b ut by default it's on.
        p = p.with_borders(
            cell_type_column="constant", colors=["#707070"], legend=False, size=10
        )
    if plot.anti_overplot_seed:
        p = p.anti_overplot(seed=plot.anti_overplot_seed)
    if plot.ascending is not None:
        p = p.anti_overplot(ascending=plot.ascending)

    p = p.panel_size(4, 4)
    return p


func_jobs = {}


all_color_deps = {}
dep_build_plotter_func = None


def build_plotter_deps(plot, plot_name, COLUMN_SOURCES, COLUMN_COLORS):
    global all_color_deps, dep_build_plotter_func
    if not hasattr(ppg.global_pipegraph, "sc_func_jobs"):
        ppg.global_pipegraph.sc_func_jobs = {}
    func_jobs = ppg.global_pipegraph.sc_func_jobs

    if dep_build_plotter_func is None:
        dep_build_plotter_func = ppg.FunctionInvariant("build_plotter", build_plotter)

    if id(COLUMN_COLORS) not in all_color_deps:
        all_color_deps[id(COLUMN_COLORS)] = {
            name: ppg.ParameterInvariant(f"color_{name}", values)
            for (name, values) in COLUMN_COLORS.items()
        }
    color_deps = all_color_deps[id(COLUMN_COLORS)]

    res = [dep_build_plotter_func]
    for column in plot.derived_columns_needed():
        if column in COLUMN_SOURCES:
            if column not in func_jobs:
                if callable(COLUMN_SOURCES[column]):
                    func_jobs[column] = ppg.FunctionInvariant(
                        f"column_{column}", COLUMN_SOURCES[column]
                    )
                    if hasattr(COLUMN_SOURCES[column], "deps"):
                        res.append(COLUMN_SOURCES[column].deps)
                elif isinstance(COLUMN_SOURCES[column], Path):
                    func_jobs[column] = ppg.FileInvariant(COLUMN_SOURCES[column])
                else:
                    raise ValueError(
                        f"Invalid column source for {column}: {COLUMN_SOURCES[column]}"
                    )
            res.append(func_jobs[column])
        if plot.filter is not None:
            res.append(ppg.FunctionInvariant(f"filter_{plot.column}", plot.filter))
        if column in color_deps:
            res.append(color_deps[column])
    res.append(dep_build_plotter_func)
    if callable(plot.hard_filter):
        if hasattr(plot.hard_filter, "deps"):
            res.append(plot.hard_filter.deps)
        else:
            res.append(
                ppg.FunctionInvariant(plot_name + "_hard_filter", plot.hard_filter)
            )
    else:
        res.append(
            ppg.ParameterInvariant(plot_name + "_hard_filter_args", plot.hard_filter)
        )
    if plot.ascending is not None:
        res.append(ppg.ParameterInvariant(plot_name + "_ascending", plot.ascending))
    return res


def register_plot(
    plot: Plot,
    rd: Path,
    input_file,
    input_embedding,
    genes_from,
    COLUMN_SOURCES,
    COLUMN_COLORS,
):
    if plot.subfolder:
        out_dir = (rd / plot.facet_name() if plot.facet else rd) / plot.subfolder
    else:
        out_dir = rd / plot.facet_name() if plot.facet else rd
    filename_name = plot.filename if plot.filename else plot.column
    input_filename, input_file_job = ppg.util.job_or_filename(input_file)
    inputs = [input_file_job] + ([genes_from] if genes_from is not None else [])
    outputs = {"scatter": out_dir / f"{filename_name}.png"}
    if plot.histogram:
        outputs["histo"] = out_dir / f"{filename_name}_histogram.png"
    if plot.global_histogram:
        outputs["global_histo"] = out_dir / f"{filename_name}_overall_histogram.png"

    if plot.violin:
        if isinstance(plot.violin, str):
            plot.violin = [plot.violin]
        for col in plot.violin:
            outputs[f"violin_{col}"] = out_dir / f"{filename_name}_violin_{col}.png"
    if plot.facet_violin:
        for x, facet in plot.facet_violin:
            outputs[f"violin_{x}_facet_{facet}"] = (
                out_dir / f"{filename_name}_violin_{x}_facet_{facet}.png"
            )
    if plot.global_relative_histogram:
        outputs["global_histo_relative"] = (
            out_dir / f"{filename_name}_overall_histogram_relative.png"
        )

    def generate(
        output_filenames,
        plot=plot,
        input_file=input_filename,
        genes_from=genes_from,
        input_embedding=input_embedding,
        global_histo_relative=plot.global_relative_histogram,
        violin=plot.violin,
        facet_violin=plot.facet_violin,
        COLUMN_SOURCES=COLUMN_SOURCES,
        COLUMN_COLORS=COLUMN_COLORS,
    ):
        output_filenames["scatter"].parent.mkdir(exist_ok=True, parents=True)
        p = build_plotter(
            plot, input_file, input_embedding, genes_from, COLUMN_SOURCES, COLUMN_COLORS
        )
        if plot.column == "-density-":
            po = p.plot_density(
                12,
                # cmap_colors=["white", "blue", "red"],
                cmap_colors=[
                    "#eFeFeF",
                    "#ECDA9A",
                    "#EFC47E",
                    "#F3AD6A",
                    "#F7945D",
                    "#F97B57",
                    "#F66356",
                    "#EE4D5A",
                ],
                quantile=0.95,
                include_counts=True,
                count_text_size=7,
            )
        else:
            po = p.plot(plot.column)
        po.save(output_filenames["scatter"], dpi=plot.dpi)
        if "histo" in output_filenames:
            p.plot_grid_histogram(plot.column, scale_by_count=True).save(
                output_filenames["histo"]
            )
        if "global_histo" in output_filenames:
            p.plot_histogram(plot.column).save(output_filenames["global_histo"])
        if "global_histo_relative" in output_filenames:
            p.plot_histogram(plot.column, normalize_to=global_histo_relative).save(
                output_filenames["global_histo_relative"]
            )
        if violin is not None:
            for violin_x_column in violin:
                colors = COLUMN_COLORS.get(violin_x_column)
                p = p.colormap_discrete(
                    cmap_or_list_or_dict=colors, title=violin_x_column
                )  # setting it to none should get default colors back
                po = p.plot_violin(plot.column, violin_x_column)
                po += p9.theme(axis_text_x=p9.element_text(rotation=90))
                po.save(output_filenames[f"violin_{violin_x_column}"])
        if facet_violin is not None:
            for violin_x_column, facet_column in facet_violin:
                colors = COLUMN_COLORS.get(violin_x_column)
                p = p.colormap_discrete(
                    cmap_or_list_or_dict=colors, title=violin_x_column
                )  # setting it to none should get default colors back
                po = p.plot_violin(plot.column, violin_x_column, [facet_column])
                po += p9.facet_wrap(facet_column, scales="free_x")
                po += p9.theme(axis_text_x=p9.element_text(rotation=90))
                po.save(
                    output_filenames[f"violin_{violin_x_column}_facet_{facet_column}"]
                )

    job = ppg.MultiFileGeneratingJob(outputs, generate, depend_on_function=True)

    for f in inputs:
        if isinstance(f, (ppg.Job, list)):
            job.depends_on(f)
        else:
            job.depends_on_file(f)
    job.depends_on(
        build_plotter_deps(plot, str(job["scatter"]), COLUMN_SOURCES, COLUMN_COLORS)
    )
    jobs = []
    jobs.append(job)

    if plot.ridges:

        def generate_density(
            output_filenames,
            plot=plot,
            input_file=input_filename,
            genes_from=genes_from,
            input_embedding=input_embedding,
            ridges=plot.ridges,
            COLUMN_SOURCES=COLUMN_SOURCES,
            COLUMN_COLORS=COLUMN_COLORS,
        ):
            plot.facet = None
            p = build_plotter(
                plot,
                input_file,
                input_embedding,
                genes_from,
                COLUMN_SOURCES,
                COLUMN_COLORS,
            )
            for split in ridges:
                output_filenames[split].parent.mkdir(exist_ok=True, parents=True)
                po = p.plot_ridgeline(plot.column, split, scales="fixed")
                po.save(output_filenames[split])

        job = ppg.MultiFileGeneratingJob(
            {k: out_dir / f"{filename_name}_{k}_ridge.png" for k in plot.ridges},
            generate_density,
        )
        for f in inputs:
            if isinstance(f, (ppg.Job, list)):
                job.depends_on(f)
            else:
                job.depends_on_file(f)
        job.depends_on(
            build_plotter_deps(
                plot, str(job[plot.ridges[0]]), COLUMN_SOURCES, COLUMN_COLORS
            )
        )
        jobs.append(job)

    return jobs
