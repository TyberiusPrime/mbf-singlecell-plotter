"""Partial italics for gene symbols inside matplotlib text.

Human gene symbols are set in italics by convention (``CD8A``, not ``CD8A``),
but a title or colour bar label is rarely *only* a gene symbol — it reads
``CD8A: log2 expression`` or ``ENSG00000153563 (CD8A)``, and only the symbol
should lean.

Matplotlib's usual answer to mixed styling is mathtext (``$\\mathit{CD8A}$``),
and it is the wrong tool here: in math mode digits keep their upright form
(``CD8A`` comes out as *CD*8*A*) and ``-`` becomes a math minus, so ``MT-CO1``
renders as *MT* − *CO*1.  Both are unacceptable for gene symbols, which are
full of digits and hyphens.  Switching ``mathtext.fontset`` to ``custom`` with
an italic face does not fix either.

The only faithful italic is the font's own italic face applied to a whole
``Text`` artist.  So instead of marking up the string, :class:`RunsText`
re-implements ``Text.draw`` to emit each line as a sequence of *runs* — a run
matching the gene pattern is drawn with the italic face, everything else with
the regular one — laying them out along the text's own baseline.

Drawing is overridden rather than post-processed into extra artists so the
result survives redraws: ``ggplot.save`` draws the figure once and then lets
``savefig`` draw it again, and any artist positioned in frozen display
coordinates would be stale by then.
"""

from __future__ import annotations

import re
from math import cos, radians, sin
from typing import Iterable, Optional

from matplotlib.text import Text

__all__ = [
    "RunsText",
    "gene_pattern",
    "is_ensembl_accession",
    "promote_matching_text",
]

#: An Ensembl accession: ``ENS``, an optional species code, a feature-type
#: letter, then the number — ``ENSG00000139618`` (human gene, no species code),
#: ``ENSMUSG00000017146`` (mouse gene), ``ENSDARG00000019949`` (zebrafish
#: gene), ``ENSMUST00000012345`` (mouse transcript).  The match is
#: left-anchored only, so a GENCODE version suffix (``ENSG00000139618.15``) is
#: covered without spelling it out.
#:
#: Feature types are gene, transcript, protein and exon.  A feature matrix can
#: be keyed by any of them — transcript-level quantification, exon-level
#: counts — and none of those accessions is a gene symbol.
#:
#: The species code is ``[A-Z]*`` rather than ``.*`` so the pattern cannot run
#: past a separator and swallow something that merely *contains* an accession.
_ENSEMBL_ACCESSION = re.compile(r"ENS[A-Z]*[EGPT]\d+")

# Alignment of the *box* along an axis, as a fraction of its extent.  Used to
# work out which edge stays put when the italic runs come out a different width
# than the upright string matplotlib measured during layout.
_H_FACTOR = {"left": 0.0, "center": 0.5, "right": 1.0}
_V_FACTOR = {
    "bottom": 0.0,
    "baseline": 0.0,
    "center_baseline": 0.5,
    "center": 0.5,
    "top": 1.0,
}


def is_ensembl_accession(name: str) -> bool:
    """Whether *name* is an Ensembl accession rather than a gene symbol.

    Accessions are identifiers and stay upright; only symbols are italicised.
    """
    return _ENSEMBL_ACCESSION.match(name) is not None


def gene_pattern(symbols: Iterable[str]) -> Optional[re.Pattern]:
    """Compile a pattern matching any of *symbols* as a whole word.

    The boundaries are deliberately stricter than ``\\b``: gene symbols contain
    hyphens and dots (``MT-CO1``, ``RP11-34P13.7``), so a plain word boundary
    would match the ``MT`` of ``MT-CO1`` and italicise half a symbol.
    """
    alternatives = sorted({s for s in symbols if s}, key=len, reverse=True)
    if not alternatives:
        return None
    body = "|".join(re.escape(s) for s in alternatives)
    return re.compile(rf"(?<![\w.\-])(?:{body})(?![\w.\-])")


def promote_matching_text(fig, text: str, pattern: re.Pattern) -> int:
    """Make every ``Text`` in *fig* reading exactly *text* italicise *pattern*.

    Returns how many artists were promoted, so callers (and tests) can tell a
    no-op apart from a hit.
    """
    count = 0
    for artist in fig.findobj(Text):
        if artist.get_text() != text:
            continue
        if not RunsText.can_render(artist):
            continue
        artist.__class__ = RunsText
        artist._italic_pattern = pattern  # ty: ignore
        count += 1
    return count


def _advance(renderer, text: str, prop) -> float:
    """Width of *text* in display units, matching what ``_get_layout`` measures."""
    if not text:
        return 0.0
    width, _height, _descent = renderer.get_text_width_height_descent(text, prop, False)
    return width


class RunsText(Text):
    """A ``Text`` that draws substrings matching ``_italic_pattern`` in italics.

    Promote an existing artist with :func:`promote_matching_text` rather than
    constructing one; the point is to restyle text that plotnine and matplotlib
    have already created and positioned.

    ``get_window_extent`` is deliberately *not* overridden.  Layout therefore
    keeps measuring the upright string, which keeps panel sizes identical to
    the non-italic case; :meth:`draw` compensates for the small width
    difference by re-applying the artist's own alignment to the italic runs.
    """

    _italic_pattern: re.Pattern

    @staticmethod
    def can_render(artist: Text) -> bool:
        """Whether splitting *artist* into runs is safe.

        Mathtext and TeX strings have their own parsers that own the whole
        string; slicing one into independently drawn pieces would corrupt it.
        """
        if artist.get_usetex():
            return False
        return "$" not in artist.get_text()

    def _runs(self, line: str):
        """Split *line* into ``(substring, italic)`` pairs, in order."""
        runs = []
        cursor = 0
        for match in self._italic_pattern.finditer(line):
            if match.start() > cursor:
                runs.append((line[cursor : match.start()], False))
            runs.append((match.group(), True))
            cursor = match.end()
        if cursor < len(line):
            runs.append((line[cursor:], False))
        return runs

    def _advance_align_factor(self) -> float:
        """Where the anchor sits along the direction the text advances in.

        0 means the start of the line is pinned and extra width grows forward,
        1 means the end is pinned, 0.5 means the line stays centred.  With the
        default rotation mode matplotlib aligns the *rotated* box, so a title
        rotated 90° (the colour bar) is positioned along its advance axis by
        its vertical alignment, not its horizontal one.
        """
        halign = _H_FACTOR.get(self.get_horizontalalignment(), 0.0)
        if self.get_rotation_mode() == "anchor":
            return halign
        valign = _V_FACTOR.get(self.get_verticalalignment(), 0.0)
        angle = self.get_rotation() % 360
        if angle < 45 or angle >= 315:
            return halign
        if angle < 135:
            return valign
        if angle < 225:
            return 1.0 - halign
        return 1.0 - valign

    def draw(self, renderer):
        if not self.get_visible() or self.get_text() == "":
            return
        if not self.can_render(self):
            super().draw(renderer)
            return

        renderer.open_group("text", self.get_gid())
        try:
            _bbox, info, _descent = self._get_layout(renderer)
            trans = self.get_transform()
            posx, posy = trans.transform(
                (
                    float(self.convert_xunits(self._x)),
                    float(self.convert_yunits(self._y)),
                )
            )
            _canvasw, canvash = renderer.get_canvas_width_height()

            gc = renderer.new_gc()
            gc.set_foreground(self.get_color())
            gc.set_alpha(self.get_alpha())
            gc.set_url(self._url)
            gc.set_antialiased(self._antialiased)
            self._set_gc_clip(gc)

            angle = self.get_rotation()
            rad = radians(angle)
            advance_x, advance_y = cos(rad), sin(rad)

            upright = self.get_fontproperties()
            italic = upright.copy()
            italic.set_style("italic")

            # Measure every line first: the block's own width feeds both the
            # multi-line alignment and the whole-block alignment correction.
            per_line = []
            for line, (old_width, _h), x, y in info:
                runs = []
                for text, is_gene in self._runs(line):
                    prop = italic if is_gene else upright
                    runs.append((text, prop, _advance(renderer, text, prop)))
                new_width = sum(width for _t, _p, width in runs)
                per_line.append((runs, new_width, old_width, x, y))

            old_block = max((old for _r, _n, old, _x, _y in per_line), default=0.0)
            new_block = max((new for _r, new, _old, _x, _y in per_line), default=0.0)
            malign = _H_FACTOR.get(self._get_multialignment(), 0.0)
            align = self._advance_align_factor()

            for runs, new_width, old_width, x, y in per_line:
                # The x/y matplotlib laid out already carry the upright
                # multi-line offset; swap in the one the italic widths imply,
                # then slide the whole block back under its own alignment.
                shift = (new_block - new_width) * malign
                shift -= (old_block - old_width) * malign
                shift -= (new_block - old_block) * align
                start_x = posx + x + advance_x * shift
                start_y = posy + y + advance_y * shift

                offset = 0.0
                for text, prop, width in runs:
                    if text:
                        draw_x = start_x + advance_x * offset
                        draw_y = start_y + advance_y * offset
                        if renderer.flipy():
                            draw_y = canvash - draw_y
                        renderer.draw_text(
                            gc, draw_x, draw_y, text, prop, angle, ismath=False
                        )
                    offset += width

            gc.restore()
        finally:
            renderer.close_group("text")
        self.stale = False
