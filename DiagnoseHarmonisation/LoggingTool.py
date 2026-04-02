import os
import io
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt  # for typing; you can create figures however you like

"""LoggingTool.py:

Enhanced logging and HTML report generation for diagnostic reports.
Provides the StatsReporter class that allows logging text and plots,
organizing them into sections, and writing a structured HTML report with
a table of contents.

Functions:
- log_section(section_id, title): mark a new named section in the log
- log_plot(fig, caption, section=None): attach a plot to a section (defaults to last section)
- write_report(...) builds a TOC with hyperlinks and places each section's plots immediately after its logs.

To start using the StatsReporter, create an instance (optionally with save_dir and report_name),
then use it as a context manager:
    E.G: 
    with StatsReporter(save_dir="reports", report_name="diagnostic_report.html") as reporter:
        reporter.log_section("batch_effects", "Batch Effects Analysis")
        reporter.log_text("Analyzing batch effects...")
        fig = plt.figure()
        # ... create plot ...
        reporter.log_plot(fig, "Batch Effects Plot")
        reporter.log_summary("Batch Effects Summary", {"Effect Size": 0.8, "p-value": 0.01})
    This will automatically write the report upon exiting the context.
"""


class _MemoryLogHandler(logging.Handler):
    """Collects formatted log records in memory (for embedding in HTML)."""
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self.records.append(msg)

    def get_records(self) -> List[str]:
        return list(self.records)

    def get_text(self) -> str:
        # fallback (not used for structured rendering)
        return "\n".join(self.records)


class StatsReporter:
    """
    Enhanced StatsReporter:
      - log_section(section_id, title): mark a new named section in the log
      - log_plot(fig, caption, section=None): attach a plot to a section (defaults to last section)
      - write_report(...) builds a TOC with hyperlinks and places each section's plots immediately after its logs.
    """
    def __init__(
        self,
        save_dir: Optional[os.PathLike] = None,
        report_name: str = "report.html",
        plaintext_log_name: str = "run.log",
        save_artifacts: bool = False,
        logger_name: str = "statsreporter",
        log_level: int = logging.INFO,
        toc: bool = True,  # new: whether to include a table of contents
    ):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.report_name = report_name
        self.plaintext_log_name = plaintext_log_name
        self.save_artifacts = save_artifacts
        self._include_toc = toc

        # Create logger
        self.logger = logging.getLogger(logger_name + f".{id(self)}")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # avoid duplicate output

        # Formatter
        self._fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

        # In-memory handler for HTML embedding
        self._mem_handler = _MemoryLogHandler()
        self._mem_handler.setFormatter(self._fmt)
        self.logger.addHandler(self._mem_handler)

        # Optional plaintext file handler (only when saving artifacts)
        self._file_handler: Optional[logging.Handler] = None
        self._artifacts_paths: List[Path] = []
        if self.save_artifacts and self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logfile = self.save_dir / self.plaintext_log_name
            self._file_handler = logging.FileHandler(logfile, encoding="utf-8")
            self._file_handler.setLevel(log_level)
            self._file_handler.setFormatter(self._fmt)
            self.logger.addHandler(self._file_handler)
            self._artifacts_paths.append(logfile)

        # Section bookkeeping:
        # list of tuples in the order created: (section_id, title, start_index_in_records)
        self._sections: List[Tuple[str, str, int]] = []
        # mapping section_id -> list of (caption, base64_png)
        self._section_plots: Dict[str, List[Tuple[str, str]]] = {}
        # plots that were logged before any section was created or explicitly placed
        self._unplaced_plots: List[Tuple[str, str]] = []

    # ---------------- Logging helpers ------------------------------------

    def log_section(self, section_id: str, title: str):
        """
        Start a named section. The section_id should be unique-ish (used for anchors).
        Logs a header line (so it appears in the plaintext log too).
        """
        # sanitize section id for anchor usage (simple)
        sec = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in section_id)
        start_index = len(self._mem_handler.records)
        self._sections.append((sec, title, start_index))
        # initialize plots list
        self._section_plots.setdefault(sec, [])
        # Also write a clear header line into the text log
        self.logger.info(f"[SECTION] {title}")

    def log_summary(self, test_name: str, summary: dict):
        """
        Write a one-line header + pretty key/val summary via logger.
        """
        self.logger.info(f"[SUMMARY] {test_name}")
        # format dict consistently:
        for k, v in summary.items():
            self.logger.info(f"  - {k}: {v}")

    def log_text(self, message: str, level: int = logging.INFO):
        """Free-form text to the log."""
        self.logger.log(level, message)

    def text_simple(self, message: str, level=logging.INFO):
        """Log without date and time to reduce clutter """
        fmt_backup = self._mem_handler.formatter
        self._mem_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.log(level, message)
        self._mem_handler.setFormatter(fmt_backup)

    def log_plot(self, fig, caption: str = "Plot", section: Optional[str] = None):
        """
        Accepts a matplotlib Figure, attaches it to a section (or last section if None),
        embeds it into the HTML report, and (optionally) saves it as a PNG in save_dir when save_artifacts=True.
        """
        # Save to a bytes buffer (PNG) and base64-encode for inline HTML
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")

        # determine destination section
        target_section = None
        if section is not None:
            sec = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in section)
            if any(s[0] == sec for s in self._sections):
                target_section = sec
            else:
                # unknown section => create it at current position
                self._sections.append((sec, section, len(self._mem_handler.records)))
                self._section_plots.setdefault(sec, [])
                target_section = sec
        else:
            # use last section if present
            if self._sections:
                target_section = self._sections[-1][0]

        if target_section is None:
            # no section to attach to -> store as unplaced
            self._unplaced_plots.append((caption, b64))
        else:
            self._section_plots.setdefault(target_section, []).append((caption, b64))

        # Save PNG of figure to same directory if save_artifacts==true
        if self.save_artifacts and self.save_dir is not None:
            fname = self._unique_png_name(caption)
            outpath = self.save_dir / fname
            fig.savefig(outpath, format="png", dpi=150, bbox_inches="tight")
            self._artifacts_paths.append(outpath)
            self.logger.info(f"[ARTIFACT] Saved plot: {outpath}")

    # ---------------- Report writing ------------------------------------

    def write_report(self, path: Optional[os.PathLike] = None) -> Path:
        """
        Write (or rewrite) the HTML report and return its path.
        """
        if path is None:
            path = self.report_path
        path = Path(path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Use the structured renderer that has access to raw records and sections
        html = self._render_html_structured(
            title="Statistical Tests Report",
            records=self._mem_handler.get_records(),
            sections=self._sections,
            section_plots=self._section_plots,
            unplaced_plots=self._unplaced_plots,
            toc=self._include_toc,
        )
        path.write_text(html, encoding="utf-8")
        return path

    @property
    def report_path(self) -> Path:
        return (self.save_dir or Path.cwd()) / self.report_name

    @property
    def artifacts(self) -> List[Path]:
        """List of any files saved in addition to the HTML report."""
        return list(self._artifacts_paths)

    # ---- context manager ---------------------------------------------------

    def __enter__(self):
        self.logger.info("=== Report session started ===")
        self.logger.info(f"save_artifacts={self.save_artifacts} | save_dir={self.save_dir}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            # preserve exception info in the plaintext log
            self.logger.exception("Exception during reporting", exc_info=(exc_type, exc_type, tb))
        self.logger.info("=== Report session finished ===")
        # Always write/update the report on exit
        report = self.write_report()

        # Clean up handlers
        self.logger.removeHandler(self._mem_handler)
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
        self._mem_handler.close()

        # Print the saved report and figures (artifacts) to console
        print(f"[StatsReporter] Wrote HTML report → {report}")
        if self._artifacts_paths:
            print("[StatsReporter] Saved Figures:")
            for p in self._artifacts_paths:
                print(f"  - {p}")

    # ---- helpers -----------------------------------------------------------

    def _unique_png_name(self, caption: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in caption)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{ts}_{safe}.png"

    @staticmethod
    def _render_html_structured(
        title: str,
        records: List[str],
        sections: List[Tuple[str, str, int]],
        section_plots: Dict[str, List[Tuple[str, str]]],
        unplaced_plots: List[Tuple[str, str]],
        toc: bool = True,
    ) -> str:
        """Render an HTML string with:
           - top Table of Contents linking to section anchors (if toc=True),
           - logs arranged by section, and each section's plots inserted right after the section logs.
        """
        # Build a simple TOC
        toc_html = ""
        if toc and sections:
            toc_items = []
            for sec_id, title, _ in sections:
                toc_items.append(f'<li><a href="#sec_{sec_id}">{title}</a></li>')
            toc_html = "<nav><h2>Contents</h2><ul>" + "\n".join(toc_items) + "</ul></nav>"

        # Build section boundaries: sort sections by their start index
        sections_sorted = sorted(sections, key=lambda x: x[2])

        # We'll iterate through record indices and allocate ranges to sections
        html_pieces = []
        current_idx = 0
        for i, (sec_id, title, start_idx) in enumerate(sections_sorted):
            # logs from current_idx up to start_idx belong to "pre-section" (or previous sections)
            if start_idx > current_idx:
                chunk = records[current_idx:start_idx]
                html_pieces.append(_render_log_chunk(chunk))
                current_idx = start_idx

            # logs for this section: from start_idx up to next_section_start (or until end)
            next_start = sections_sorted[i + 1][2] if i + 1 < len(sections_sorted) else len(records)
            section_chunk = records[start_idx:next_start]
            # Render section header + logs
            section_html = f'<section id="sec_{sec_id}" class="section"><h2>{title}</h2>'
            section_html += _render_log_chunk(section_chunk)
            # add plots for this section (if any)
            plots = section_plots.get(sec_id, [])
            if plots:
                section_html += "<div class='plots'>"
                for caption, b64 in plots:
                    section_html += f"""
                    <figure>
                      <img src="data:image/png;base64,{b64}" alt="{caption}" />
                      <figcaption>{caption}</figcaption>
                    </figure>
                    """
                section_html += "</div>"
            section_html += "</section>"
            html_pieces.append(section_html)
            current_idx = next_start

        # Any remaining logs after last section
        if current_idx < len(records):
            tail_chunk = records[current_idx:]
            html_pieces.append(_render_log_chunk(tail_chunk))

        # Unplaced plots (those without a section)
        unplaced_html = ""
        if unplaced_plots:
            unplaced_html = "<section id='unplaced_plots' class='section'><h2>Unplaced plots</h2><div class='plots'>"
            for caption, b64 in unplaced_plots:
                unplaced_html += f"""
                <figure>
                  <img src="data:image/png;base64,{b64}" alt="{caption}" />
                  <figcaption>{caption}</figcaption>
                </figure>
                """
            unplaced_html += "</div></section>"

        # Combine into final HTML
        html_body = "\n".join([toc_html] + html_pieces + ([unplaced_html] if unplaced_html else []))

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }}
    h1 {{ margin-top: 0; }}
    figure {{ margin: 0 0 1.5rem 0; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ color: #555; font-size: 0.9rem; margin-top: 0.25rem; }}
    .log {{ background: #f6f8fa; border: 1px solid #e5e7eb; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }}
    .section {{ margin-bottom: 2rem; }}
    nav ul {{ margin: 0 0 1rem 1.25rem; }}
    nav li {{ margin: 0.25rem 0; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {html_body}
</body>
</html>
"""


def _render_log_chunk(records: List[str]) -> str:
    # Escape HTML characters in each record and join into a single pre block
    safe_lines = []
    for r in records:
        safe = (r or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_lines.append(safe)
    return "<div class='log'><pre style='margin:0'>" + "\n".join(safe_lines) + "</pre></div>"
