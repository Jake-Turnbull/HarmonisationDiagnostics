#!/usr/bin/env python3
"""
Longitudinal GUI for DiagnoseHarmonisation.

Design goals:
- No auto-detection in the run path
- Explicit subject ID, timepoint, batch, features
- Optional covariates from the same data file or a separate covariates file
- Tkinter-based UI similar in spirit to the cross-sectional GUI
- Fully static layout: no internal scrollbars anywhere. The window
  resizes itself to fit whatever is loaded (clamped to the screen size).
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from DiagnoseHarmonisation.longitudinal_workflow import (
    LongitudinalRunConfig,
    inspect_longitudinal_inputs,
    run_longitudinal_report,
)


# --------------------------------------------------------------------------
# Visual theme
# --------------------------------------------------------------------------

COLORS = {
    "bg": "#1f2430",
    "panel": "#1f2430",
    "border": "#2A2B2B",
    "text": "#cbcdcf",
    "muted": "#6b7280",
    "accent": "#2f6fed",
    "accent_active": "#2557c4",
    "accent_text": "#ffffff",
    "success": "#1e8e5a",
    "danger": "#c0392b",
}

FONT_FAMILY = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"
FONTS = {
    "base": (FONT_FAMILY, 11),
    "bold": (FONT_FAMILY, 11, "bold"),
    "heading": (FONT_FAMILY, 12, "bold"),
    "mono": ("Consolas" if sys.platform.startswith("win") else "Menlo", 10),
    "hint": (FONT_FAMILY, 10),
}


def _configure_style(root: tk.Tk) -> None:
    root.configure(bg=COLORS["bg"])

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=FONTS["base"], background=COLORS["bg"], foreground=COLORS["text"])

    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Panel.TFrame", background=COLORS["panel"])

    style.configure(
        "TLabelframe",
        background=COLORS["panel"],
        bordercolor=COLORS["border"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=COLORS["panel"],
        foreground=COLORS["text"],
        font=FONTS["heading"],
    )

    style.configure("TLabel", background=COLORS["panel"], foreground=COLORS["text"])
    style.configure("OnBg.TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Hint.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=FONTS["hint"])
    style.configure("HintOnBg.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=FONTS["hint"])

    style.configure("TCheckbutton", background=COLORS["panel"], foreground=COLORS["text"])
    style.map("TCheckbutton", background=[("active", COLORS["panel"])])

    style.configure(
        "TEntry",
        fieldbackground=COLORS["panel"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
    )
    style.configure(
        "TCombobox",
        fieldbackground=COLORS["panel"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
    )

    style.configure(
        "TButton",
        background=COLORS["panel"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        padding=(10, 6),
    )
    style.map(
        "TButton",
        background=[("active", "#eceff3")],
    )

    style.configure(
        "Accent.TButton",
        background=COLORS["accent"],
        foreground=COLORS["accent_text"],
        bordercolor=COLORS["accent"],
        padding=(14, 8),
        font=FONTS["bold"],
    )
    style.map(
        "Accent.TButton",
        background=[("active", COLORS["accent_active"]), ("disabled", "#a9b8d6")],
        foreground=[("disabled", "#eef1f8")],
    )


def _normalize_path(value: str | None) -> str:
    return value.strip() if value else ""


def open_path_in_file_browser(path: str | Path) -> None:
    target = str(Path(path))
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])
        return
    subprocess.Popen(["xdg-open", target])


def build_gui_run_config(
    *,
    data_path: str,
    covariates_path: str,
    output_dir: str,
    subject_id_column: str,
    timepoint_column: str,
    batch_column: str,
    selected_features: list[str],
    selected_covariates: list[str],
    report_name: str,
    covariates_subject_id_column: str | None,
    covariates_timepoint_column: str | None,
) -> LongitudinalRunConfig:
    data_path = _normalize_path(data_path)
    covariates_path = _normalize_path(covariates_path)
    output_dir = _normalize_path(output_dir)
    subject_id_column = _normalize_path(subject_id_column)
    timepoint_column = _normalize_path(timepoint_column)
    batch_column = _normalize_path(batch_column)
    report_name = _normalize_path(report_name)

    if not data_path:
        raise ValueError("Please choose a data file.")
    if not output_dir:
        output_dir = str(Path.cwd())
    if not subject_id_column:
        raise ValueError("Please choose the subject ID column for the data file.")
    if not timepoint_column:
        raise ValueError("Please choose the timepoint column for the data file.")
    if not batch_column:
        raise ValueError("Please choose the batch column for the data file.")
    if not selected_features:
        raise ValueError("Please select at least one feature column.")

    covariates_path_value = covariates_path or None

    if covariates_path_value is not None and selected_covariates:
        if not covariates_subject_id_column:
            raise ValueError(
                "Please choose the subject ID column for the separate covariates file."
            )
        if not covariates_timepoint_column:
            raise ValueError(
                "Please choose the timepoint column for the separate covariates file."
            )

    reserved_columns = {
        subject_id_column,
        timepoint_column,
        batch_column,
    }

    if covariates_path_value is not None:
        reserved_columns.update(
            {
                _normalize_path(covariates_subject_id_column),
                _normalize_path(covariates_timepoint_column),
            }
        )

    feature_overlap = sorted(set(selected_features) & reserved_columns)
    if feature_overlap:
        raise ValueError(
            "Feature selection includes required structural columns: "
            f"{feature_overlap}"
        )

    covariate_overlap = sorted(set(selected_covariates) & reserved_columns)
    if covariate_overlap:
        raise ValueError(
            "Covariate selection includes required structural columns: "
            f"{covariate_overlap}"
        )

    overlap = sorted(set(selected_features) & set(selected_covariates))
    if overlap:
        raise ValueError(
            "Feature and covariate selections must be disjoint. Overlapping columns: "
            f"{overlap}"
        )

    return LongitudinalRunConfig(
        data_path=data_path,
        subject_id_column=subject_id_column,
        timepoint_column=timepoint_column,
        batch_column=batch_column,
        selected_features=selected_features,
        covariates_path=covariates_path_value,
        covariates_subject_id_column=_normalize_path(covariates_subject_id_column) or None,
        covariates_timepoint_column=_normalize_path(covariates_timepoint_column) or None,
        selected_covariates=selected_covariates,
        output_dir=output_dir,
        report_name=report_name or None,
        save_data=False,
        timestamped_reports=True,
    )


# --------------------------------------------------------------------------
# A fixed-height, internally-scrollable column picker.
#
# The list itself can scroll (so it works whether someone has 5 features or
# 500), but its *footprint* in the window never changes size, so the outer
# window never needs to scroll to reveal the rest of the GUI.
# --------------------------------------------------------------------------

class ScrollableColumnList(ttk.Frame):
    def __init__(self, parent: tk.Widget, height_rows: int = 9):
        super().__init__(parent, style="Panel.TFrame")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(
            self,
            selectmode="multiple",
            exportselection=False,
            height=height_rows,
            background=COLORS["panel"],
            foreground=COLORS["text"],
            selectbackground=COLORS["accent"],
            selectforeground=COLORS["accent_text"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"],
            relief="flat",
            font=FONTS["base"],
            activestyle="none",
        )
        self.listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

    def set_items(self, items: list[str]) -> None:
        self.listbox.delete(0, tk.END)
        for name in items:
            self.listbox.insert(tk.END, name)

    def select_all(self) -> None:
        self.listbox.select_set(0, tk.END)

    def clear(self) -> None:
        self.listbox.selection_clear(0, tk.END)

    def get_selected(self) -> list[str]:
        return [self.listbox.get(i) for i in self.listbox.curselection()]


class LongitudinalGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DiagnoseHarmonisation — Longitudinal Batch Effect Diagnostic Report Generation ")

        self.status_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.last_output_dir: Path | None = None
        self.last_report_path: Path | None = None
        self.current_data_columns: list[str] = []
        self.current_covariate_columns: list[str] = []
        self._status_lines: list[str] = []
        self._max_status_lines = 10

        self.data_path_var = tk.StringVar()
        self.covariates_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.cwd()))
        self.subject_id_var = tk.StringVar()
        self.timepoint_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.cov_subject_id_var = tk.StringVar()
        self.cov_timepoint_var = tk.StringVar()
        self.report_name_var = tk.StringVar()
        self.cov_source_label_var = tk.StringVar(value="Covariate columns — source: data file")

        _configure_style(self.root)
        self._build_layout()
        self.root.after(150, self._drain_status_queue)
        self.root.after(0, self._set_static_window_size)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(4, weight=1)  # selectors row absorbs extra space

        header = ttk.Label(
            outer,
            text="Longitudinal Batch Effect Diagnostic Report Generation",
            style="OnBg.TLabel",
            font=(FONT_FAMILY, 15, "bold"),
        )
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))
        subheader = ttk.Label(
            outer,
            text="Configure inputs, choose columns, then generate a longitudinal harmonisation report.",
            style="HintOnBg.TLabel",
        )
        subheader.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # Files
        inputs_frame = ttk.LabelFrame(outer, text="Files", padding=12)
        inputs_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        inputs_frame.columnconfigure(1, weight=1)

        ttk.Label(inputs_frame, text="Data file").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(inputs_frame, textvariable=self.data_path_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 8), pady=5
        )
        ttk.Button(inputs_frame, text="Browse…", command=self._browse_data_file).grid(
            row=0, column=2, pady=5
        )

        ttk.Label(inputs_frame, text="Covariates file").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(inputs_frame, textvariable=self.covariates_path_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 8), pady=5
        )
        ttk.Button(inputs_frame, text="Browse…", command=self._browse_covariates_file).grid(
            row=1, column=2, pady=5
        )

        ttk.Label(inputs_frame, text="Output directory").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inputs_frame, textvariable=self.output_dir_var).grid(
            row=2, column=1, sticky="ew", padx=(8, 8), pady=5
        )
        ttk.Button(inputs_frame, text="Browse…", command=self._browse_output_dir).grid(
            row=2, column=2, pady=5
        )

        controls_row = ttk.Frame(inputs_frame, style="Panel.TFrame")
        controls_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(controls_row, text="Load Columns", command=self._load_columns).pack(side="left")
        ttk.Label(
            controls_row,
            text="Load the file(s) before choosing columns below.",
            style="Hint.TLabel",
        ).pack(side="left", padx=(10, 0))

        # Required columns
        columns_frame = ttk.LabelFrame(outer, text="Required columns", padding=12)
        columns_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        columns_frame.columnconfigure(1, weight=1)
        columns_frame.columnconfigure(3, weight=1)

        ttk.Label(columns_frame, text="Data subject ID").grid(row=0, column=0, sticky="w", pady=5)
        self.subject_id_combo = ttk.Combobox(
            columns_frame, textvariable=self.subject_id_var, state="readonly"
        )
        self.subject_id_combo.grid(row=0, column=1, sticky="ew", padx=(8, 16), pady=5)

        ttk.Label(columns_frame, text="Timepoint").grid(row=0, column=2, sticky="w", pady=5)
        self.timepoint_combo = ttk.Combobox(
            columns_frame, textvariable=self.timepoint_var, state="readonly"
        )
        self.timepoint_combo.grid(row=0, column=3, sticky="ew", padx=(8, 0), pady=5)

        ttk.Label(columns_frame, text="Batch").grid(row=1, column=0, sticky="w", pady=5)
        self.batch_combo = ttk.Combobox(
            columns_frame, textvariable=self.batch_var, state="readonly"
        )
        self.batch_combo.grid(row=1, column=1, sticky="ew", padx=(8, 16), pady=5)

        ttk.Label(columns_frame, text="Report name").grid(row=1, column=2, sticky="w", pady=5)
        ttk.Entry(columns_frame, textvariable=self.report_name_var).grid(
            row=1, column=3, sticky="ew", padx=(8, 0), pady=5
        )

        # Feature / covariate selection — fixed-footprint panels with their
        # own internal scrollbars, so long column lists never grow the
        # outer window.
        selectors_frame = ttk.Frame(outer)
        selectors_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        selectors_frame.columnconfigure(0, weight=1)
        selectors_frame.columnconfigure(1, weight=1)
        selectors_frame.rowconfigure(0, weight=1)

        feature_frame = ttk.LabelFrame(selectors_frame, text="Feature columns", padding=12)
        feature_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        feature_frame.columnconfigure(0, weight=1)

        ttk.Label(
            feature_frame,
            text="Select one or more feature / IDP columns from the data file.",
            style="Hint.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.feature_grid = ScrollableColumnList(feature_frame, height_rows=9)
        self.feature_grid.grid(row=1, column=0, sticky="nsew")
        feature_frame.rowconfigure(1, weight=1)

        feature_buttons = ttk.Frame(feature_frame, style="Panel.TFrame")
        feature_buttons.grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Button(feature_buttons, text="Select All", command=self._select_all_features).pack(side="left")
        ttk.Button(feature_buttons, text="Clear", command=self._clear_features).pack(
            side="left", padx=(8, 0)
        )

        self.cov_frame = ttk.LabelFrame(selectors_frame, text=self.cov_source_label_var.get(), padding=12)
        self.cov_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.cov_frame.columnconfigure(0, weight=1)

        ttk.Label(
            self.cov_frame,
            text="Select one or more covariate columns from the chosen source.",
            style="Hint.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.covariate_grid = ScrollableColumnList(self.cov_frame, height_rows=9)
        self.covariate_grid.grid(row=1, column=0, sticky="nsew")
        self.cov_frame.rowconfigure(1, weight=1)

        cov_buttons = ttk.Frame(self.cov_frame, style="Panel.TFrame")
        cov_buttons.grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Button(cov_buttons, text="Select All", command=self._select_all_covariates).pack(side="left")
        ttk.Button(cov_buttons, text="Clear", command=self._clear_covariates).pack(side="left", padx=(8, 0))

        # Covariate key columns if separate file is used
        self.sep_cov_frame = ttk.LabelFrame(
            outer, text="Separate covariates file — alignment columns (These are only required when a separate covariates file is used.)", padding=12
        )
        self.sep_cov_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.sep_cov_frame.columnconfigure(1, weight=1)
        self.sep_cov_frame.columnconfigure(3, weight=1)

        ttk.Label(self.sep_cov_frame, text="Covariate subject ID").grid(row=0, column=0, sticky="w", pady=5)
        self.cov_subject_combo = ttk.Combobox(
            self.sep_cov_frame, textvariable=self.cov_subject_id_var, state="disabled"
        )
        self.cov_subject_combo.grid(row=0, column=1, sticky="ew", padx=(8, 16), pady=5)

        ttk.Label(self.sep_cov_frame, text="Covariate timepoint").grid(row=0, column=2, sticky="w", pady=5)
        self.cov_timepoint_combo = ttk.Combobox(
            self.sep_cov_frame, textvariable=self.cov_timepoint_var, state="disabled"
        )
        self.cov_timepoint_combo.grid(row=0, column=3, sticky="ew", padx=(8, 0), pady=5)

        # Actions
        actions_frame = ttk.Frame(outer)
        actions_frame.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        self.run_button = ttk.Button(
            actions_frame,
            text="Generate Longitudinal Report",
            style="Accent.TButton",
            command=self._start_run,
        )
        self.run_button.pack(side="left")

        self.open_folder_button = ttk.Button(
            actions_frame,
            text="Open Output Folder",
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_folder_button.pack(side="left", padx=(10, 0))

        ttk.Label(
            actions_frame,
            text="Note: save-data is disabled in the GUI to avoid the current covariate handling mismatch.",
            style="HintOnBg.TLabel",
        ).pack(side="left", padx=(16, 0))

        # Status
        status_frame = ttk.LabelFrame(outer, text="Status", padding=12)
        status_frame.grid(row=7, column=0, columnspan=2, sticky="ew")
        status_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            status_frame,
            text="Choose the data file and optional covariates file, then load columns.",
            font=FONTS["mono"],
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # Window sizing — set once, stays fixed regardless of how many
    # columns are loaded (the column lists scroll internally instead).
    # ------------------------------------------------------------------

    def _set_static_window_size(self) -> None:
        self.root.update_idletasks()
        req_w = self.root.winfo_reqwidth()
        req_h = self.root.winfo_reqheight()

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        margin = 80
        width = min(max(req_w, 1180), screen_w - margin)
        height = min(max(req_h, 840), screen_h - margin)

        self.root.geometry(f"{int(width)}x{int(height)}")
        self.root.minsize(int(min(width, 1100)), int(min(height, 760)))

    # ------------------------------------------------------------------
    # File / column handling
    # ------------------------------------------------------------------

    def _browse_data_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose data file",
            filetypes=[("Supported files", "*.csv *.xls *.xlsx"), ("All files", "*.*")],
        )
        if path:
            self.data_path_var.set(path)
            self._load_columns()

    def _browse_covariates_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose covariates file",
            filetypes=[("Supported files", "*.csv *.xls *.xlsx"), ("All files", "*.*")],
        )
        if path:
            self.covariates_path_var.set(path)
            self._load_columns()
        else:
            self._update_covariate_source_state()

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output directory")
        if path:
            self.output_dir_var.set(path)

    def _load_columns(self) -> None:
        data_path = _normalize_path(self.data_path_var.get())
        cov_path = _normalize_path(self.covariates_path_var.get()) or None

        if not data_path:
            self._append_status("Choose a data file before loading columns.")
            return

        try:
            summary = inspect_longitudinal_inputs(data_path, cov_path)
        except Exception as exc:
            messagebox.showerror("Load Columns", str(exc))
            self._append_status(f"Error loading columns: {exc}")
            return

        self.current_data_columns = summary.data_columns
        self.current_covariate_columns = summary.covariates_columns

        self.subject_id_combo["values"] = summary.data_columns
        self.timepoint_combo["values"] = summary.data_columns
        self.batch_combo["values"] = summary.data_columns

        self.subject_id_var.set("")
        self.timepoint_var.set("")
        self.batch_var.set("")
        self.report_name_var.set(self.report_name_var.get().strip())

        self.feature_grid.set_items(summary.data_columns)

        self._update_covariate_source_state()

        if cov_path is not None:
            self.covariate_grid.set_items(summary.covariates_columns)
            self._append_status("Loaded columns. Covariate list is sourced from the separate covariates file.")
        else:
            self.covariate_grid.set_items(summary.data_columns)
            self._append_status("Loaded columns. Covariate list is sourced from the data file.")

    def _set_cov_frame_title(self, title: str) -> None:
        self.cov_source_label_var.set(title)
        try:
            self.cov_frame.configure(text=title)
        except Exception:
            pass

    def _update_covariate_source_state(self) -> None:
        cov_path = _normalize_path(self.covariates_path_var.get())
        if cov_path:
            self._set_cov_frame_title("Covariate columns — source: separate covariates file")
            self.cov_subject_combo.configure(state="readonly")
            self.cov_timepoint_combo.configure(state="readonly")
            self.cov_subject_combo["values"] = self.current_covariate_columns
            self.cov_timepoint_combo["values"] = self.current_covariate_columns
            self.sep_cov_frame.grid()
        else:
            self._set_cov_frame_title("Covariate columns — source: data file")
            self.cov_subject_combo.set("")
            self.cov_timepoint_combo.set("")
            self.cov_subject_combo.configure(state="disabled")
            self.cov_timepoint_combo.configure(state="disabled")
            self.cov_subject_combo["values"] = []
            self.cov_timepoint_combo["values"] = []
            self.sep_cov_frame.grid_remove()

    def _select_all_features(self) -> None:
        self.feature_grid.select_all()

    def _clear_features(self) -> None:
        self.feature_grid.clear()

    def _select_all_covariates(self) -> None:
        self.covariate_grid.select_all()

    def _clear_covariates(self) -> None:
        self.covariate_grid.clear()

    # ------------------------------------------------------------------
    # Run handling
    # ------------------------------------------------------------------

    def _start_run(self) -> None:
        try:
            config = build_gui_run_config(
                data_path=self.data_path_var.get(),
                covariates_path=self.covariates_path_var.get(),
                output_dir=self.output_dir_var.get(),
                subject_id_column=self.subject_id_var.get(),
                timepoint_column=self.timepoint_var.get(),
                batch_column=self.batch_var.get(),
                selected_features=self.feature_grid.get_selected(),
                selected_covariates=self.covariate_grid.get_selected(),
                report_name=self.report_name_var.get(),
                covariates_subject_id_column=self.cov_subject_id_var.get(),
                covariates_timepoint_column=self.cov_timepoint_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.last_output_dir = Path(config.output_dir) if config.output_dir else None
        self.last_report_path = None
        self._append_status("Starting longitudinal report generation...")

        worker = threading.Thread(
            target=self._run_in_background,
            args=(config,),
            daemon=True,
        )
        worker.start()

    def _run_in_background(self, config: LongitudinalRunConfig) -> None:
        try:
            result = run_longitudinal_report(
                config,
                status_callback=lambda message: self.status_queue.put(("status", message)),
            )
        except Exception as exc:
            self.status_queue.put(("error", str(exc)))
            return

        self.status_queue.put(("success", result))

    def _drain_status_queue(self) -> None:
        try:
            while True:
                kind, payload = self.status_queue.get_nowait()
                if kind == "status":
                    self._append_status(str(payload))
                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self._append_status(f"Error: {payload}")
                    messagebox.showerror("Report Generation Failed", str(payload))
                elif kind == "success":
                    self.run_button.configure(state="normal")
                    result = payload
                    self.last_report_path = result.report_path
                    self.last_output_dir = result.save_dir
                    self.open_folder_button.configure(state="normal")

                    if result.report_path is not None:
                        success_message = (
                            f"Report generation completed successfully.\n"
                            f"Report: {result.report_path}"
                        )
                    else:
                        success_message = "Report generation completed successfully."

                    self._append_status("-" * 60)
                    self._append_status(success_message)

                    # Force the window to redraw and come to the front so the
                    # completed status is actually visible, then show a clear
                    # confirmation popup.
                    self.root.update_idletasks()
                    self.root.deiconify()
                    self.root.lift()
                    self.root.focus_force()
                    messagebox.showinfo("Report Generation Complete", success_message)
        except queue.Empty:
            pass

        self.root.after(150, self._drain_status_queue)

    def _append_status(self, message: str) -> None:
        # No scrollbar: keep only the last N lines visible, newest at the bottom.
        for line in message.rstrip().splitlines() or [""]:
            self._status_lines.append(line)
        self._status_lines = self._status_lines[-self._max_status_lines :]
        self.status_label.configure(text="\n".join(self._status_lines))

    def _open_output_folder(self) -> None:
        if self.last_output_dir is None:
            return
        open_path_in_file_browser(self.last_output_dir)


def launch_longitudinal_gui() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "Unable to launch the longitudinal GUI. A graphical display is required."
        ) from exc

    app = LongitudinalGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_longitudinal_gui()
