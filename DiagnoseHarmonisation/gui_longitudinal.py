#!/usr/bin/env python3
"""
Longitudinal GUI for DiagnoseHarmonisation.

Design goals:
- No auto-detection in the run path
- Explicit subject ID, timepoint, batch, features
- Optional covariates from the same data file or a separate covariates file
- Tkinter-based UI similar in spirit to the cross-sectional GUI
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
from tkinter.scrolledtext import ScrolledText

from DiagnoseHarmonisation.longitudinal_workflow import (
    LongitudinalRunConfig,
    inspect_longitudinal_inputs,
    run_longitudinal_report,
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
    if not selected_covariates:
        raise ValueError("Please select at least one covariate column.")

    covariates_path_value = covariates_path or None

    if covariates_path_value is not None:
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


class LongitudinalGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DiagnoseHarmonisation Longitudinal GUI")
        self.root.minsize(1000, 760)

        self.status_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.last_output_dir: Path | None = None
        self.last_report_path: Path | None = None
        self.current_data_columns: list[str] = []
        self.current_covariate_columns: list[str] = []

        self.data_path_var = tk.StringVar()
        self.covariates_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.cwd()))
        self.subject_id_var = tk.StringVar()
        self.timepoint_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.cov_subject_id_var = tk.StringVar()
        self.cov_timepoint_var = tk.StringVar()
        self.report_name_var = tk.StringVar()
        self.cov_source_label_var = tk.StringVar(
            value="Covariate columns source: data file"
        )

        self._build_layout()
        self.root.after(150, self._drain_status_queue)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container.columnconfigure(1, weight=1)
        container.rowconfigure(6, weight=1)
        container.rowconfigure(7, weight=1)

        # Inputs
        inputs_frame = ttk.LabelFrame(container, text="Files", padding=10)
        inputs_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        inputs_frame.columnconfigure(1, weight=1)

        ttk.Label(inputs_frame, text="Data file").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(inputs_frame, textvariable=self.data_path_var).grid(
            row=0, column=1, sticky="ew", pady=4
        )
        ttk.Button(inputs_frame, text="Browse", command=self._browse_data_file).grid(
            row=0, column=2, padx=(8, 0), pady=4
        )

        ttk.Label(inputs_frame, text="Covariates file").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(inputs_frame, textvariable=self.covariates_path_var).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        ttk.Button(inputs_frame, text="Browse", command=self._browse_covariates_file).grid(
            row=1, column=2, padx=(8, 0), pady=4
        )

        ttk.Label(inputs_frame, text="Output directory").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(inputs_frame, textvariable=self.output_dir_var).grid(
            row=2, column=1, sticky="ew", pady=4
        )
        ttk.Button(inputs_frame, text="Browse", command=self._browse_output_dir).grid(
            row=2, column=2, padx=(8, 0), pady=4
        )

        controls_row = ttk.Frame(inputs_frame)
        controls_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(controls_row, text="Load Columns", command=self._load_columns).pack(
            side="left"
        )
        ttk.Label(
            controls_row,
            text="Load the file(s) before choosing columns.",
        ).pack(side="left", padx=(12, 0))

        # Required columns
        columns_frame = ttk.LabelFrame(container, text="Required columns", padding=10)
        columns_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        columns_frame.columnconfigure(1, weight=1)
        columns_frame.columnconfigure(3, weight=1)

        ttk.Label(columns_frame, text="Data subject ID").grid(row=0, column=0, sticky="w", pady=4)
        self.subject_id_combo = ttk.Combobox(
            columns_frame, textvariable=self.subject_id_var, state="readonly"
        )
        self.subject_id_combo.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(columns_frame, text="Timepoint").grid(row=0, column=2, sticky="w", padx=(16, 0), pady=4)
        self.timepoint_combo = ttk.Combobox(
            columns_frame, textvariable=self.timepoint_var, state="readonly"
        )
        self.timepoint_combo.grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(columns_frame, text="Batch").grid(row=1, column=0, sticky="w", pady=4)
        self.batch_combo = ttk.Combobox(
            columns_frame, textvariable=self.batch_var, state="readonly"
        )
        self.batch_combo.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(columns_frame, text="Report name").grid(row=1, column=2, sticky="w", padx=(16, 0), pady=4)
        ttk.Entry(columns_frame, textvariable=self.report_name_var).grid(
            row=1, column=3, sticky="ew", pady=4
        )

        # Features
        feature_frame = ttk.LabelFrame(container, text="Feature columns", padding=10)
        feature_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        feature_frame.columnconfigure(0, weight=1)
        feature_frame.rowconfigure(1, weight=1)

        ttk.Label(
            feature_frame,
            text="Select one or more feature / IDP columns from the data file.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.feature_listbox = tk.Listbox(
            feature_frame,
            selectmode="multiple",
            exportselection=False,
            height=10,
        )
        self.feature_listbox.grid(row=1, column=0, sticky="nsew")

        feature_scroll = ttk.Scrollbar(
            feature_frame,
            orient="vertical",
            command=self.feature_listbox.yview,
        )
        feature_scroll.grid(row=1, column=1, sticky="ns")
        self.feature_listbox.configure(yscrollcommand=feature_scroll.set)

        feature_buttons = ttk.Frame(feature_frame)
        feature_buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(feature_buttons, text="Select All", command=self._select_all_features).pack(
            side="left"
        )
        ttk.Button(feature_buttons, text="Clear", command=self._clear_features).pack(
            side="left", padx=(8, 0)
        )

        # Covariates
        cov_frame = ttk.LabelFrame(container, textvariable=self.cov_source_label_var, padding=10)
        cov_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        cov_frame.columnconfigure(0, weight=1)
        cov_frame.rowconfigure(1, weight=1)

        ttk.Label(
            cov_frame,
            text="Select one or more covariate columns from the chosen source.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.covariate_listbox = tk.Listbox(
            cov_frame,
            selectmode="multiple",
            exportselection=False,
            height=10,
        )
        self.covariate_listbox.grid(row=1, column=0, sticky="nsew")

        cov_scroll = ttk.Scrollbar(
            cov_frame,
            orient="vertical",
            command=self.covariate_listbox.yview,
        )
        cov_scroll.grid(row=1, column=1, sticky="ns")
        self.covariate_listbox.configure(yscrollcommand=cov_scroll.set)

        cov_buttons = ttk.Frame(cov_frame)
        cov_buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(cov_buttons, text="Select All", command=self._select_all_covariates).pack(
            side="left"
        )
        ttk.Button(cov_buttons, text="Clear", command=self._clear_covariates).pack(
            side="left", padx=(8, 0)
        )

        # Covariate key columns if separate file is used
        sep_cov_frame = ttk.LabelFrame(
            container,
            text="Separate covariates file alignment columns",
            padding=10,
        )
        sep_cov_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        sep_cov_frame.columnconfigure(1, weight=1)
        sep_cov_frame.columnconfigure(3, weight=1)

        ttk.Label(sep_cov_frame, text="Covariate subject ID").grid(row=0, column=0, sticky="w", pady=4)
        self.cov_subject_combo = ttk.Combobox(
            sep_cov_frame, textvariable=self.cov_subject_id_var, state="disabled"
        )
        self.cov_subject_combo.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(sep_cov_frame, text="Covariate timepoint").grid(row=0, column=2, sticky="w", padx=(16, 0), pady=4)
        self.cov_timepoint_combo = ttk.Combobox(
            sep_cov_frame, textvariable=self.cov_timepoint_var, state="disabled"
        )
        self.cov_timepoint_combo.grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(
            sep_cov_frame,
            text="These are only required when a separate covariates file is used.",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # Actions
        actions_frame = ttk.Frame(container)
        actions_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 10))

        self.run_button = ttk.Button(
            actions_frame,
            text="Generate Longitudinal Report",
            command=self._start_run,
        )
        self.run_button.pack(side="left")

        self.open_folder_button = ttk.Button(
            actions_frame,
            text="Open Output Folder",
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_folder_button.pack(side="left", padx=(8, 0))

        ttk.Label(
            actions_frame,
            text="Save-data is disabled in the GUI to avoid the current covariate handling mismatch.",
        ).pack(side="left", padx=(16, 0))

        # Status
        status_frame = ttk.LabelFrame(container, text="Status", padding=10)
        status_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.status_text = ScrolledText(status_frame, height=14, wrap="word", state="disabled")
        self.status_text.grid(row=0, column=0, sticky="nsew")

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

        self.feature_listbox.delete(0, tk.END)
        for column in summary.data_columns:
            self.feature_listbox.insert(tk.END, column)

        self._update_covariate_source_state()

        self.covariate_listbox.delete(0, tk.END)
        if cov_path is not None:
            for column in summary.covariates_columns:
                self.covariate_listbox.insert(tk.END, column)
            self._append_status(
                "Loaded columns. Covariate list is sourced from the separate covariates file."
            )
        else:
            for column in summary.data_columns:
                self.covariate_listbox.insert(tk.END, column)
            self._append_status(
                "Loaded columns. Covariate list is sourced from the data file."
            )

    def _update_covariate_source_state(self) -> None:
        cov_path = _normalize_path(self.covariates_path_var.get())
        if cov_path:
            self.cov_source_label_var.set("Covariate columns source: separate covariates file")
            self.cov_subject_combo.configure(state="readonly")
            self.cov_timepoint_combo.configure(state="readonly")
            self.cov_subject_combo["values"] = self.current_covariate_columns
            self.cov_timepoint_combo["values"] = self.current_covariate_columns
        else:
            self.cov_source_label_var.set("Covariate columns source: data file")
            self.cov_subject_combo.set("")
            self.cov_timepoint_combo.set("")
            self.cov_subject_combo.configure(state="disabled")
            self.cov_timepoint_combo.configure(state="disabled")
            self.cov_subject_combo["values"] = []
            self.cov_timepoint_combo["values"] = []

    def _select_all_features(self) -> None:
        self.feature_listbox.select_set(0, tk.END)

    def _clear_features(self) -> None:
        self.feature_listbox.selection_clear(0, tk.END)

    def _select_all_covariates(self) -> None:
        self.covariate_listbox.select_set(0, tk.END)

    def _clear_covariates(self) -> None:
        self.covariate_listbox.selection_clear(0, tk.END)

    def _selected_items(self, listbox: tk.Listbox) -> list[str]:
        return [listbox.get(index) for index in listbox.curselection()]

    def _start_run(self) -> None:
        try:
            config = build_gui_run_config(
                data_path=self.data_path_var.get(),
                covariates_path=self.covariates_path_var.get(),
                output_dir=self.output_dir_var.get(),
                subject_id_column=self.subject_id_var.get(),
                timepoint_column=self.timepoint_var.get(),
                batch_column=self.batch_var.get(),
                selected_features=self._selected_items(self.feature_listbox),
                selected_covariates=self._selected_items(self.covariate_listbox),
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
                        self._append_status(
                            f"Completed successfully. Report: {result.report_path}"
                        )
                    else:
                        self._append_status("Completed successfully.")
        except queue.Empty:
            pass

        self.root.after(150, self._drain_status_queue)

    def _append_status(self, message: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert(tk.END, message.rstrip() + "\n")
        self.status_text.see(tk.END)
        self.status_text.configure(state="disabled")

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
    app._append_status("Choose the data file and optional covariates file, then load columns.")
    root.mainloop()


if __name__ == "__main__":
    launch_longitudinal_gui()