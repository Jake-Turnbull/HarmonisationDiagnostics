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

from DiagnoseHarmonisation.cross_sectional_workflow import (
    CrossSectionalRunConfig,
    inspect_cross_sectional_inputs,
    run_cross_sectional_report,
)


AUTO_DETECT_BATCH = "[Auto-detect batch column]"
NO_BATCH_COLUMN = "[No batch column]"


def build_gui_run_config(
    *,
    data_path: str,
    covariates_path: str,
    output_dir: str,
    data_id_column: str,
    covariates_id_column: str,
    batch_selection: str,
    selected_covariates: list[str],
    report_name: str,
    save_data: bool,
    timestamped_reports: bool,
) -> CrossSectionalRunConfig:
    if not data_path:
        raise ValueError("Please choose a data file.")
    if not covariates_path:
        raise ValueError("Please choose a covariates file.")
    if not output_dir:
        raise ValueError("Please choose an output directory.")
    if not data_id_column:
        raise ValueError("Please choose the subject ID column for the data file.")
    if not covariates_id_column:
        raise ValueError("Please choose the subject ID column for the covariates file.")
    if not selected_covariates:
        raise ValueError("Please select at least one covariate column.")

    batch_mode = "selected"
    batch_column: str | None = batch_selection
    if batch_selection == AUTO_DETECT_BATCH:
        batch_mode = "auto"
        batch_column = None
    elif batch_selection == NO_BATCH_COLUMN:
        batch_mode = "none"
        batch_column = None
    elif not batch_selection:
        raise ValueError("Please choose a batch column or explicitly select no batch column.")

    clean_report_name = report_name.strip() or None

    return CrossSectionalRunConfig(
        data_path=data_path,
        covariates_path=covariates_path,
        data_id_column=data_id_column,
        covariates_id_column=covariates_id_column,
        batch_mode=batch_mode,
        batch_column=batch_column,
        selected_covariates=selected_covariates,
        output_dir=output_dir,
        report_name=clean_report_name,
        save_data=save_data,
        timestamped_reports=timestamped_reports,
        allow_missing_batch_in_auto=False,
    )


class CrossSectionalGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DiagnoseHarmonisation Cross-Sectional GUI")
        self.root.minsize(900, 700)

        self.status_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_summary = None
        self.last_output_dir: Path | None = None
        self.last_report_path: Path | None = None

        self.data_path_var = tk.StringVar()
        self.covariates_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.cwd()))
        self.data_id_var = tk.StringVar()
        self.covariates_id_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.report_name_var = tk.StringVar()
        self.save_data_var = tk.BooleanVar(value=True)
        self.timestamped_var = tk.BooleanVar(value=True)

        self._build_layout()
        self.root.after(150, self._drain_status_queue)

    def _build_layout(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        frame.rowconfigure(8, weight=1)

        ttk.Label(frame, text="Data file").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(frame, textvariable=self.data_path_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(frame, text="Browse", command=self._browse_data_file).grid(row=0, column=2, padx=(8, 0), pady=(0, 8))

        ttk.Label(frame, text="Covariates file").grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(frame, textvariable=self.covariates_path_var).grid(row=1, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(frame, text="Browse", command=self._browse_covariates_file).grid(row=1, column=2, padx=(8, 0), pady=(0, 8))

        ttk.Label(frame, text="Output directory").grid(row=2, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(frame, textvariable=self.output_dir_var).grid(row=2, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(frame, text="Browse", command=self._browse_output_dir).grid(row=2, column=2, padx=(8, 0), pady=(0, 8))

        controls_row = ttk.Frame(frame)
        controls_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 8))
        ttk.Button(controls_row, text="Load Columns", command=self._load_columns).pack(side="left")
        ttk.Label(
            controls_row,
            text="Load files to populate subject ID, batch, and covariate options.",
        ).pack(side="left", padx=(12, 0))

        ttk.Label(frame, text="Data subject ID column").grid(row=4, column=0, sticky="w", pady=(0, 8))
        self.data_id_combo = ttk.Combobox(frame, textvariable=self.data_id_var, state="readonly")
        self.data_id_combo.grid(row=4, column=1, sticky="ew", pady=(0, 8))

        ttk.Label(frame, text="Covariates subject ID column").grid(row=5, column=0, sticky="w", pady=(0, 8))
        self.covariates_id_combo = ttk.Combobox(frame, textvariable=self.covariates_id_var, state="readonly")
        self.covariates_id_combo.grid(row=5, column=1, sticky="ew", pady=(0, 8))

        ttk.Label(frame, text="Batch column").grid(row=6, column=0, sticky="w", pady=(0, 8))
        self.batch_combo = ttk.Combobox(frame, textvariable=self.batch_var, state="readonly")
        self.batch_combo.grid(row=6, column=1, sticky="ew", pady=(0, 8))

        ttk.Label(frame, text="Report name").grid(row=7, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(frame, textvariable=self.report_name_var).grid(row=7, column=1, sticky="ew", pady=(0, 8))

        covariates_frame = ttk.LabelFrame(frame, text="Covariates to include", padding=8)
        covariates_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(8, 8))
        covariates_frame.columnconfigure(0, weight=1)
        covariates_frame.rowconfigure(0, weight=1)
        self.covariates_listbox = tk.Listbox(
            covariates_frame,
            selectmode="multiple",
            exportselection=False,
            height=8,
        )
        self.covariates_listbox.grid(row=0, column=0, sticky="nsew")
        covariates_scroll = ttk.Scrollbar(
            covariates_frame,
            orient="vertical",
            command=self.covariates_listbox.yview,
        )
        covariates_scroll.grid(row=0, column=1, sticky="ns")
        self.covariates_listbox.configure(yscrollcommand=covariates_scroll.set)

        covariate_buttons = ttk.Frame(covariates_frame)
        covariate_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(covariate_buttons, text="Select All", command=self._select_all_covariates).pack(side="left")
        ttk.Button(covariate_buttons, text="Clear", command=self._clear_covariates).pack(side="left", padx=(8, 0))

        options_frame = ttk.Frame(frame)
        options_frame.grid(row=9, column=0, columnspan=3, sticky="w", pady=(4, 8))
        ttk.Checkbutton(options_frame, text="Save aligned data", variable=self.save_data_var).pack(side="left")
        ttk.Checkbutton(options_frame, text="Timestamp report name", variable=self.timestamped_var).pack(side="left", padx=(16, 0))

        actions_frame = ttk.Frame(frame)
        actions_frame.grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 8))
        self.run_button = ttk.Button(actions_frame, text="Generate Cross-Sectional Report", command=self._start_run)
        self.run_button.pack(side="left")
        self.open_folder_button = ttk.Button(
            actions_frame,
            text="Open Output Folder",
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_folder_button.pack(side="left", padx=(8, 0))

        status_frame = ttk.LabelFrame(frame, text="Status", padding=8)
        status_frame.grid(row=11, column=0, columnspan=3, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.status_text = ScrolledText(status_frame, height=12, wrap="word", state="disabled")
        self.status_text.grid(row=0, column=0, sticky="nsew")

    def _browse_data_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose data file",
            filetypes=[("Supported files", "*.csv *.xls *.xlsx"), ("All files", "*.*")],
        )
        if path:
            self.data_path_var.set(path)
            self._maybe_load_columns()

    def _browse_covariates_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose covariates file",
            filetypes=[("Supported files", "*.csv *.xls *.xlsx"), ("All files", "*.*")],
        )
        if path:
            self.covariates_path_var.set(path)
            self._maybe_load_columns()

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output directory")
        if path:
            self.output_dir_var.set(path)

    def _maybe_load_columns(self) -> None:
        if self.data_path_var.get() and self.covariates_path_var.get():
            self._load_columns()

    def _load_columns(self) -> None:
        try:
            summary = inspect_cross_sectional_inputs(
                self.data_path_var.get(),
                self.covariates_path_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Load Columns", str(exc))
            self._append_status(f"Error loading columns: {exc}")
            return

        self.current_summary = summary
        self.data_id_combo["values"] = summary.data_columns
        self.covariates_id_combo["values"] = summary.covariates_columns
        self.data_id_var.set(summary.default_data_id_column or "")
        self.covariates_id_var.set(summary.default_covariates_id_column or "")

        batch_options = [AUTO_DETECT_BATCH] + summary.covariates_columns + [NO_BATCH_COLUMN]
        self.batch_combo["values"] = batch_options
        if summary.auto_batch_column:
            self.batch_var.set(AUTO_DETECT_BATCH)
            self._append_status(
                f"Loaded columns. Auto-detect will use '{summary.auto_batch_column}' as the batch column."
            )
        else:
            self.batch_var.set("")
            self._append_status(
                "Loaded columns. No batch-like column was detected automatically; choose a batch column or explicitly select no batch column."
            )

        self.covariates_listbox.delete(0, tk.END)
        for column in summary.default_covariate_columns:
            self.covariates_listbox.insert(tk.END, column)
        self._select_all_covariates()

    def _select_all_covariates(self) -> None:
        self.covariates_listbox.select_set(0, tk.END)

    def _clear_covariates(self) -> None:
        self.covariates_listbox.selection_clear(0, tk.END)

    def _selected_covariates(self) -> list[str]:
        return [
            self.covariates_listbox.get(index)
            for index in self.covariates_listbox.curselection()
        ]

    def _start_run(self) -> None:
        try:
            config = build_gui_run_config(
                data_path=self.data_path_var.get().strip(),
                covariates_path=self.covariates_path_var.get().strip(),
                output_dir=self.output_dir_var.get().strip(),
                data_id_column=self.data_id_var.get().strip(),
                covariates_id_column=self.covariates_id_var.get().strip(),
                batch_selection=self.batch_var.get().strip(),
                selected_covariates=self._selected_covariates(),
                report_name=self.report_name_var.get(),
                save_data=self.save_data_var.get(),
                timestamped_reports=self.timestamped_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.last_output_dir = Path(config.output_dir)
        self.last_report_path = None
        self._append_status("Starting report generation...")

        worker = threading.Thread(
            target=self._run_in_background,
            args=(config,),
            daemon=True,
        )
        worker.start()

    def _run_in_background(self, config: CrossSectionalRunConfig) -> None:
        try:
            result = run_cross_sectional_report(
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
                    self.last_output_dir = result.save_dir
                    self.last_report_path = result.report_path
                    if self.last_output_dir is not None:
                        self.open_folder_button.configure(state="normal")
                    if result.report_path is not None:
                        self._append_status(f"Completed successfully. Report: {result.report_path}")
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


def open_path_in_file_browser(path: str | Path) -> None:
    target = str(Path(path))
    if sys.platform.startswith("win"):
        os.startfile(target)
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])
        return
    subprocess.Popen(["xdg-open", target])


def launch_cross_sectional_gui() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "Unable to launch the desktop GUI. A graphical display is required."
        ) from exc

    app = CrossSectionalGuiApp(root)
    app._append_status("Choose data and covariates files to begin.")
    root.mainloop()
