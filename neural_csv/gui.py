"""Desktop window for training on a CSV or predicting with the saved network."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neural_csv.charts import actual_vs_predicted_figure, qa_figure
from neural_csv.engine import (
    comparison_for_rows,
    load_charts,
    load_csv,
    load_training_history,
    model_summary,
    parse_schema,
    predict,
    reprocess_saved_training,
    reverse_forecast,
    reset_system,
    train_model,
    training_state,
)

PREVIEW_ROWS = 12
RESULT_ROWS = 200


class CsvNetworkApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CSV Neural Network")
        self.geometry("980x700")
        self.minsize(760, 540)

        self.frame: pd.DataFrame | None = None
        self.csv_path: Path | None = None
        self.predictions: pd.DataFrame | None = None
        self.comparison: dict | None = None
        self.qa: dict | None = None
        self.graph_window: tk.Toplevel | None = None
        self._busy = False

        self._build()
        self._refresh_model_status()
        self._reprocess_on_startup()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(12, 12, 12, 4))
        header.grid(row=0, column=0, sticky="ew")

        ttk.Button(header, text="Select CSV file…", command=self._select_csv).grid(row=0, column=0, padx=(0, 8))
        self.graph_button = ttk.Button(header, text="Show graphs", command=self._open_graphs, state="disabled")
        self.graph_button.grid(row=0, column=1, padx=(0, 8))
        self.reset_button = ttk.Button(header, text="Reset system…", command=self._confirm_reset)
        self.reset_button.grid(row=0, column=2, padx=(0, 8))
        self.path_var = tk.StringVar(value="No file selected")
        ttk.Label(header, textvariable=self.path_var).grid(row=0, column=3, sticky="w")
        header.columnconfigure(3, weight=1)

        self.model_var = tk.StringVar()
        ttk.Label(header, textvariable=self.model_var, foreground="#333").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )
        saved = load_charts()
        if saved is not None:
            self.comparison, self.qa = saved

        self.body = ttk.Frame(self, padding=(12, 4, 12, 12))
        self.body.grid(row=2, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)

        intro = ttk.Label(
            self.body,
            text=(
                "CSV layout: column 1 = ID, column 2 = target, columns after that = prediction variables.\n"
                "Select a file, then choose training or prediction. Training rows are kept across restarts "
                "and processed again when the program runs. Reset system clears them only if you confirm."
            ),
            wraplength=900,
            justify="left",
        )
        intro.grid(row=0, column=0, sticky="nw")

    def _select_csv(self) -> None:
        if self._busy:
            return
        chosen = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not chosen:
            return
        try:
            frame = load_csv(chosen)
            parse_schema(frame)
        except Exception as error:  # noqa: BLE001 - show any read failure in the dialog
            messagebox.showerror("Could not read CSV", str(error))
            return

        self.frame = frame
        self.csv_path = Path(chosen)
        self.predictions = None
        self.path_var.set(str(self.csv_path))
        self._ask_mode()

    def _ask_mode(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("How should this file be used?")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        schema = parse_schema(self.frame)
        ttk.Label(
            dialog,
            text=(
                f"{self.csv_path.name}\n"
                f"{len(self.frame)} rows · ID={schema.id_column} · target={schema.target_column}\n"
                f"prediction variables: {len(schema.feature_columns)}"
            ),
            padding=(16, 16, 16, 8),
            justify="center",
        ).grid(row=0, column=0, columnspan=3)

        ttk.Label(
            dialog,
            text="Train, predict the target, or reverse-forecast each indicator?",
            padding=(16, 0, 16, 12),
        ).grid(row=1, column=0, columnspan=3)

        def choose(mode: str) -> None:
            dialog.destroy()
            if mode == "train":
                self._show_train()
            elif mode == "reverse":
                self._show_reverse()
            else:
                self._show_predict()

        ttk.Button(dialog, text="Use as training data", command=lambda: choose("train")).grid(
            row=2, column=0, padx=12, pady=(0, 16), ipadx=6, ipady=6
        )
        ttk.Button(dialog, text="Make predictions", command=lambda: choose("predict")).grid(
            row=2, column=1, padx=12, pady=(0, 16), ipadx=6, ipady=6
        )
        ttk.Button(dialog, text="Reverse forecast", command=lambda: choose("reverse")).grid(
            row=2, column=2, padx=12, pady=(0, 16), ipadx=6, ipady=6
        )
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        self._center(dialog)

    def _show_train(self) -> None:
        self._clear_body()
        frame = self.frame
        assert frame is not None
        schema = parse_schema(frame)

        controls = ttk.LabelFrame(self.body, text="Training", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="ID column (fixed)").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text=schema.id_column).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(controls, text="Target column (fixed)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(controls, text=schema.target_column).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(controls, text="Problem type").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.task_var = tk.StringVar(value="auto")
        task_box = ttk.Combobox(
            controls,
            textvariable=self.task_var,
            values=("auto", "classification", "regression"),
            state="readonly",
        )
        task_box.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(controls, text="Prediction variables").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        list_wrap = ttk.Frame(controls)
        list_wrap.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        self.feature_list = tk.Listbox(list_wrap, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        feature_columns = self._training_feature_choices(schema)
        for column in feature_columns:
            self.feature_list.insert(tk.END, column)
        self._select_default_features(feature_columns)
        self.feature_list.pack(side="left", fill="x", expand=True)

        dim_buttons = ttk.Frame(controls)
        dim_buttons.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Button(dim_buttons, text="Use all variables", command=self._select_all_features).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(dim_buttons, text="Clear variables", command=self._clear_features).grid(row=0, column=1)

        self.train_button = ttk.Button(controls, text="Train neural network", command=self._start_training)
        self.train_button.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self.result_text = tk.Text(self.body, height=8, wrap="word")
        self.result_text.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        retained = load_training_history()
        retained_text = (
            f" {len(retained)} training rows are already retained and will be kept."
            if retained is not None
            else ""
        )
        hint = (
            f"Target is always '{schema.target_column}' (column 2). "
            "Add or remove prediction variables below, or clear them to train from the target column alone."
            + retained_text
        )
        self.result_text.insert("1.0", hint)
        self.result_text.configure(state="disabled")

        self._show_table(frame.head(PREVIEW_ROWS), row=2)

    def _training_feature_choices(self, schema) -> list[str]:
        """Current-file variables plus any columns already stored from earlier runs."""
        history = load_training_history()
        state = training_state()
        if history is None or state is None:
            return list(schema.feature_columns)
        retained = [
            column
            for column in history.columns
            if column not in {state["id_column"], state["target_column"]}
        ]
        return list(dict.fromkeys([*schema.feature_columns, *retained]))

    def _select_default_features(self, columns: list[str]) -> None:
        state = training_state()
        history = load_training_history()
        history_columns = set() if history is None else set(history.columns)
        saved = None if state is None else set(state["features"])
        for index, column in enumerate(columns):
            is_new = column not in history_columns
            if saved is None or column in saved or is_new:
                self.feature_list.selection_set(index)

    def _select_all_features(self) -> None:
        self.feature_list.selection_set(0, tk.END)

    def _clear_features(self) -> None:
        self.feature_list.selection_clear(0, tk.END)

    def _start_training(self) -> None:
        if self._busy or self.frame is None:
            return
        features = [self.feature_list.get(index) for index in self.feature_list.curselection()]
        task = self.task_var.get()
        snapshot = self.frame.copy()
        self._set_busy(True)
        self._set_result("Training and running QA testing…")

        def work() -> None:
            try:
                result = train_model(snapshot, features, task)
            except Exception as error:  # noqa: BLE001 - surface training errors in the window
                self.after(0, lambda: self._training_failed(str(error)))
                return
            self.after(0, lambda: self._training_finished(result))

        threading.Thread(target=work, daemon=True).start()

    def _training_finished(self, result) -> None:
        self.comparison = result.comparison
        self.qa = result.qa
        self._set_busy(False)
        self._refresh_model_status()
        feature_text = ", ".join(result.features) if result.features else "(target column only)"
        lines = [
            f"Trained a {result.task} network to fill '{result.target}'.",
            f"ID column: {result.id_column}",
            f"Rows used: {result.rows}",
            f"Prediction variables: {feature_text}",
            f"Saved model: {result.model_path}",
        ]
        for name, value in result.metrics.items():
            lines.append(f"{name}: {value}")
        lines.extend(result.warnings)
        lines.append("Graphs: actual vs predicted, and QA testing (green = pass, red = fail).")
        self._set_result("\n".join(lines))
        self._open_graphs()

    def _training_failed(self, message: str) -> None:
        self._set_busy(False)
        self._set_result(message)
        messagebox.showerror("Training failed", message)

    def _show_predict(self) -> None:
        self._clear_body()
        summary = model_summary()
        if summary is None:
            ttk.Label(
                self.body,
                text="There is no saved model yet. Select a CSV and choose Use as training data first.",
                wraplength=860,
            ).grid(row=0, column=0, sticky="nw")
            return

        schema = parse_schema(self.frame)
        feature_text = ", ".join(summary["features"]) if summary["features"] else "(target column only)"
        controls = ttk.LabelFrame(self.body, text="Prediction", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            controls,
            text=(
                f"Saved {summary['task']} model fills column 2 ('{summary['target']}').\n"
                f"Prediction variables used by the model: {feature_text}\n"
                f"This file: ID={schema.id_column}, target={schema.target_column}"
            ),
            wraplength=860,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        buttons = ttk.Frame(controls)
        buttons.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.predict_button = ttk.Button(buttons, text="Run predictions", command=self._start_prediction)
        self.predict_button.grid(row=0, column=0, padx=(0, 8))
        self.save_button = ttk.Button(
            buttons, text="Save predictions CSV…", command=self._save_predictions, state="disabled"
        )
        self.save_button.grid(row=0, column=1)

        self.result_text = tk.Text(self.body, height=4, wrap="word")
        self.result_text.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self._set_result(
            "Run predictions to fill the target column (column 2). "
            "Leave that column blank in the CSV if you only need the model output."
        )
        self._show_table(self.frame.head(PREVIEW_ROWS), row=2)

    def _show_reverse(self) -> None:
        self._clear_body()
        summary = model_summary()
        if summary is None:
            ttk.Label(
                self.body,
                text="There is no saved model yet. Select a CSV and choose Use as training data first.",
                wraplength=860,
            ).grid(row=0, column=0, sticky="nw")
            return
        if not summary["features"]:
            ttk.Label(
                self.body,
                text="The saved training used no indicators, so reverse forecasting has nothing to predict.",
                wraplength=860,
            ).grid(row=0, column=0, sticky="nw")
            return

        schema = parse_schema(self.frame)
        controls = ttk.LabelFrame(self.body, text="Reverse forecast", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            controls,
            text=(
                f"Each indicator is predicted from '{summary['target']}' using models fit on retained training rows only.\n"
                f"Indicators: {', '.join(summary['features'])}\n"
                f"This file supplies the target in column 2 ('{schema.target_column}') and is not added to training."
            ),
            wraplength=860,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        buttons = ttk.Frame(controls)
        buttons.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.reverse_button = ttk.Button(buttons, text="Run reverse forecast", command=self._start_reverse)
        self.reverse_button.grid(row=0, column=0, padx=(0, 8))
        self.save_button = ttk.Button(
            buttons, text="Save predictions CSV…", command=self._save_predictions, state="disabled"
        )
        self.save_button.grid(row=0, column=1)

        self.result_text = tk.Text(self.body, height=4, wrap="word")
        self.result_text.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self._set_result("Put known target values in column 2. Indicator columns will be filled from the saved training data.")
        self._show_table(self.frame.head(PREVIEW_ROWS), row=2)

    def _start_reverse(self) -> None:
        if self._busy or self.frame is None:
            return
        snapshot = self.frame.copy()
        self._set_busy(True)
        self._set_result("Reverse forecasting each indicator from retained training data…")

        def work() -> None:
            try:
                result = reverse_forecast(snapshot)
            except Exception as error:  # noqa: BLE001 - surface reverse-forecast errors in the window
                self.after(0, lambda message=str(error): self._prediction_failed(message))
                return
            self.after(0, lambda result=result: self._reverse_finished(result))

        threading.Thread(target=work, daemon=True).start()

    def _reverse_finished(self, result: pd.DataFrame) -> None:
        self._set_busy(False)
        self.predictions = result
        self.save_button.configure(state="normal")
        shown = min(len(result), RESULT_ROWS)
        self._set_result(
            f"Filled {len(result.columns) - 2} indicator column(s) for {len(result)} row(s). "
            f"Showing the first {shown}. This file was not used to retrain the models."
        )
        self._show_table(result.head(RESULT_ROWS), row=2)

    def _start_prediction(self) -> None:
        if self._busy or self.frame is None:
            return
        snapshot = self.frame.copy()
        self._set_busy(True)
        self._set_result("Predicting…")

        def work() -> None:
            try:
                result = predict(snapshot)
            except Exception as error:  # noqa: BLE001 - surface prediction errors in the window
                self.after(0, lambda: self._prediction_failed(str(error)))
                return
            self.after(0, lambda result=result, snapshot=snapshot: self._prediction_finished(result, snapshot))

        threading.Thread(target=work, daemon=True).start()

    def _prediction_finished(self, result: pd.DataFrame, original: pd.DataFrame) -> None:
        self._set_busy(False)
        self.predictions = result
        self.save_button.configure(state="normal")
        schema = parse_schema(result)
        shown = min(len(result), RESULT_ROWS)
        message = f"Filled '{schema.target_column}' for {len(result)} row(s). Showing the first {shown}."
        compared = comparison_for_rows(original, result)
        if compared is not None and self.qa is not None:
            self.comparison = compared
            message += " Actual vs predicted now uses rows in this file that already had a target value."
            self._open_graphs()
        self._set_result(message)
        self._show_table(result.head(RESULT_ROWS), row=2)

    def _open_graphs(self) -> None:
        if self.comparison is None or self.qa is None:
            messagebox.showinfo("No graphs yet", "Train a model first. Graphs are created with the training QA check.")
            return
        window = self.graph_window
        if window is None or not window.winfo_exists():
            window = tk.Toplevel(self)
            window.title("Actual vs predicted and QA testing")
            window.geometry("1120x540")
            window.minsize(860, 420)
            window.columnconfigure(0, weight=1)
            window.columnconfigure(1, weight=1)
            window.rowconfigure(1, weight=1)
            ttk.Label(
                window,
                text="Left: actual values versus predicted values. Right: QA testing (green = pass, red = fail).",
                padding=(12, 8, 12, 4),
            ).grid(row=0, column=0, columnspan=2, sticky="w")
            window.left = ttk.Frame(window)
            window.right = ttk.Frame(window)
            window.left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
            window.right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
            window.canvases = []
            window.protocol("WM_DELETE_WINDOW", window.destroy)
            self.graph_window = window
        for canvas in window.canvases:
            canvas.get_tk_widget().destroy()
        window.canvases.clear()
        panels = (
            (window.left, actual_vs_predicted_figure(self.comparison)),
            (window.right, qa_figure(self.qa)),
        )
        for frame, figure in panels:
            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill="both", expand=True)
            window.canvases.append(canvas)
        window.deiconify()
        window.lift()

    def _prediction_failed(self, message: str) -> None:
        self._set_busy(False)
        self._set_result(message)
        messagebox.showerror("Prediction failed", message)

    def _save_predictions(self) -> None:
        if self.predictions is None:
            return
        default_name = "predictions.csv"
        if self.csv_path is not None:
            default_name = f"{self.csv_path.stem}_predictions.csv"
        chosen = filedialog.asksaveasfilename(
            title="Save predictions",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )
        if not chosen:
            return
        self.predictions.to_csv(chosen, index=False)
        self._set_result(f"Saved {len(self.predictions)} rows to {chosen}")

    def _show_table(self, frame: pd.DataFrame, row: int) -> None:
        if hasattr(self, "table_wrap") and self.table_wrap.winfo_exists():
            self.table_wrap.destroy()
        self.table_wrap = ttk.Frame(self.body)
        self.table_wrap.grid(row=row, column=0, sticky="nsew")
        self.body.rowconfigure(row, weight=1)
        self.table_wrap.columnconfigure(0, weight=1)
        self.table_wrap.rowconfigure(0, weight=1)

        columns = list(frame.columns)
        tree = ttk.Treeview(self.table_wrap, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=str(column))
            tree.column(column, width=120, anchor="w")
        for record in frame.itertuples(index=False):
            tree.insert("", tk.END, values=["" if pd.isna(value) else value for value in record])

        y_scroll = ttk.Scrollbar(self.table_wrap, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(self.table_wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

    def _clear_body(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()

    def _set_result(self, text: str) -> None:
        if not hasattr(self, "result_text") or not self.result_text.winfo_exists():
            return
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        if hasattr(self, "train_button") and self.train_button.winfo_exists():
            self.train_button.configure(state=state)
        if hasattr(self, "predict_button") and self.predict_button.winfo_exists():
            self.predict_button.configure(state=state)
        if hasattr(self, "reverse_button") and self.reverse_button.winfo_exists():
            self.reverse_button.configure(state=state)
        if hasattr(self, "reset_button") and self.reset_button.winfo_exists():
            self.reset_button.configure(state=state)
        if self.comparison is not None and self.qa is not None and not busy:
            self.graph_button.configure(state="normal")
        elif busy:
            self.graph_button.configure(state="disabled")

    def _refresh_model_status(self) -> None:
        summary = model_summary()
        if summary is None:
            self.model_var.set("No model saved yet.")
            return
        if self.comparison is not None and self.qa is not None:
            self.graph_button.configure(state="normal")
        retained = summary.get("retained_rows") or 0
        feature_text = ", ".join(summary["features"]) if summary["features"] else "target only"
        metric_bits = ", ".join(f"{key}={value}" for key, value in summary["metrics"].items())
        self.model_var.set(
            f"Saved model: {summary['task']} → {summary['target']} "
            f"[{feature_text}] ({metric_bits}). Retained training rows: {retained}."
        )

    def _reprocess_on_startup(self) -> None:
        if load_training_history() is None or training_state() is None:
            return
        self._set_busy(True)
        self.model_var.set("Reprocessing retained training data…")

        def work() -> None:
            try:
                result = reprocess_saved_training()
            except Exception as error:  # noqa: BLE001 - keep the saved rows and show the failure
                self.after(0, lambda message=str(error): self._reprocess_failed(message))
                return
            self.after(0, lambda result=result: self._reprocess_finished(result))

        threading.Thread(target=work, daemon=True).start()

    def _reprocess_finished(self, result) -> None:
        self._set_busy(False)
        if result is not None:
            self.comparison = result.comparison
            self.qa = result.qa
        self._refresh_model_status()
        if result is not None:
            self.model_var.set(self.model_var.get() + " Reprocessed on startup; previous rows were kept.")

    def _reprocess_failed(self, message: str) -> None:
        self._set_busy(False)
        self._refresh_model_status()
        messagebox.showerror(
            "Could not reprocess training data",
            f"{message}\n\nSaved training rows were kept. The system was not reset.",
        )

    def _confirm_reset(self) -> None:
        if self._busy:
            return
        agreed = messagebox.askyesno(
            "Reset system",
            "Delete all retained training rows and the saved model?\n\n"
            "Choose Yes only if you intend to clear the system. This cannot be undone.",
            icon="warning",
            default="no",
        )
        if not agreed:
            return
        reset_system()
        self.comparison = None
        self.qa = None
        self.graph_button.configure(state="disabled")
        if self.graph_window is not None and self.graph_window.winfo_exists():
            self.graph_window.destroy()
        self._refresh_model_status()
        messagebox.showinfo("Reset complete", "Retained training data and the saved model were deleted.")

    def _center(self, window: tk.Toplevel) -> None:
        window.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - window.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - window.winfo_height()) // 2
        window.geometry(f"+{x}+{y}")


def main() -> None:
    app = CsvNetworkApp()
    app.mainloop()
