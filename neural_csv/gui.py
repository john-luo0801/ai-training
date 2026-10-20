"""Desktop window for training on a CSV or predicting with the saved network."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neural_csv.charts import prediction_curve_figure
from neural_csv.engine import (
    forecast_series,
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

NAVY = "#141A24"
NAVY_2 = "#243044"
INK = "#1A2333"
MUTED = "#5E564C"
LINE = "#D9D0C3"
PAPER = "#EFEAE2"
WHITE = "#FBFAF6"
ACCENT = "#8C6239"
ACCENT_DARK = "#6E4C2C"
DANGER = "#8E2F2F"
FONT = "Segoe UI"
SERIF = "Georgia"


class CsvNetworkApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Neural CSV")
        self.geometry("1120x760")
        self.minsize(780, 520)
        self.configure(bg=PAPER)

        self.frame: pd.DataFrame | None = None
        self.csv_path: Path | None = None
        self.predictions: pd.DataFrame | None = None
        self.comparison: dict | None = None
        self.qa: dict | None = None
        self.forecast: dict | None = None
        self.graph_window: tk.Toplevel | None = None
        self._busy = False

        self._apply_theme()
        self._build()
        self._refresh_model_status()
        self._reprocess_on_startup()

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=(FONT, 10), background=PAPER, foreground=INK)
        style.configure("App.TFrame", background=PAPER)
        style.configure("Card.TFrame", background=WHITE)
        style.configure("Toolbar.TFrame", background=WHITE)
        style.configure("Title.TLabel", background=PAPER, foreground=INK, font=(SERIF, 20))
        style.configure("Section.TLabel", background=WHITE, foreground=INK, font=(SERIF, 13))
        style.configure("Body.TLabel", background=WHITE, foreground=INK, font=(FONT, 10))
        style.configure("Muted.TLabel", background=WHITE, foreground=MUTED, font=(FONT, 10))
        style.configure("PaperMuted.TLabel", background=PAPER, foreground=MUTED, font=(FONT, 10))
        style.configure("Status.TLabel", background=WHITE, foreground=INK, font=(FONT, 10))
        style.configure("Card.TLabelframe", background=WHITE, relief="flat", borderwidth=0)
        style.configure("Card.TLabelframe.Label", background=WHITE, foreground=INK, font=(FONT, 11, "bold"))
        style.configure("Primary.TButton", font=(FONT, 10, "bold"), padding=(14, 8), background=ACCENT, foreground=WHITE, borderwidth=0)
        style.map(
            "Primary.TButton",
            background=[("pressed", ACCENT_DARK), ("active", ACCENT_DARK), ("disabled", "#B7C3D4")],
            foreground=[("disabled", "#F4EFE6")],
        )
        style.configure("Secondary.TButton", font=(FONT, 10), padding=(12, 8), background=WHITE, foreground=NAVY, borderwidth=1)
        style.map("Secondary.TButton", background=[("pressed", "#E7DDD0"), ("active", "#F3EDE4"), ("disabled", PAPER)])
        style.configure("Danger.TButton", font=(FONT, 10), padding=(12, 8), background=WHITE, foreground=DANGER, borderwidth=1)
        style.map("Danger.TButton", background=[("pressed", "#F0DDD6"), ("active", "#F8EEEA"), ("disabled", PAPER)])
        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK, rowheight=28, borderwidth=0, font=(FONT, 10))
        style.configure("Treeview.Heading", background=NAVY, foreground="#F4EFE6", font=(FONT, 10, "bold"), relief="flat", padding=(8, 6))
        style.map("Treeview", background=[("selected", "#E7D7C3")], foreground=[("selected", INK)])
        style.map("Treeview.Heading", background=[("active", NAVY_2)])

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        bar = tk.Frame(self, bg=NAVY)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="Neural CSV", bg=NAVY, fg="#F4EFE6", font=(SERIF, 20)).grid(
            row=0, column=0, sticky="w", padx=20, pady=(14, 0)
        )
        tk.Label(
            bar,
            text="Train, predict, and reverse-forecast tabular data",
            bg=NAVY,
            fg="#C8BBA8",
            font=(FONT, 10),
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 14))
        self.file_var = tk.StringVar(value="No file open")
        self.path_var = tk.StringVar(value="Column 1 = ID    Column 2 = target    Column 3+ = indicators")
        tk.Label(bar, textvariable=self.file_var, bg=NAVY, fg="#F4EFE6", font=(SERIF, 13)).grid(
            row=0, column=1, sticky="e", padx=20, pady=(14, 0)
        )
        tk.Label(bar, textvariable=self.path_var, bg=NAVY, fg="#C8BBA8", font=(FONT, 9)).grid(
            row=1, column=1, sticky="e", padx=20, pady=(0, 14)
        )

        toolbar = tk.Frame(self, bg=WHITE, highlightthickness=1, highlightbackground=LINE)
        toolbar.grid(row=1, column=0, sticky="ew")
        actions = ttk.Frame(toolbar, style="Toolbar.TFrame", padding=(16, 10, 16, 4))
        actions.pack(fill="x")
        ttk.Button(actions, text="Open CSV", style="Primary.TButton", command=self._select_csv).pack(side="left")
        self.graph_button = ttk.Button(actions, text="View graphs", style="Secondary.TButton", command=self._open_graphs, state="disabled")
        self.graph_button.pack(side="left", padx=(8, 0))
        self.reset_button = ttk.Button(actions, text="Reset system", style="Danger.TButton", command=self._confirm_reset)
        self.reset_button.pack(side="right")
        self.model_var = tk.StringVar(value="No model saved yet.")
        ttk.Label(toolbar, textvariable=self.model_var, style="Status.TLabel", wraplength=1040, justify="left").pack(
            anchor="w", padx=16, pady=(0, 10)
        )

        self.work = tk.PanedWindow(
            self,
            orient=tk.VERTICAL,
            sashwidth=8,
            sashrelief=tk.FLAT,
            bg=ACCENT,
            opaqueresize=True,
            bd=0,
        )
        self.work.grid(row=2, column=0, sticky="nsew", padx=16, pady=(12, 12))

        self.form_host = tk.Frame(self.work, bg=PAPER)
        self.form_canvas = tk.Canvas(self.form_host, bg=PAPER, highlightthickness=0, bd=0)
        self.form_scroll = ttk.Scrollbar(self.form_host, orient="vertical", command=self.form_canvas.yview)
        self.form_canvas.configure(yscrollcommand=self.form_scroll.set)
        self.form_scroll.pack(side="right", fill="y")
        self.form_canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.form_canvas, style="App.TFrame", padding=(4, 4, 8, 8))
        self.form_canvas.create_window((0, 0), window=self.body, anchor="nw", tags="body")
        self.body.bind("<Configure>", self._sync_form)
        self.form_canvas.bind("<Configure>", self._sync_form)
        self.form_canvas.bind("<Enter>", lambda _event: self.form_canvas.bind_all("<MouseWheel>", self._scroll_form))
        self.form_canvas.bind("<Leave>", lambda _event: self.form_canvas.unbind_all("<MouseWheel>"))

        self.table_host = tk.Frame(self.work, bg=PAPER)
        self.table_host.rowconfigure(0, weight=1)
        self.table_host.columnconfigure(0, weight=1)
        self.work.add(self.form_host, minsize=160, stretch="always")
        self._show_welcome()

    def _sync_form(self, _event=None) -> None:
        self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all"))
        width = self.form_canvas.winfo_width()
        if width > 1:
            self.form_canvas.itemconfigure("body", width=width)

    def _scroll_form(self, event) -> None:
        self.form_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _show_table_pane(self) -> None:
        if str(self.table_host) not in self.work.panes():
            self.work.add(self.table_host, minsize=240, stretch="always")
        self.after_idle(self._place_sash)

    def _hide_table_pane(self) -> None:
        if str(self.table_host) in self.work.panes():
            self.work.forget(self.table_host)

    def _place_sash(self) -> None:
        if str(self.table_host) not in self.work.panes():
            return
        height = self.work.winfo_height()
        if height < 120:
            return
        table_height = min(360, max(240, int(height * 0.48)))
        table_height = min(table_height, height - 150)
        self.work.sash_place(0, 0, max(140, height - table_height))

    def _show_welcome(self) -> None:
        self._hide_table_pane()
        self._clear_body()
        head = ttk.Frame(self.body, style="App.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(head, text="Start with a CSV file", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            head,
            text="The first column is the ID, the second is the value to learn, and every column after that is an indicator.",
            style="PaperMuted.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(4, 0))

        cards = ttk.Frame(self.body, style="App.TFrame")
        cards.grid(row=1, column=0, sticky="ew")
        cards.columnconfigure((0, 1, 2), weight=1)
        steps = (
            ("01", "Open", "Choose a CSV. Training rows you have already saved stay on disk until you reset."),
            ("02", "Choose an action", "Train a model, predict the target, or reverse-forecast each indicator."),
            ("03", "Review", "Read the metrics and the table. A graph appears after you predict, not while you train."),
        )
        for column, (number, title, body) in enumerate(steps):
            self._info_card(cards, column, number, title, body)

    def _info_card(self, parent: ttk.Frame, column: int, number: str, title: str, body: str) -> None:
        shell = tk.Frame(parent, bg=WHITE, highlightthickness=1, highlightbackground=LINE)
        shell.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0), pady=(4, 0))
        tk.Label(shell, text=number, bg=WHITE, fg=ACCENT, font=(SERIF, 12)).pack(anchor="w", padx=16, pady=(14, 0))
        tk.Label(shell, text=title, bg=WHITE, fg=INK, font=(SERIF, 14)).pack(anchor="w", padx=16, pady=(2, 0))
        tk.Label(shell, text=body, bg=WHITE, fg=MUTED, font=(FONT, 10), wraplength=280, justify="left").pack(
            anchor="w", padx=16, pady=(6, 16)
        )

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
        self.file_var.set(self.csv_path.name)
        self.path_var.set(str(self.csv_path))
        self._show_mode()

    def _show_mode(self) -> None:
        self._clear_body()
        assert self.frame is not None and self.csv_path is not None
        schema = parse_schema(self.frame)
        self._page_heading(
            "What should this file do?",
            f"{len(self.frame)} rows  ·  ID {schema.id_column}  ·  target {schema.target_column}  ·  {len(schema.feature_columns)} indicators",
        )
        cards = ttk.Frame(self.body, style="App.TFrame")
        cards.grid(row=1, column=0, sticky="ew")
        cards.columnconfigure((0, 1, 2), weight=1)
        choices = (
            (
                "Train",
                "Learn from this file and keep every previous training row.",
                "Use as training data",
                lambda: self._show_train(),
            ),
            (
                "Predict",
                "Fill the target column with the saved network.",
                "Make predictions",
                lambda: self._show_predict(),
            ),
            (
                "Reverse forecast",
                "Predict each indicator from the target, using saved training data only.",
                "Reverse forecast",
                lambda: self._show_reverse(),
            ),
        )
        for column, (title, body, button, command) in enumerate(choices):
            shell = tk.Frame(cards, bg=WHITE, highlightthickness=1, highlightbackground=LINE)
            shell.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
            tk.Label(shell, text=title, bg=WHITE, fg=INK, font=(SERIF, 16)).pack(anchor="w", padx=16, pady=(16, 0))
            tk.Label(shell, text=body, bg=WHITE, fg=MUTED, font=(FONT, 10), wraplength=280, justify="left").pack(
                anchor="w", padx=16, pady=(8, 12)
            )
            ttk.Button(shell, text=button, style="Primary.TButton", command=command).pack(anchor="w", padx=16, pady=(0, 16))

        self._show_table(self.frame.head(PREVIEW_ROWS))

    def _page_heading(self, title: str, subtitle: str) -> None:
        head = ttk.Frame(self.body, style="App.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(head, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text=subtitle, style="PaperMuted.TLabel", wraplength=720).pack(anchor="w", pady=(4, 0))

    def _show_train(self) -> None:
        self._clear_body()
        frame = self.frame
        assert frame is not None
        schema = parse_schema(frame)
        self._page_heading("Training", "Column 2 is the target. Choose which indicators the network may use.")

        controls = self._card(row=1)
        controls.columnconfigure(1, weight=1)
        self._field(controls, 0, "ID", schema.id_column)
        self._field(controls, 1, "Target", schema.target_column)

        ttk.Label(controls, text="Problem type", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.task_var = tk.StringVar(value="auto")
        task_box = ttk.Combobox(
            controls,
            textvariable=self.task_var,
            values=("auto", "classification", "regression"),
            state="readonly",
            width=28,
        )
        task_box.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(10, 0))

        ttk.Label(controls, text="Indicators", style="Muted.TLabel").grid(row=3, column=0, sticky="nw", pady=(12, 0))
        list_wrap = tk.Frame(controls, bg=WHITE, highlightthickness=1, highlightbackground=LINE)
        list_wrap.grid(row=3, column=1, sticky="ew", padx=(12, 0), pady=(12, 0))
        self.feature_list = tk.Listbox(
            list_wrap,
            selectmode=tk.MULTIPLE,
            height=5,
            exportselection=False,
            bg=WHITE,
            fg=INK,
            font=(FONT, 10),
            relief="flat",
            highlightthickness=0,
            selectbackground="#E7D7C3",
            selectforeground=INK,
            activestyle="none",
        )
        feature_columns = self._training_feature_choices(schema)
        for column in feature_columns:
            self.feature_list.insert(tk.END, column)
        self._select_default_features(feature_columns)
        self.feature_list.pack(fill="x", padx=4, pady=4)

        dim_buttons = ttk.Frame(controls, style="Card.TFrame")
        dim_buttons.grid(row=4, column=1, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Button(dim_buttons, text="Use all", style="Secondary.TButton", command=self._select_all_features).pack(side="left")
        ttk.Button(dim_buttons, text="Clear", style="Secondary.TButton", command=self._clear_features).pack(side="left", padx=(8, 0))

        actions = ttk.Frame(controls, style="Card.TFrame")
        actions.grid(row=5, column=1, sticky="w", padx=(12, 0), pady=(14, 0))
        self.train_button = ttk.Button(actions, text="Train network", style="Primary.TButton", command=self._start_training)
        self.train_button.pack(side="left")
        ttk.Button(actions, text="Back", style="Secondary.TButton", command=self._show_mode).pack(side="left", padx=(8, 0))

        retained = load_training_history()
        retained_text = (
            f" {len(retained)} training rows are already retained and will be kept."
            if retained is not None
            else ""
        )
        self._note(
            f"Clear the indicator list to train from the target column alone.{retained_text}",
            row=2,
        )
        self._show_table(frame.head(PREVIEW_ROWS))

    def _field(self, parent: ttk.Frame, row: int, label: str, value: str) -> None:
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=(8 if row else 0, 0))
        ttk.Label(parent, text=value, style="Body.TLabel").grid(row=row, column=1, sticky="w", padx=(12, 0), pady=(8 if row else 0, 0))

    def _card(self, row: int) -> ttk.Frame:
        shell = tk.Frame(self.body, bg=LINE)
        shell.grid(row=row, column=0, sticky="ew")
        inner = ttk.Frame(shell, style="Card.TFrame", padding=16)
        inner.pack(fill="x", padx=1, pady=1)
        return inner

    def _note(self, text: str, row: int) -> None:
        shell = tk.Frame(self.body, bg=WHITE, highlightthickness=1, highlightbackground=LINE)
        shell.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        self.result_text = tk.Text(
            shell,
            height=3,
            wrap="word",
            bg="#F8FAFC",
            fg=INK,
            font=(FONT, 10),
            relief="flat",
            highlightthickness=0,
            padx=12,
            pady=10,
        )
        self.result_text.pack(fill="x")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

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
                self.after(0, lambda message=str(error): self._training_failed(message))
                return
            self.after(0, lambda result=result: self._training_finished(result))

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
        self._set_result("\n".join(lines))

    def _training_failed(self, message: str) -> None:
        self._set_busy(False)
        self._set_result(message)
        messagebox.showerror("Training failed", message)

    def _show_predict(self) -> None:
        self._clear_body()
        summary = model_summary()
        if summary is None:
            self._empty_state("Train a model first", "Open a CSV and choose Use as training data before predicting.")
            return

        schema = parse_schema(self.frame)
        feature_text = ", ".join(summary["features"]) if summary["features"] else "target column only"
        self._page_heading("Predict the target", f"The saved {summary['task']} model writes '{summary['target']}'.")
        controls = self._card(row=1)
        ttk.Label(
            controls,
            text=f"Indicators the model expects: {feature_text}\nThis file uses ID {schema.id_column} and target {schema.target_column}.",
            style="Body.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        buttons = ttk.Frame(controls, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(14, 0))
        self.predict_button = ttk.Button(buttons, text="Run predictions", style="Primary.TButton", command=self._start_prediction)
        self.predict_button.pack(side="left")
        self.save_button = ttk.Button(buttons, text="Save CSV", style="Secondary.TButton", command=self._save_predictions, state="disabled")
        self.save_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Back", style="Secondary.TButton", command=self._show_mode).pack(side="left", padx=(8, 0))
        self._note("Leave the target column blank when you only need the model output.", row=2)
        self._show_table(self.frame.head(PREVIEW_ROWS))

    def _show_reverse(self) -> None:
        self._clear_body()
        summary = model_summary()
        if summary is None:
            self._empty_state("Train a model first", "Reverse forecasting uses indicators learned from retained training rows.")
            return
        if not summary["features"]:
            self._empty_state(
                "No indicators to forecast",
                "This model was trained on the target column alone. Train again with at least one indicator.",
            )
            return

        schema = parse_schema(self.frame)
        self._page_heading(
            "Reverse forecast",
            "Each indicator is predicted from the target. This file is not added to training.",
        )
        controls = self._card(row=1)
        ttk.Label(
            controls,
            text=(
                f"Known target: {schema.target_column} (column 2), mapped to '{summary['target']}'.\n"
                f"Indicators: {', '.join(summary['features'])}"
            ),
            style="Body.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        buttons = ttk.Frame(controls, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(14, 0))
        self.reverse_button = ttk.Button(buttons, text="Run reverse forecast", style="Primary.TButton", command=self._start_reverse)
        self.reverse_button.pack(side="left")
        self.save_button = ttk.Button(buttons, text="Save CSV", style="Secondary.TButton", command=self._save_predictions, state="disabled")
        self.save_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Back", style="Secondary.TButton", command=self._show_mode).pack(side="left", padx=(8, 0))
        self._note("Put known target values in column 2. Indicator columns are filled from the saved training data.", row=2)
        self._show_table(self.frame.head(PREVIEW_ROWS))

    def _empty_state(self, title: str, body: str) -> None:
        self._hide_table_pane()
        self._page_heading(title, body)
        ttk.Button(self.body, text="Back", style="Secondary.TButton", command=self._show_mode).grid(row=1, column=0, sticky="w")

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
        self._show_table(result.head(RESULT_ROWS))

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
                self.after(0, lambda message=str(error): self._prediction_failed(message))
                return
            self.after(0, lambda result=result, snapshot=snapshot: self._prediction_finished(result, snapshot))

        threading.Thread(target=work, daemon=True).start()

    def _prediction_finished(self, result: pd.DataFrame, _original: pd.DataFrame) -> None:
        self._set_busy(False)
        self.predictions = result
        self.save_button.configure(state="normal")
        schema = parse_schema(result)
        shown = min(len(result), RESULT_ROWS)
        message = f"Filled '{schema.target_column}' for {len(result)} row(s). Showing the first {shown}."
        try:
            self.forecast = forecast_series(result)
        except Exception as error:  # noqa: BLE001 - prediction succeeded; only the graph failed
            self.forecast = None
            message += f" Graph was not drawn: {error}"
        else:
            message += " The graph continues the retained training values into these predictions."
            self._open_graphs()
        self._set_result(message)
        self._show_table(result.head(RESULT_ROWS))

    def _open_graphs(self) -> None:
        if self.forecast is None:
            messagebox.showinfo("No graph yet", "Run a prediction first. Training does not open a graph.")
            return
        window = self.graph_window
        if window is None or not window.winfo_exists():
            window = tk.Toplevel(self)
            window.title("Prediction")
            window.geometry("1080x560")
            window.minsize(760, 420)
            window.configure(bg="#EFEAE2")
            window.columnconfigure(0, weight=1)
            window.rowconfigure(1, weight=1)
            header = tk.Frame(window, bg="#141A24")
            header.grid(row=0, column=0, sticky="ew")
            tk.Label(header, text="Prediction", bg="#141A24", fg="#F4EFE6", font=("Georgia", 18)).pack(anchor="w", padx=20, pady=(14, 0))
            tk.Label(
                header,
                text="One continuous line. Ink is retained training data. Bronze is the prediction.",
                bg="#141A24",
                fg="#C8BBA8",
                font=(FONT, 10),
            ).pack(anchor="w", padx=20, pady=(2, 14))
            window.chart = tk.Frame(window, bg="#F7F4EE", highlightthickness=1, highlightbackground="#D9D0C3")
            window.chart.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
            window.canvases = []
            window.protocol("WM_DELETE_WINDOW", window.destroy)
            self.graph_window = window
        for canvas in window.canvases:
            canvas.get_tk_widget().destroy()
        window.canvases.clear()
        figure = prediction_curve_figure(self.forecast)
        canvas = FigureCanvasTkAgg(figure, master=window.chart)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
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

    def _show_table(self, frame: pd.DataFrame, row: int = 0) -> None:
        del row
        self._show_table_pane()
        if hasattr(self, "table_wrap") and self.table_wrap.winfo_exists():
            self.table_wrap.destroy()
        self.table_wrap = tk.Frame(self.table_host, bg=WHITE, highlightthickness=1, highlightbackground=LINE)
        self.table_wrap.grid(row=0, column=0, sticky="nsew")
        self.table_wrap.columnconfigure(0, weight=1)
        self.table_wrap.rowconfigure(1, weight=1)
        tk.Label(self.table_wrap, text="Data", bg=WHITE, fg=INK, font=(SERIF, 13)).grid(
            row=0, column=0, sticky="w", padx=12, pady=(8, 4)
        )

        columns = list(frame.columns)
        tree = ttk.Treeview(self.table_wrap, columns=columns, show="headings", height=8)
        for column in columns:
            tree.heading(column, text=str(column))
            tree.column(column, width=140, anchor="w", stretch=True)
        tree.tag_configure("odd", background="#F6F1E8")
        tree.tag_configure("even", background=WHITE)
        for index, record in enumerate(frame.itertuples(index=False)):
            tag = "odd" if index % 2 else "even"
            tree.insert("", tk.END, values=["" if pd.isna(value) else value for value in record], tags=(tag,))

        y_scroll = ttk.Scrollbar(self.table_wrap, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(self.table_wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=1, column=0, sticky="nsew", padx=(8, 0))
        y_scroll.grid(row=1, column=1, sticky="ns")
        x_scroll.grid(row=2, column=0, sticky="ew", padx=(8, 0))

    def _clear_body(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        for row in range(6):
            self.body.rowconfigure(row, weight=0)

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
        if self.forecast is not None and not busy:
            self.graph_button.configure(state="normal")
        elif busy:
            self.graph_button.configure(state="disabled")

    def _refresh_model_status(self) -> None:
        summary = model_summary()
        if summary is None:
            self.model_var.set("No model saved yet.")
            return
        if self.forecast is not None:
            self.graph_button.configure(state="normal")
        retained = summary.get("retained_rows") or 0
        feature_text = ", ".join(summary["features"]) if summary["features"] else "target only"
        metric_bits = ", ".join(f"{key}={value}" for key, value in summary["metrics"].items())
        self.model_var.set(
            f"{summary['task']} → {summary['target']}  ·  {feature_text}  ·  {retained} retained rows  ·  {metric_bits}"
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
            self.model_var.set(self.model_var.get() + "  ·  reprocessed on startup")

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
        self.forecast = None
        self.graph_button.configure(state="disabled")
        if self.graph_window is not None and self.graph_window.winfo_exists():
            self.graph_window.destroy()
        self._refresh_model_status()
        self._show_welcome()
        self.file_var.set("No file open")
        self.path_var.set("Column 1 = ID    Column 2 = target    Column 3+ = indicators")
        messagebox.showinfo("Reset complete", "Retained training data and the saved model were deleted.")


def main() -> None:
    app = CsvNetworkApp()
    app.mainloop()
