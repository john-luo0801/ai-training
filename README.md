# Neural CSV

Neural CSV is a small program that learns from a CSV file and fills in a
missing value for new rows. You open it in your web browser, choose a CSV,
and then train a neural network, make predictions, or run a reverse forecast.

No programming is needed for normal use. If you do want to change how it
behaves, the [Customize the code](#-customize-the-code) section at the bottom
explains the few places worth editing, one at a time.

---

## 1. What it can do

| Action | What happens |
| --- | --- |
| **Train** | The network learns from your CSV. 70% of the rows teach it, and the other 30% are kept aside to test it (the QA test). |
| **Predict** | You give it rows with an empty target column, and it fills the target in for you. |
| **Reverse forecast** | You give it rows with a known target, and it works backwards to fill in the indicator columns. |
| **Remember training data** | Every row you have ever trained on is kept when you close and reopen the program, until you press Reset and confirm. |

---

## 2. Quick start (Windows)

You need Python 3 installed. Then, in the project folder:

```powershell
py -3 -m pip install -r requirements.txt
py -3 main.py
```

Or just double-click **`RUN.bat`**.

Your browser opens at:

```
http://127.0.0.1:8765
```

> **Two things that trip people up**
>
> 1. Use `py -3`, not `python`. On many Windows machines `python` opens the
>    Microsoft Store instead of running your code.
> 2. No internet connection is needed. The page styling (Tailwind), charts
>    (Chart.js), and the Inter font are bundled with the program in
>    `neural_csv/static/vendor/` and served locally.

To stop the program, go back to the terminal window and press `Ctrl+C`.

---

## 3. The CSV format — the only rule that matters

Your file needs a header row (the column names) and at least two columns.
**The program reads columns by position, not by name**, so your column names
can be anything you like:

| Position | Meaning | Example names that all work |
| --- | --- | --- |
| Column 1 | ID / index for each row | `id`, `code`, `patient_no`, `record` |
| Column 2 | **Target** — the value to learn and predict | `price`, `cost`, `outcome`, `sales` |
| Column 3 onwards | Indicators (optional) — facts that help predict the target | `size_sqft`, `bedrooms`, `age` … |

A small example:

```csv
id,price,size_sqft,bedrooms,age_years,neighborhood
H01,59500,800,2,1,south
H02,74000,900,3,2,north
H03,88500,1000,4,3,south
```

Good to know:

- **Indicators are optional.** A file with only two columns (ID + target)
  still trains. See `data/sample_target_only.csv`.
- **You can add or remove indicator columns.** Tick or untick them on the
  Train screen to choose which ones the network uses.
- **Indicator values can be numbers or words.** Words (like `north` /
  `south`) are converted to numbers automatically.
- **Leave the target empty to ask for a prediction.** A prediction file
  looks like the training file, but with column 2 blank.
- Column names must not repeat inside one file.
- Training needs at least **4 rows** that have a target value, because of
  the 70 / 30 split described below.

Because position is what counts, a later file may even use different column
names than the file you trained on — column 1 is matched to the ID, column 2
to the target, and the rest to the indicators in order, as long as the
indicator count is the same.

---

## 4. Using the program, step by step

### Step 1 — Open a CSV

Click **Open CSV** and pick a file. You will see the row count, the target
column name, and a table of your data.

### Step 2 — Train

1. Click the **Train** card.
2. Leave **Problem type** on *Auto* unless you know which one you need
   (see [How it decides](#how-it-decides-classification-or-regression)).
3. Tick the indicators the network may use, or untick all of them to train
   on the target column alone.
4. Click **Train network**. The button is disabled while training runs, and
   a banner shows that training is in progress, so it cannot be started
   twice by accident.

When training finishes, two things appear:

- A short report (how many rows were used, and the test score).
- The **QA test graph**. During training, 70% of the rows taught the
  network and 30% were hidden from it. The graph compares the real values
  of those hidden rows (dark line) with what the network guessed (blue
  line). Green dots passed, red dots failed. It only appears after
  training — not when you open a file or make predictions.

### Step 3 — Predict

1. Open a CSV whose target column (column 2) is empty.
2. Click **Predict**, then **Run predictions**.
3. The table now shows the filled-in targets, and a graph joins your saved
   training values (dark) to the new predictions (blue) as one line.

### Step 4 — Reverse forecast (optional)

1. Open a CSV that has values in the target column but blanks in the
   indicator columns.
2. Click **Reverse forecast**, then **Run reverse forecast**.
3. Each indicator is predicted from the target, using only the data from
   earlier training. This file is **never** used for training and is not
   saved.

### Reset

**Reset system** deletes all retained training rows and the saved model.
It always asks you to confirm first, and nothing else in the program can
clear your data.

---

## 5. Sample files to try

Everything below is in the `data` folder.

| File | Rows | Use it for |
| --- | --- | --- |
| `sample_houses.csv` | 10 | Your first training run |
| `sample_houses_to_predict.csv` | 1 | Predicting one house price |
| `sample_target_only.csv` | 10 | Training with no indicators |
| `sample_houses_to_predict_10k.csv` | 10,000 | A bigger prediction run |
| `sample_target_only_10k.csv` | 10,000 | Bigger target-only training |
| `sample_target_only_to_predict_10k.csv` | 10,000 | Bigger target-only prediction |
| `sample_houses_1m.csv` | 1,000,000 (~30 MB) | Testing with a lot of data |

A good first run: train on `sample_houses.csv` with all indicators ticked
and Problem type set to *Regression*, then open
`sample_houses_to_predict.csv` and predict.

To rebuild the large sample files:

```powershell
py -3 data/generate_sample_houses_1m.py   :: or double-click Data_generation.bat
py -3 data/generate_sample_sets.py        :: or double-click sampleset.bat
```

---

## 6. How it works (in plain words)

- **The network** is a multilayer perceptron (MLP) from scikit-learn: one
  hidden layer of 24 neurons. It is small on purpose, so it can still learn
  from a tiny file such as 10 rows.
- **Before learning**, missing numbers are filled with the middle value of
  their column, numbers are scaled to a similar range, and word columns are
  turned into on/off columns. This preprocessing matters — neural networks
  struggle when one column is in the thousands and another is in single
  digits.
- **Numbers vs labels.** If the target is a number, the program predicts
  numbers (regression). If it is a word or a small set of whole numbers, it
  predicts a label (classification).

### How it decides classification or regression

With Problem type on *Auto*: a target that is not numeric, or a whole-number
target with 2 to 15 different values, is treated as classification.
Anything else is regression. You can always override it on the Train
screen.

### How the QA test scores a row

- **Regression:** a row passes when the prediction is within **5%** of the
  real value.
- **Classification:** a row passes only when the label matches exactly, so
  a low pass count on label data is common and does not mean training
  failed — compare the two lines on the graph instead.

For very large test sets the score still counts every held-out row, but
the graph draws at most 200 points so the page stays fast.

### Where your data is saved

Training writes three files into the `models` folder:

| File | Contents |
| --- | --- |
| `models/training_history.csv` | Every training row kept so far |
| `models/model.joblib` | The trained network and its settings |
| `models/training_state.joblib` | Which columns are the ID, target, and indicators |

When you train again, rows whose ID already exists are updated, new IDs
are added, and all other rows are kept. When the program starts, it
retrains from the saved rows so the model always matches the history.

---

## 7. Project layout

| File / folder | What it is |
| --- | --- |
| `main.py` | The file you run. It starts the web page. |
| `RUN.bat` | Same thing, by double-clicking. |
| `neural_csv/engine.py` | All the learning logic: reading CSVs, training, predicting, reverse forecasting, saving history. |
| `neural_csv/web.py` | The local web server and its API (runs on port 8765). |
| `neural_csv/templates/index.html` | The page you see — layout, colours, charts, buttons. Styled with Tailwind. |
| `neural_csv/static/vendor/` | Bundled copies of Tailwind, Chart.js, and the Inter font, so the page works fully offline. |
| `neural_csv/charts.py` | Matplotlib versions of the graphs, used by the tests. |
| `neural_csv/gui.py` | The old desktop (tkinter) window. Kept for reference; `main.py` no longer opens it. |
| `tests/test_engine.py` | Automated checks for the learning logic. |
| `data/` | Sample CSV files and the scripts that generate them. |
| `models/` | Created automatically. Your saved model and training rows. |

---

## 8. ✏️ Customize the code

Each change below is small and independent. After changing Python code,
stop the program (`Ctrl+C`) and start it again with `py -3 main.py`.
After changing only `index.html`, a browser refresh is enough.

### Make the network bigger or smaller

In `neural_csv/engine.py`, find `_build_pipeline` and change:

```python
"hidden_layer_sizes": (24,),
```

- `(24,)` means one hidden layer with 24 neurons.
- Try `(64, 32)` for two layers. Bigger can fit harder patterns but is
  slower and can be worse on tiny files.
- `alpha` (just below it) controls how strongly the network avoids
  over-fitting. Larger `alpha` = smoother, simpler answers.

Solvers are chosen by size in the same function: small datasets (≤ 500
training rows) use `lbfgs`; larger ones use `adam` with early stopping.
Change the `500` threshold if you want to experiment.

### Change the 70 / 30 split

In `neural_csv/engine.py`, `_split_learning_rows` contains
`test_size=0.3`. Change it to `0.2` for an 80 / 20 split, for example.
The `random_state=42` makes the split the same on every run — change the
number to get a different shuffle.

### Change what counts as a QA pass

In `neural_csv/engine.py`, `_qa_passed` contains `<= 0.05` (5%). Raise it
to `0.10` to accept predictions within 10% of the real value.

### Change when a numeric target counts as labels

In `neural_csv/engine.py`, `infer_task` treats a whole-number target with
up to `15` different values as classification. Change that number if your
labels use more values, or force the choice with the Problem type dropdown
instead.

### Change the colours and layout of the page

Everything visual is in `neural_csv/templates/index.html`, written with
Tailwind utility classes (for example `bg-zinc-950` for the dark sidebar,
`bg-white` for cards). Search the file for the colour you see on screen
and change the class. Chart colours are plain hex values inside the
`drawChart` and `drawQa` functions: `#18181b` (training / actual),
`#0369a1` (predicted), `#15803d` (passed), `#b91c1c` (failed).

### Change the port or the upload limit

In `neural_csv/web.py`: `port=8765` in `main()` sets the address, and
`MAX_CONTENT_LENGTH` near the top sets the largest CSV you can open
(currently 512 MB).

### Change how much the page loads at once

- The data table shows the first 200 rows: change `limit: int = 200` in
  `_records` in `neural_csv/web.py`.
- The QA graph draws at most 200 points: change `limit=200` in
  `_qa_report` in `neural_csv/engine.py`.

### Change the sample data

The generator scripts in `data/` build prices with the formula
`50 * size + 10000 * bedrooms - 500 * age`. Edit the formula, the row
count, or the `seed` (a different seed gives different random rows), then
run the script again.

### Use the engine in your own script

The web page is only a front-end. You can drive the learning code
directly:

```python
import pandas as pd
import neural_csv.engine as engine
from neural_csv.engine import train_model, predict, reverse_forecast

frame = pd.read_csv("data/sample_houses.csv")
result = train_model(frame, task="regression")
print(result.metrics)          # score on the held-out 30%
print(result.qa)               # per-row QA test details

to_fill = pd.read_csv("data/sample_houses_to_predict.csv")
print(predict(to_fill))        # target column filled in
```

> ⚠️ This writes to the real `models` folder, exactly like training in
> the browser. If you are only experimenting, call
> `engine.set_storage_dir("some/other/folder")` first so you do not touch
> your saved model — and remember that **Reset** in the browser deletes
> whatever the `models` folder holds.

---

## 9. Run the tests

```powershell
py -3 tests/test_engine.py
```

It prints `ok` when everything passes. The tests use their own scratch
folder (`tests/_tmp`), so they do not touch your saved model.

---

## 10. Troubleshooting

| Problem | Likely fix |
| --- | --- |
| `'python' is not recognized` or the Microsoft Store opens | Use `py -3` instead of `python`. |
| Page opens but looks unstyled | Hard-refresh the browser (Ctrl+F5) so it drops the old cached page. The styling is bundled locally, so no connection is needed. |
| `Address already in use` / port 8765 busy | The program is already running in another window, or change the port (see Customize). |
| “This file is missing prediction variables…” | Your prediction file has fewer indicator columns than the model was trained on. Add the missing columns (values can be blank for reverse forecasting, but prediction needs them filled). |
| Training says it needs at least 4 rows | Add more rows with a target value — the 70 / 30 split needs a few rows on each side. |
| QA test shows 0 passed on label data | Classification requires an exact match. Check whether your target should be Regression instead (override it under Problem type). |
| Predictions look poor | Try ticking more indicators, adding more training rows, or a bigger network (see Customize). Small files give rough answers. |
| You want to start completely over | Use **Reset system** in the page and confirm. Nothing else deletes your retained rows. |
