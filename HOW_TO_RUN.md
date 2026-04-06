# How to run

## 1) Set up environment (recommended)

### Option A: Conda (matches your `(llm)` environment)

```bash
conda activate llm
conda install -c pytorch -c conda-forge pytorch torchvision torchaudio
conda install -c conda-forge datasets pyarrow numpy matplotlib
```

### Option B: venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch datasets numpy matplotlib
```

## 2) Run the script

```bash
python3 llm_architectures_hometask_1.py
```

Notes:
- On first run, HuggingFace `datasets` will download `SetFit/sst2` (requires internet) and cache it locally.
- By default, the **experiment/plot sections** are **off** in `llm_architectures_hometask_1.py`:
  - `RUN_TASK_1_3 = False` (Task 1.3 sweep + heatmaps)
  - `RUN_TASK_1_4 = False` (Task 1.4 L1 experiments + plots)
  - `RUN_PART_2 = False` (optimizer comparisons + plots)
  This is intentional because those runs can take a while and generate many plots/files.

## Troubleshooting: `datasets` import error (pyarrow mismatch)

If you see an error like `AttributeError: readonly attribute` while importing `datasets`, upgrade `datasets` + `pyarrow`:

```bash
pip install -U datasets pyarrow
```

Conda alternative:

```bash
conda install -c conda-forge datasets pyarrow
```

## 3) Generate homework plots (Tasks 1.3 / 1.4 / Part 2)

Edit `llm_architectures_hometask_1.py` and set one or more of:
- `RUN_TASK_1_3 = True` (around line ~984)
- `RUN_TASK_1_4 = True` (around line ~1256)
- `RUN_PART_2 = True` (around line ~1648)

You can jump to them quickly with:

```bash
rg -n "^RUN_TASK_1_3\\b|^RUN_TASK_1_4\\b|^RUN_PART_2\\b|^RUN_DATA_PIPELINE\\b" llm_architectures_hometask_1.py
```

Tip: if you only want Part 2 (toy optimizers), set `RUN_DATA_PIPELINE = False` (around line ~118) to skip dataset/BoW work.

Then run:

```bash
python3 llm_architectures_hometask_1.py
```

Plots are saved as PNGs in:
- `plots/`

If you want interactive display instead of saving, set:
- `SHOW_PLOTS = True` (and/or `SAVE_PLOTS = False`)
