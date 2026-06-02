# Hybrid Architectures for Helmholtz Scattering

This repository contains the code used for the paper:

**Hybrid operator learning of wave scattering maps in high-contrast media**

Paper: [arXiv:2602.11197](https://arxiv.org/abs/2602.11197)

The paper studies 40 Hz Helmholtz forward maps for high-contrast salt-body velocity models. The reproducibility code covers the three learning tasks used in the experiments:

- `smooth2smooth`: smoothed velocity `v_smooth -> p_smooth`
- `contrast`: residual correction `(v_delta, p_smooth) -> p_delta`
- `sharp2sharp`: high-contrast velocity `v_sharp -> p_sharp`

The hybrid model evaluates `sharp2sharp` by composing an FNO trained on `smooth2smooth` with a scOT transformer trained on `contrast`.

## Repository Layout

```text
.
|-- helmholtz_hybrid/          Reusable Python package: datasets, losses, evaluation, and runtime helpers
|-- configs/                   Paper model/training defaults used by the training scripts
|-- scripts/                   Command-line wrappers for data prep, training, evaluation, figures, and releases
|-- papers/                    Local paper artifacts, when present
|-- run_all.sh                 End-to-end reproduction shell script
|-- pyproject.toml             Python package metadata and dependencies
|-- README.md                  Reproduction instructions
|-- data/                      Downloaded and prepared datasets; ignored by Git
|   |-- raw/                   Raw Hugging Face arrays, grouped by dataset
|   |   `-- const_back/
|   `-- processed/             Train/validation/test arrays produced by scripts/prepare_data.py
|       `-- const_back/
|-- outputs/                   Generated artifacts; ignored by Git
|   |-- checkpoints/           Trained or downloaded model checkpoints
|   |   `-- const_back/
|   |-- figures/               Generated paper-style figures
|   |   `-- const_back/
|   `-- logs/                  Long-running training and workflow logs
|       `-- const_back/
`-- results/                   Evaluation metrics and aggregate CSVs; ignored by Git
    `-- const_back/
```

Large data arrays and trained checkpoints are intentionally not stored in Git.
They are available on Hugging Face and should be downloaded into the layout
shown above. Follow the "Quickstart" instructions below.

## Installation

We use the official PyTorch `neuraloperator` library for FNO and the Poseidon repo `github.com/camlab-ethz/poseidon` for scOT. Maintaining Cuda/PyTorch dependencies that work for both libraries is a bit delicate. The instructions below work for our machine, but you may need to adjust them based on your hardware.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Blackwell/CUDA 12.8 GPU machines:
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e .

# scOT currently declares torch==2.0.1, so install it without dependencies to
# avoid downgrading the Blackwell-compatible PyTorch wheel.
python -m pip install --no-deps "scot @ git+https://github.com/camlab-ethz/poseidon.git"
```

Use Python 3.10 or 3.11. For the RTX PRO 6000 Blackwell Server Edition machines, use a CUDA 12.8-compatible PyTorch wheel such as `torch==2.7.0+cu128`. Do not install scOT with dependencies, because the upstream package metadata pins `torch==2.0.1`.

## Quickstart

### Prepare directories (optional)

The datasets and trained models are about 100G in total. They are expected
under `data/` and `outputs/`. If desired, create your own symlink at one of
those paths before downloading.

### Download datasets and trained models

The raw datasets and trained models used in the paper are hosted on
Hugging Face:

- Raw data: [davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media](https://huggingface.co/datasets/davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media)
- Trained models: [davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media](https://huggingface.co/davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media)

The dataset and model repos intentionally use the same slug. Use the matching
`--repo-type` shown below.

```bash
HF_REPO=davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media

hf download "$HF_REPO" \
  --repo-type dataset \
  --local-dir data/raw

hf download "$HF_REPO" \
  --repo-type model \
  --local-dir outputs/checkpoints/const_back/paper
```

### Prepare data

Prepare the processed train/validation/test arrays from the raw download. By default this writes the paper split: 40,000 train, 5,000 validation, and 5,000 test samples.

```bash
python scripts/prepare_data.py \
  --raw-root data/raw \
  --output-root data/processed \
  --dataset const_back
```

### Evaluate models on test data

To evaluate the downloaded paper checkpoints without retraining:

```bash
MODEL_VERSION=paper \
MODEL_ACTION=evaluate \
bash run_all.sh
```

`run_all.sh` prepares processed arrays if they are missing, evaluates the
existing checkpoints, collects metrics, and regenerates the paper-style figures.
`MODEL_ACTION=evaluate` skips training and reads checkpoints from
`outputs/checkpoints/<dataset>/<model-version>/`.

## Data Layout

Large arrays and trained checkpoints are expected under the repository-local
`data/` and `outputs/` directories. If you want those directories to live on a
different filesystem, create your own symlink at `data/` or `outputs/`.

```text
data/
  raw/
    const_back/
      velocity_sharp.npy
      velocity_smooth.npy
      pressure_sharp.npy
      pressure_smooth.npy

  processed/
    const_back/
      {train,validation,test}/
        velocity_sharp.npy
        velocity_smooth.npy
        velocity_delta.npy
        pressure_sharp.npy
        pressure_smooth.npy
        pressure_delta.npy

outputs/
  checkpoints/
    const_back/
      <model-version>/
        fno/
        scot/
  figures/
    const_back/
      <model-version>/
  logs/
    const_back/
      <model-version>/

results/
  const_back/
    <model-version>/
      evaluation/
      parameter_scaling.csv
```

The downloaded checkpoints use `MODEL_VERSION=paper`. Training runs default to
`MODEL_VERSION=seed<seed>` unless you set `MODEL_VERSION` yourself, for example
`seed123_rerun2` or `seed456`.

Prepared data written by `scripts/prepare_data.py`:

```text
data/processed/{dataset}/{train,validation,test}/
  velocity_sharp.npy
  velocity_smooth.npy
  velocity_delta.npy
  pressure_sharp.npy
  pressure_smooth.npy
  pressure_delta.npy
```

Raw pressure arrays are complex `[N,H,W]` arrays. Prepared pressure arrays are real-valued `[N,2,H,W]` arrays with channel order `[real, imag]`. The `delta` arrays are derived during preprocessing as `sharp - smooth`.

## Train Models

The paper trains FNO and scOT models on all three tasks and sweeps FNO layers/scOT depths over `2, 4, 6, 8, 10`. Fixed paper hyperparameters live in `configs/fno_paper.yaml` and `configs/scot_paper.yaml`; the CLI supplies only the dataset, task, sweep size, seed, and output naming. Use a fixed `--seed` and a matching `MODEL_VERSION`, such as `seed123`, for reproducible reruns. Each training script writes a native checkpoint plus `run_manifest.json` with command-line arguments, parameter count, package/runtime metadata, and Git state.

When `--output-root` is omitted, single training runs write to
`outputs/checkpoints/<dataset>/seed<seed>/{fno,scot}/`.

Single FNO run:

```bash
python scripts/train_fno.py \
  --config configs/fno_paper.yaml \
  --data-root data/processed \
  --dataset const_back \
  --task smooth2smooth \
  --num-layers 6 \
  --seed 123
```

Single scOT run:

```bash
python scripts/train_scot.py \
  --config configs/scot_paper.yaml \
  --data-root data/processed \
  --dataset const_back \
  --task contrast \
  --depths 6 6 6 6 \
  --seed 123
```

Full paper sweep:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export DATASET=const_back
export SEED=123
export MODEL_VERSION=seed123
export MODEL_ACTION=train
bash run_all.sh
```

`MODEL_ACTION=train` trains the full FNO/scOT sweep, evaluates the trained
checkpoints, collects the parameter-scaling CSV, and generates figures. Training
is split across the first three entries in `CUDA_VISIBLE_DEVICES`:
`smooth2smooth`, `contrast`, and `sharp2sharp` each run on one GPU. Set
`WANDB_PROJECT=` to disable W&B logging.

## Evaluate

Single model:

```bash
python scripts/evaluate.py \
  --model-type fno \
  --task sharp2sharp \
  --checkpoint outputs/checkpoints/const_back/paper/fno/<run-name> \
  --data-root data/processed \
  --dataset const_back
```

Hybrid model:

```bash
python scripts/evaluate.py \
  --model-type hybrid \
  --fno-smooth-checkpoint outputs/checkpoints/const_back/paper/fno/<smooth-run> \
  --scot-contrast-checkpoint outputs/checkpoints/const_back/paper/scot/<contrast-run> \
  --data-root data/processed \
  --dataset const_back
```

## Figures

Data examples:

```bash
python scripts/plot_data_examples.py \
  --data-root data/processed \
  --dataset const_back \
  --output-dir outputs/figures/const_back/paper
```

Qualitative comparison:

```bash
python scripts/plot_result_comparison.py \
  --data-root data/processed \
  --dataset const_back \
  --index 0 \
  --fno-sharp-checkpoint outputs/checkpoints/const_back/paper/fno/<sharp-run> \
  --scot-sharp-checkpoint outputs/checkpoints/const_back/paper/scot/<sharp-run> \
  --fno-smooth-checkpoint outputs/checkpoints/const_back/paper/fno/<smooth-run> \
  --scot-contrast-checkpoint outputs/checkpoints/const_back/paper/scot/<contrast-run>
```

Parameter scaling:

```bash
python scripts/collect_parameter_scaling.py \
  results/const_back/paper/evaluation/*.json \
  --output results/const_back/paper/parameter_scaling.csv

python scripts/plot_parameter_scaling.py \
  --metrics-csv results/const_back/paper/parameter_scaling.csv \
  --output outputs/figures/const_back/paper/parameter_scaling.png
```

The metrics CSV must contain columns `panel,model,parameters,rel_l2`, where `panel` is one of `Smooth`, `Residual`, or `Sharp`.

## Hugging Face Artifact Release

This section is for maintainers who need to update the Hugging Face artifact
repos.

Use separate Hugging Face repos for the raw arrays and the trained checkpoints:

- Dataset repo: `davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media`
- Model repo: `davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media`

The release utility builds clean staging folders before uploading. This avoids
publishing optimizer state, trainer state, intermediate checkpoint state, logs,
W&B files, or cache directories.

```bash
source .venv/bin/activate
hf auth login

python scripts/upload_hf_artifacts.py \
  --dataset-repo-id davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media \
  --model-repo-id davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media \
  --replace-staging
```

Inspect the generated staging folders under `.hf_staging/`. When the payloads
look right, run the same command with `--upload`:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
HF_XET_CACHE=/tmp/hf_xet_$USER \
python scripts/upload_hf_artifacts.py \
  --dataset-repo-id davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media \
  --model-repo-id davidMis/hybrid-operator-learning-of-wave-scattering-maps-in-high-contrast-media \
  --replace-staging \
  --upload
```

By default the staging utility uses hardlinks. If `data/` or `outputs/` is a
symlink to another filesystem and hardlink staging fails, pass `--staging-root`
on that filesystem or use `--link-mode copy`.
