# A1: Pruning, Sparsity, Quantization and Adaptive Layerwise Compression

Read this first. Code covers the seven requested phases. Included measurements
are **synthetic verification**, not trained ResNet-18/CIFAR-10 results. A separate
integration test executes the actual 11,173,962-parameter CIFAR ResNet-18 forward
and backward. Full CIFAR training has not been executed in the delivered package.
Do not submit the verification numbers as assignment experimental findings.

## 1. Set up

Use Python 3.10+ (verified with Python 3.12). From this extracted folder:

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`
Linux/macOS: `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For a CUDA GPU, install the matching torch/torchvision build using the selector
at https://pytorch.org/get-started/locally/ before installing other dependencies.
CPU-only installation: `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
Internet is needed to install dependencies and download CIFAR-10 once. If the
dataset download is blocked, put the official extracted `cifar-10-batches-py`
folder inside `data/` and omit `--download`. No credentials are required.

## 2. Verify the complete pipeline quickly

```bash
python run.py --smoke --rounds 2 --out my_smoke
```

This intentionally uses a tiny network and artificial images. It exercises all
phases, export/reload, plotting, allocation, QAT, tables and hypervolume. It is
not a shortened CIFAR experiment and is not evidence of model quality.

## 3. Run the actual assignment

```bash
python run.py --download --device auto --epochs 100 --finetune-epochs 10 --rounds 3 --batch-size 128 --probe-samples 512 --targets 4 8 16 --out results_seed42
```

GPU strongly recommended. There are many experiments: baseline training, four
QAT runs, 12 uniform joint baselines with 30 fine-tuning epochs each, and three
adaptive targets with three rounds each. This can take many GPU hours; runtime
and memory vary by device. Increase baseline epochs to 150-200 if validation
performance is still improving. No fixed accuracy is promised.

To reuse a completed FP32 baseline:

```bash
python run.py --checkpoint results_seed42/models/baseline_fp32.pt --device auto --finetune-epochs 10 --rounds 3 --out results_reused
```

Use the same architecture and seed/split as the saved checkpoint. Each run uses
a separate output directory. The runner writes results.csv after each evaluated
model; baseline reuse restarts subsequent compression phases, not mid-phase
optimizer state. Use `--seed 43` and `--seed 44` in new folders for repeat runs.
Report means and standard deviations across seeds if compute permits.

Google Colab: open `A1_walkthrough.ipynb`, enable a GPU runtime, upload this
folder's Python files, and execute cells in order. The notebook launches the
same tested implementation instead of duplicating it.

## 4. What the files mean

- `run.py`: dataset split, training, every experiment, sensitivity probes,
  adaptive refinement, evaluation, plots and generated results report.
- `compression.py`: numerical quantizer, pruning, straight-through gradients,
  real bit-packed storage/reconstruction, Pareto frontier and hypervolume.
- `tests/test_compression.py`: regression tests including actual ResNet-18.
- `report/REPORT.md` and PDF: explained methodology and honest execution status.
- `verification/`: actual local test log and complete synthetic smoke outputs.
- `results*/results.csv`: all evaluated models, accuracy, actual archive size,
  actual zero fraction and dequantized floating-point batch latency.
- `sensitivity.csv`: phase 5 per-layer/per-option validation drops.
- `allocation_target*_round*.csv`: adaptive layer choices and budgets.
- `models/*.pkl`: actual packed weights, scale factors, masks, other FP32 state.
- `plots/`: baseline/per-layer histograms; pruning/quantization histograms;
  accuracy-sparsity, accuracy-bits, accuracy-size, PTQ-QAT, sensitivity plots.
- `RESULTS_REPORT.md`: table generated from that run's actual measurements.

## 5. Technical choices you must be able to explain

1. ResNet-18 uses a CIFAR stem (3x3 convolution, no maxpool), 10 outputs, no
   pretrained weights. The seed fixes a 45,000/5,000 train/validation split;
   the official 10,000 test set is reserved for reporting. All tuning and
   reallocation use validation/probe data. Probe is the first 512 validation
   examples with no random augmentation.
2. Global unstructured pruning uses PyTorch global L1 magnitude. Filter pruning
   uses PyTorch output-dimension L2 masking. Whole filters are zero, but tensor
   dimensions are unchanged: this is structured masking, not physical channel
   removal. Requested percentages can yield different actual sparsities because
   structured pruning rounds filter counts per layer. Use actual sparsity in
   comparisons. No reduction in FLOPs or latency is implied.
3. Quantization is **weight-only** with per-output-channel scales. Q8/Q4/Q2 use
   symmetric signed integer levels, Q1 uses binary +/-mean(abs(nonpruned weights)).
   Q1 pruning needs a zero mask. BN, biases and activations stay FP32. FP16 uses
   a half-precision storage round trip with FP32 execution. This is a portable
   numerical study, not native INT4/INT2/1-bit compute or NPU benchmarking.
4. QAT uses FP32 master weights, quantized forward values and a straight-through
   gradient. Masks stay fixed during fine-tuning; BN statistics are frozen.
   Saved integer codes reconstruct the evaluated weights and predictions, checked
   after every model. It does not use torchao native backend conversion.
5. The allocator measures all 4 bits x 3 sparsities for each conv/linear layer
   independently. It uses validation CE for a finer-grained distortion score
   and Q4/30% accuracy drop for sensitivity. `exp(-drop/tau)` represents desired
   compression; sensitive layers receive less compression pressure. The PDF's
   formula is ambiguous if B_l means retained bytes: allocating fewer retained
   bytes to sensitive layers would be counterproductive. This interpretation is
   stated rather than silently reversing the algorithm.
6. Adaptive optimization starts with minimum estimated storage per layer, then
   buys feasible upgrades maximizing sensitivity-weighted loss reduction/byte.
   It is a discrete heuristic, not a global optimality guarantee. Two or three
   rounds fine-tune, re-probe effective weights and update the allocation while
   preserving master weights. Re-probing may underestimate newly restored
   weights; joint layer interactions also limit isolated-layer predictions.
7. Target 4x/8x/16x refers to estimated compressible-weight bytes relative to
   FP32 conv/linear weights. Masks and scales count in the estimate. BN/biases
   and serialization overhead count in the **actual archive size**, so total
   model compression may miss the nominal target. Per-layer metadata estimate
   is conservative; the actual file is authoritative. Do not label a nominal
   target as an achieved total-model ratio.
8. Uniform joint baselines get the same total fine-tuning epochs as an adaptive
   target. PTQ gets none; QAT-only runs use the configured fine-tuning epochs.
   Probe overhead is additional and should be disclosed in compute comparisons.
9. Pareto frontier minimizes actual archive size and maximizes test accuracy.
   Hypervolume uses one shared reference (1.05 x largest archive, accuracy floor
   zero) and reports MB x percentage points. `hv_gain` is union(baselines,
   adaptive) minus all baselines; do not call it proof that every adaptive point
   dominates every uniform point. The final frontier is descriptive test-set
   reporting, not a way to retune allocation on the test labels.
10. Each result includes floating-point batch latency after archive decompression,
    with warm-up and CUDA synchronization. It is **not** low-bit kernel latency.
    The archive uses pickle: load only your own trusted experiment artifacts.

## 6. Finish the submission honestly

Run the real dataset command, inspect `results.csv` and plots, then insert the
measured results into the report. Include limitations and unsuccessful adaptive
points. Select configurations by validation accuracy under an actual size cap,
then describe their test performance. The slide's 95%, 1-3% gains, 5x layer
heterogeneity and Q4 NPU advantage are hypotheses/illustrations, not supplied
experimental results. Marks and Pareto dominance cannot be guaranteed.

If your instructor expects activation quantization, native torchao conversion,
or physically removed channels, this implementation needs that extension;
confirm that scope before describing those capabilities in a submission.
