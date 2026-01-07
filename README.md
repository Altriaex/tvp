<br>
<div align="center">
<h1>Transductive Visual Programming (TVP)</h1>
<h3>Evolving Tool Libraries from Experience for Spatial Reasoning</h3>

[![Website](https://img.shields.io/badge/Website-TVP%20Project%20Page-blue)](https://transductive-visualprogram.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2512.20934-blue)](https://arxiv.org/abs/2512.20934)

</div>

---

**TVP** is a visual programming framework that builds reusable tools from its own problem-solving experience. TVP implements a closed-loop paradigm via two interconnected libraries: an <b>Example Library</b> that accumulates program solutions as experience, and a <b>Tool Library</b> that maintains functions abstracted from these programs. The dual-libraries enable the circular <b>program-tool-program</b> cycle: solving problems generates experience, experience guides tool creation, and newly created tools improve future problem-solving.

<p align="center">
  <img src="assets/method_two_phase_cycle.png" width="100%" alt="TVP Method Overview">
</p>

---

<a name="updates"></a>

## 📢 Updates

- **[2026-01-06]** Check out our [X thread](https://x.com/ShengguangWu/status/2008713289568596267)
- **[2025-12-24]** Code released.
- **[2025-12-23]** Paper available on [arXiv](https://arxiv.org/abs/2512.20934).

---

## 🚀 Quick Start

### Environment Setup

Create a conda environment and run the setup script:

```bash
conda create -n tvp python=3.10 -y
conda activate tvp
bash env_setup.sh
```

### API Key Setup

Create a file named `api.key` in the project root with your OpenAI API key:

```bash
echo "your-openai-api-key" > api.key
```

### Download Omni3D-Bench

[Omni3D-Bench](https://huggingface.co/datasets/dmarsili/Omni3D-Bench) contains 501 challenging 3D spatial reasoning questions on real-world indoor scenes. Download and prepare the dataset:

```bash
mkdir -p data
huggingface-cli download dmarsili/Omni3D-Bench --repo-type dataset --include "omni3d-bench.zip" --local-dir ./data
unzip ./data/omni3d-bench.zip -d ./data
rm -rf ./data/.cache ./data/omni3d-bench.zip
```

After setup, your data directory should look like:
```
data/
└── omni3d-bench/
    ├── images/
    │   └── ARKitScenes/
    │       └── Training/
    │           └── .../*.jpg
    └── annotations.json
```

---

## 🔬 Running TVP

### Toy Run (Quick Demo)

To quickly see how TVP works, run a toy subset on 10 datapoints with relaxed thresholds. This demonstrates all pipeline components—example accumulation, tool abstraction, and deduplication—in a short run:

```bash
python main.py \
    --resume tvp_toy_subset \
    --num-questions 10 \
    --num-iterations 2 \
    --programs-per-question 2 \
    --quality-threshold 8.0 \
    --similarity-threshold 0.5 \
    --abstraction-interval 1 \
    --clustering-threshold 0.5 \
    --min-cluster-size 2 \
    --abstraction-potential-threshold 5.0 \
    --min-correctness-rate 0.5 \
    --deduplication-interval 1 \
    --deduplication-threshold 0.6 \
    --save-exe-trace
```

Results will be saved to `results/tvp_toy_subset/`. The `--save-exe-trace` flag generates HTML visualizations of each program execution for inspection.

### Complete Run

To experiment on the full Omni3D-Bench dataset:

```bash
python main.py \
    --resume tvp_omni3d_full \
    --num-questions -1 \
    --num-iterations 3 \
    --programs-per-question 4 \
    --quality-threshold 8.5 \
    --similarity-threshold 0.8 \
    --abstraction-interval 1 \
    --clustering-threshold 0.8 \
    --min-cluster-size 4 \
    --abstraction-potential-threshold 9.0 \
    --min-correctness-rate 0.85 \
    --deduplication-interval 1 \
    --deduplication-threshold 0.95
```

**Estimated cost:** ~$80 per iteration with GPT-4o + o4-mini.

See `python main.py --help` for all available configuration options.

### 🔄 Resumability

TVP automatically saves checkpoints after processing each question. To resume an interrupted run:

```bash
python main.py --resume your_run_id
```

State files are stored in `results/your_run_id/state/`.

---

## 📈 Evaluation

Evaluation is a two-step process: (1) extract results from the TVP pipeline run, and (2) compute metrics.

### Step 1: Extract Results

After running TVP, extract solutions and associated metadata into a single JSON file:

```bash
python eval/extract_results.py --run-id tvp_omni3d_full
```

This reads from `results/tvp_omni3d_full/` and outputs to `eval/extracted_results/tvp_omni3d_full.json`.

### Step 2: Compute Metrics

Evaluate metrics on the extracted results:

```bash
python eval/evaluate_metrics.py --run-id tvp_omni3d_full
```

This computes per-iteration metrics and saves detailed results to `eval/metrics/tvp_omni3d_full/`.

### Metrics

The evaluation reports the following metrics by question type:

| Question Type | Metric |
|---------------|--------|
| **Yes/No** | Accuracy |
| **Multiple Choice** | Accuracy |
| **Count** | Accuracy (exact match) |
| **Float** | Accuracy (±10% relative error) |
| **Float** | MRA (Mean Relative Accuracy across thresholds 5%–50%) |
| **Overall** | Accuracy (using Float Accuracy ±10%) |

Results are saved as both a summary JSON and per-iteration CSV files with details on each question.

---

## Citation

If you find TVP useful, please consider citing our paper:

```bibtex
@article{wu2025transductive,
  title={Transductive Visual Programming: Evolving Tool Libraries from Experience for Spatial Reasoning},
  author={Wu, Shengguang and Wang, Xiaohan and Zhang, Yuhui and Zhu, Hao and Yeung-Levy, Serena},
  journal={arXiv preprint arXiv:2512.20934},
  year={2025}
}
```

---

## Acknowledgments

TVP builds upon several open-source projects. We thank the authors for their codebase: [VADAR](https://github.com/damianomarsili/VADAR), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).
