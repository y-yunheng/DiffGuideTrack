This is a professional and structured `README.md` for your GitHub repository, tailored specifically for **DiffGuideTrack** (the enhanced version of ODTrack). I have integrated your paper's specific modules (IDF, TGG) and your provided download links.

---

# DiffGuideTrack: Template-Guided Feature Refinement for Infrared Small Target Tracking with Vision Transformers

Official implementation of **DiffGuideTrack**, an advanced infrared small target tracker built upon the one-stream Vision Transformer paradigm. DiffGuideTrack introduces two synergistic modules—**Inverted Difference Fusion (IDF)** and **Template-Guided Gating (TGG)**—to overcome the "feature dilution" problem in harsh infrared environments.

## Table of Contents
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Resources & Downloads](#resources--downloads)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Features
- **Inverted Difference Fusion (IDF):** Enhances local target contrast using multi-scale differential responses to reinforce weak infrared signals.
- **Template-Guided Gating (TGG):** A dynamic channel-wise filter that suppresses background clutter using global descriptors from the template.
- **State-of-the-Art Tracking:** Specifically optimized for **Anti-UAV410** and **CST-AntiUAV** datasets.
- **High Efficiency:** Maintains real-time inference speed (~60 FPS) on an NVIDIA RTX 3090.

---

## Project Architecture
The framework is built on top of the ODTrack backbone, with the following additions:
- **IDF Module:** Located before the features enter the Transformer or as a refinement stage.
- **TGG Module:** Leverages template identity to guide search region feature selection.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/DiffGuideTrack.git
   cd DiffGuideTrack
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize local environment:**
   Create `lib/train/admin/local.py` by running:
   ```bash
   python tracking/create_default_local_file.py
   ```

---

## Resources & Downloads

| Resource Type | Description | Link | Extraction Code |
| :--- | :--- | :--- | :--- |
| **Pre-trained ViT** | MAE/ViT backbone weights | [Baidu Pan](https://pan.baidu.com/s/1MhtHwN8Vvof9pZBTA1YMzQ?pwd=qieq) | `qieq` |
| **Trained Models** | Trained `odtrack_IDF` and final weights | [Baidu Pan](https://pan.baidu.com/s/1ncy5dksAGKKcttkgQAy2Bw?pwd=cxzt) | `cxzt` |
| **Datasets** | Anti-UAV410 & CST-AntiUAV | [Baidu Pan](https://pan.baidu.com/s/1C9e7_cJ7gXDzgI1CnUpONg?pwd=a684 | `a684`|
--来自百度网盘超级会员v9的分享) | N/A |

---

## Dataset Setup

The framework expects datasets in **LaSOT** format. Organize your `datasets` directory as follows:

```text
datasets/
├── Anti-UAV410/
│   └── lasot/
│       ├── train/
│       │   ├── uav-1/
│       │   └── ... (video sequences)
│       └── val/
├── CST-AntiUAV/
│   └── lasot/
│       ├── train/
│       └── val/
```

Update your `lib/train/admin/local.py` with the absolute paths to these directories.

---

## Training

### Automated Training & Evaluation
We provide a `run.py` script that handles the end-to-end process (Dataset switching -> Training -> Testing -> Analysis):

```bash
python run.py
```

### Manual Training
**Single GPU:**
```bash
python tracking/train.py --mode single --script odtrack --config baseline --save_dir ./experiments/DiffGuide
```

**Multi-GPU (DDP):**
```bash
python tracking/train.py --mode multiple --nproc_per_node 2 --script odtrack --config baseline --save_dir ./experiments/DiffGuide
```

---

## Evaluation

### Testing
Run evaluation on the infrared benchmarks:
```bash
python tracking/test.py odtrack baseline --dataset_name lasot --threads 0 --num_gpus 1
```

### Results Analysis
To generate Success and Precision plots:
```bash
python tracking/analysis_results.py
```

---

## Project Structure
- `lib/models/odtrack/`: Core implementation of DiffGuideTrack, including IDF and TGG modules.
- `experiments/odtrack/`: YAML configuration files for different model variants.
- `tracking/`: Main entry points for training, testing, and visualization.
- `run.py`: Convenient script for batch experiments on IR datasets.

---

## Citation
If you find DiffGuideTrack useful for your research, please cite our paper:

```bibtex
@article{yi2025diffguidetrack,
  title={DiffGuideTrack: Template-Guided Feature Refinement for Infrared Small Target Tracking with Vision Transformers},
  author={Yi, Yunheng and Shi, Jingcheng and Chen, Shiguo and Tian, Chunna and Xu, Ying},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments
This code is based on [ODTrack](https://github.com/ZhengYuan-01/ODTrack) and [OSTrack](https://github.com/bnu-wanghe/OSTrack). We thank the authors for their excellent work.
