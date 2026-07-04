"""
Feature Map SNR/CNR Analysis Script (Three-Stage Decoupled Architecture)
=========================================================================
Extracts heatmap data and computes SNR/CNR metrics for three feature stages
defined in odtrack.py:
  - F_base (feat_1_input):  Baseline feature from ViT Backbone, before IDF module
  - F_idf  (feat_2_idf):    IDF enhanced feature (dual-branch 3x3/5x5 + fusion)
  - F_tgg  (feat_3_dfr):    TGG final feature (IDF × template-guided gate + residual)

Three independent stages (controlled by RUN_STAGE_1/2/3 flags):

  Stage 1 - Predict & Save:
    Run tracker for entire dataset, save raw visualization_data + pred_box_norm.
    Output: experiment/{dataset}/heatmap/{video}/frame_{NNNNNN}/raw_data.npz

  Stage 2 - Select Paper Frames:
    Load saved raw data, compute focus/SNR/CNR scores, select top-5 frames.
    Output: experiment/{dataset}/heatmap_for_paper/*.png

  Stage 3 - Compute SNR/CNR:
    Load saved raw data, compute SNR/CNR for entire dataset, save CSVs & charts.
    Output: experiment/{dataset}/SNR_CNR/{summary.csv, SNR_CNR_paper.csv, img/}


Usage:
    python feature_snr_analysis.py
"""

import os
import sys
import warnings
import importlib
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import cv2
from tqdm import tqdm

# Add project root to path
prj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import run_config
from lib.test.evaluation.tracker import Tracker
import lib.test.evaluation.lasotdataset as lasot_dataset_module

# ======================== Configuration ========================
DATASET_NAMES = ['Anti-UAV410', 'CST-AntiUAV']
TRACKER_NAME = 'odtrack'
TRACKER_PARAM = 'baseline'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'heatmap_feature_experiment')

MAX_VIDEOS = 0          # Max videos per dataset (0 = all)
FRAME_INTERVAL = 1      # Process every N-th frame (1 = all frames)
FEAT_SZ = 16            # Feature map spatial size (SEARCH_SIZE // STRIDE = 256 // 16)
HEATMAP_UPSAMPLE_SIZE = 256

# Feature type mapping:
#   (output_suffix, visualization_data_key, display_label)
FEATURE_TYPES = [
    ('baseline', 'feat_1_input', r'$F_{base}$'),
    ('idf',      'feat_2_idf',   r'$F_{idf}$'),
    ('tgg',      'feat_3_dfr',   r'$F_{tgg}$'),
]

TOP_K_FRAMES = 5        # Number of contrast-progressive frames to keep for paper

# Run stages (set to False to skip after first run)
RUN_STAGE_1 = True   # Predict & save raw data
RUN_STAGE_2 = True   # Select paper frames
RUN_STAGE_3 = True   # Compute SNR/CNR

NUM_WORKERS = 16      # Thread pool size for Stage 2/3 parallel computation
# ===============================================================


# -------------------- Utility Functions --------------------

def load_lasot_dataset(dataset_name):
    """Load LaSOT-format dataset by temporarily switching run_config.dataname."""
    run_config.dataname = dataset_name
    importlib.reload(lasot_dataset_module)
    return lasot_dataset_module.LaSOTDataset().get_sequence_list()


def create_target_mask(feat_sz, pred_box_norm):
    """
    Create a boolean mask for the target region in the low-res feature map.
    pred_box_norm: [cx, cy, w, h] in [0, 1] relative to search region.
    """
    cx, cy, w, h = pred_box_norm
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))

    cx_f = cx * feat_sz
    cy_f = cy * feat_sz
    w_f = max(w * feat_sz, 1.0)
    h_f = max(h * feat_sz, 1.0)

    mask = np.zeros((feat_sz, feat_sz), dtype=bool)
    x1 = int(max(np.floor(cx_f - w_f / 2), 0))
    x2 = int(min(np.ceil(cx_f + w_f / 2), feat_sz))
    y1 = int(max(np.floor(cy_f - h_f / 2), 0))
    y2 = int(min(np.ceil(cy_f + h_f / 2), feat_sz))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    mask[y1:y2, x1:x2] = True
    return mask


def compute_snr_cnr(feat_map, target_mask):
    """
    面向真实特征响应分布的 SNR / CNR 计算。
    必须传入未经 Min-Max 归一化的原始特征图 (如果是多通道，需先沿通道维度做均值或最大值聚合)。
    """
    if feat_map.size == 0:
        return 0.0, 0.0
        
    # 为了防止网络输出存在负数（例如未过激活层的残差特征），可以取绝对值或 ReLU
    # 如果明确特征图已经是非负的，这步可以省略
    feat_map = np.abs(feat_map) 

    signal = feat_map[target_mask]
    background = feat_map[~target_mask]

    if signal.size == 0 or background.size == 0:
        return 0.0, 0.0

    mu_s = float(np.mean(signal))
    mu_b = float(np.mean(background))
    sigma_b = float(np.std(background))
    eps = 1e-8

    # 标准 CNR 定义：目标与背景均值的距离，除以背景的标准差
    cnr = np.abs(mu_s - mu_b) / (sigma_b + eps)
    
    # 图像/雷达信号处理中更标准的 SNR 形式（对数尺度，dB）
    # 如果非要用原始比值，可直接返回 mu_s / (mu_b + eps)
    # 审稿人通常更认可 dB 形式的 SNR
    snr_ratio = mu_s / (sigma_b + eps) 
    snr_db = 10 * np.log10(snr_ratio + eps)

    return snr_db, cnr


def process_feature_map(feat, feat_sz, target_size=HEATMAP_UPSAMPLE_SIZE):
    """
    Process raw feature tokens into a smooth 256x256 heatmap, aligned with
    the pipeline in heat_map.py.

    Input feat shape: [B, HW, C] or [HW, C].
    Output: [target_size, target_size] float32 heatmap in [0, 1].
    """
    if feat.ndim == 3:
        feat = feat[0]

    if feat.ndim == 2:
        HW, C = feat.shape
        side = int(np.sqrt(HW))
        feat_2d = feat.reshape(side, side, C)
    elif feat.ndim == 3:
        feat_2d = feat
    else:
        raise ValueError(f"Unsupported feature ndim: {feat.ndim}")

    # 使用逐通道 L2 范数替代绝对值均值。
    # L2 范数能保留特征的方向能量，避免绝对值均值在门控/残差特征
    # (如 F_tgg) 上把负向抑制信号也累加为“正激活”，导致目标区能量
    # 与背景区能量被拉平、SNR/CNR 反而低于中间层 F_idf 的假象。
    heatmap = np.sqrt(np.sum(feat_2d.astype(np.float32) ** 2, axis=2))
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC
    )
    return heatmap


# -------------------- Frame Scoring & Selection --------------------

def compute_focus_score(heatmap, target_mask_up):
    """Focus score: target energy / total energy."""
    target_mask_up = target_mask_up.astype(bool)
    if target_mask_up.sum() == 0:
        return 0.0
    energy_in = float(heatmap[target_mask_up].sum())
    energy_total = float(heatmap.sum()) + 1e-8
    return energy_in / energy_total


def select_top_k_frames(frame_records, k=TOP_K_FRAMES):
    """Select k frames with baseline < idf < tgg improvement."""
    candidates = []
    for rec in frame_records:
        f_base = rec['focus_baseline']
        f_idf = rec['focus_idf']
        f_tgg = rec['focus_tgg']

        c_base = rec['CNR_baseline']
        c_idf = rec['CNR_idf']
        c_tgg = rec['CNR_tgg']

        if not (f_base < f_idf < f_tgg):
            continue
        if not (c_base < c_idf < c_tgg):
            continue

        score = (c_tgg - c_base) + 2.0 * (f_tgg - f_base)
        rec_copy = rec.copy()
        rec_copy['selection_score'] = score
        candidates.append(rec_copy)

    if not candidates:
        for rec in frame_records:
            f_base = rec['focus_baseline']
            f_idf = rec['focus_idf']
            f_tgg = rec['focus_tgg']
            if not (f_base < f_idf < f_tgg):
                continue
            score = f_tgg - f_base
            rec_copy = rec.copy()
            rec_copy['selection_score'] = score
            candidates.append(rec_copy)

    candidates.sort(key=lambda x: x['selection_score'], reverse=True)
    return candidates[:k]


# -------------------- Visualization Helpers --------------------

def _draw_target_rect(ax, target_mask, upsample_size=None):
    """Draw a dashed white rectangle around the target region.

    使用分离的 x/y 缩放比例，保证在非正方形特征图下也能正确对齐；
    使用像素边界坐标 (x1, y1)-(x2+1, y2+1) 保证与 imshow 的像素对齐一致。
    """
    ys, xs = np.where(target_mask)
    if len(ys) == 0:
        return
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    if upsample_size is not None:
        h, w = target_mask.shape
        scale_x = upsample_size / float(w)
        scale_y = upsample_size / float(h)
        x1f = x1 * scale_x
        x2f = (x2 + 1) * scale_x
        y1f = y1 * scale_y
        y2f = (y2 + 1) * scale_y
    else:
        x1f, y1f = x1, y1
        x2f, y2f = x2 + 1, y2 + 1

    # imshow 的像素中心是整数坐标，像素边界在 x-0.5 / x+0.5，
    # 因此这里从缩放后的边界再减 0.5 得到真实的像素外沿。
    rect = mpatches.Rectangle(
        (x1f - 0.5, y1f - 0.5), x2f - x1f, y2f - y1f,
        linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    ax.add_patch(rect)


def save_paper_figure(feat_maps, target_mask, save_path, titles, dpi=300):
    """Save a 1x3 combined figure for paper."""
    n = len(feat_maps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([f.ravel() for f in feat_maps])
    gmin, gmax = all_vals.min(), all_vals.max()

    for ax, feat, title in zip(axes, feat_maps, titles):
        feat_norm = (feat - gmin) / (gmax - gmin + 1e-8)
        im = ax.imshow(feat_norm, cmap='jet', interpolation='bicubic', vmin=0, vmax=1)
        _draw_target_rect(ax, target_mask, upsample_size=HEATMAP_UPSAMPLE_SIZE)
        ax.set_title(title, fontsize=16)
        ax.axis('off')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_summary_charts(df, img_dir):
    """Generate bar/box charts from SNR/CNR results."""
    feature_order = ['baseline', 'idf', 'tgg']
    feature_labels = [r'$F_{base}$', r'$F_{idf}$', r'$F_{tgg}$']
    colors = ['#4C72B0', '#55A868', '#C44E52']

    os.makedirs(img_dir, exist_ok=True)

    # SNR bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    snr_means = df.groupby('feature_type')['SNR'].mean().reindex(feature_order)
    bars = ax.bar(feature_labels, snr_means.values, color=colors, width=0.5)
    ax.set_ylabel('SNR', fontsize=14)
    ax.set_title('Average SNR Across Feature Stages', fontsize=16)
    for bar, v in zip(bars, snr_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01 * max(abs(snr_means.values)),
                f'{v:.3f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'snr_bar.png'), dpi=200)
    plt.close(fig)

    # CNR bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    cnr_means = df.groupby('feature_type')['CNR'].mean().reindex(feature_order)
    bars = ax.bar(feature_labels, cnr_means.values, color=colors, width=0.5)
    ax.set_ylabel('CNR', fontsize=14)
    ax.set_title('Average CNR Across Feature Stages', fontsize=16)
    for bar, v in zip(bars, cnr_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01 * max(abs(cnr_means.values)),
                f'{v:.3f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'cnr_bar.png'), dpi=200)
    plt.close(fig)

    # SNR box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    snr_data = [df[df['feature_type'] == ft]['SNR'].values for ft in feature_order]
    bp = ax.boxplot(snr_data, labels=feature_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('SNR', fontsize=14)
    ax.set_title('SNR Distribution Across Feature Stages', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'snr_box.png'), dpi=200)
    plt.close(fig)

    # CNR box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cnr_data = [df[df['feature_type'] == ft]['CNR'].values for ft in feature_order]
    bp = ax.boxplot(cnr_data, labels=feature_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('CNR', fontsize=14)
    ax.set_title('CNR Distribution Across Feature Stages', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'cnr_box.png'), dpi=200)
    plt.close(fig)

    # SNR / CNR trend line
    fig, ax1 = plt.subplots(figsize=(10, 6))
    snr_avg = df.groupby('feature_type')['SNR'].mean().reindex(feature_order).values
    cnr_avg = df.groupby('feature_type')['CNR'].mean().reindex(feature_order).values

    ax1.plot(range(3), snr_avg, 'o-', color='#4C72B0', linewidth=2, markersize=10, label='SNR')
    ax1.set_ylabel('SNR', fontsize=14, color='#4C72B0')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(feature_labels, fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(range(3), cnr_avg, 's-', color='#C44E52', linewidth=2, markersize=10, label='CNR')
    ax2.set_ylabel('CNR', fontsize=14, color='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    ax1.set_title('SNR / CNR Trend Across Feature Stages', fontsize=16)
    ax1.set_xlabel('Feature Stage', fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'snr_cnr_trend.png'), dpi=200)
    plt.close(fig)




# -------------------- Data Loading Helper --------------------

def _load_all_raw_frames(dataset_name, experiment_dir):
    """Scan saved raw data for a dataset, return list of frame info dicts."""
    heatmap_root = os.path.join(experiment_dir, dataset_name, 'heatmap')
    if not os.path.isdir(heatmap_root):
        print(f"  Warning: {heatmap_root} does not exist. Run Stage 1 first.")
        return []

    frames = []
    for video_name in sorted(os.listdir(heatmap_root)):
        video_dir = os.path.join(heatmap_root, video_name)
        if not os.path.isdir(video_dir):
            continue
        for frame_name in sorted(os.listdir(video_dir)):
            frame_dir = os.path.join(video_dir, frame_name)
            raw_path = os.path.join(frame_dir, 'raw_data.npz')
            if not os.path.isfile(raw_path):
                continue
            frames.append({
                'video': video_name,
                'frame_name': frame_name,
                'raw_path': raw_path,
            })
    return frames


# -------------------- Stage 1: Predict & Save --------------------

def stage1_predict_and_save(dataset_name, experiment_dir):
    """Stage 1: Run tracker for entire dataset, save raw visualization_data and pred_box_norm.

    Output: experiment/{dataset_name}/heatmap/{video_name}/frame_{NNNNNN}/raw_data.npz
    """
    print(f"\n=== Stage 1: Predict & Save Raw Data ===")
    print(f"[1/3] Loading dataset: {dataset_name}")
    dataset = load_lasot_dataset(dataset_name)

    print(f"[2/3] Initializing tracker: {TRACKER_NAME}/{TRACKER_PARAM}")
    tracker_wrapper = Tracker(TRACKER_NAME, TRACKER_PARAM, dataset_name)
    params = tracker_wrapper.get_parameters()
    params.debug = 0

    heatmap_root = os.path.join(experiment_dir, dataset_name, 'heatmap')
    os.makedirs(heatmap_root, exist_ok=True)

    n_videos = len(dataset)
    if MAX_VIDEOS > 0:
        n_videos = min(n_videos, MAX_VIDEOS)
    print(f"[3/3] Tracking {n_videos} videos ...")

    for seq_idx in range(n_videos):
        seq = dataset[seq_idx]
        video_name = seq.name
        print(f"  ({seq_idx + 1}/{n_videos}) Video: {video_name}")

        # Skip already-processed videos (folder exists)
        video_dir = os.path.join(heatmap_root, video_name)
        if os.path.isdir(video_dir):
            print(f"    -> Already processed, skipping.")
            continue

        output = tracker_wrapper.run_sequence(seq, debug=0)
        vis_data_list = output.get('visualization_data', [])
        pred_box_list = output.get('pred_box_norm', [])

        if not vis_data_list:
            print(f"    -> No visualization data, skipping.")
            continue

        n_frames = len(vis_data_list)
        frame_indices = list(range(0, n_frames, FRAME_INTERVAL))
        if not frame_indices:
            frame_indices = [0]

        saved_count = 0
        for fidx in frame_indices:
            vis_data = vis_data_list[fidx]
            pred_box_norm = pred_box_list[fidx] if fidx < len(pred_box_list) else None

            if vis_data is None or not vis_data or pred_box_norm is None:
                continue

            # Check all required feature keys exist
            missing = False
            for _, feat_key, _ in FEATURE_TYPES:
                if feat_key not in vis_data:
                    missing = True
                    break
            if missing:
                continue

            frame_num = fidx + 1
            frame_name = f"frame_{frame_num:06d}"
            frame_dir = os.path.join(heatmap_root, video_name, frame_name)
            os.makedirs(frame_dir, exist_ok=True)

            # Save raw visualization_data + pred_box_norm
            save_dict = {
                'pred_box_norm': np.array(pred_box_norm),
                'frame_num': frame_num,
                'video': video_name,
            }
            for _, feat_key, _ in FEATURE_TYPES:
                feat = vis_data[feat_key]
                if hasattr(feat, 'cpu'):          # torch tensor -> numpy
                    feat = feat.cpu().numpy()
                save_dict[feat_key] = feat

            np.savez_compressed(
                os.path.join(frame_dir, 'raw_data.npz'), **save_dict
            )
            saved_count += 1

        print(f"    -> {n_frames} tracked frames, saved {saved_count} -> {heatmap_root}")


# -------------------- Stage 2: Select Paper Frames --------------------

def _safe_np_load(raw_path):
    """Load a .npz safely; return None (and remove the file) if it's corrupt/empty."""
    try:
        if os.path.getsize(raw_path) == 0:
            raise zipfile.BadZipFile("zero-byte file")
        return np.load(raw_path, allow_pickle=True)
    except (zipfile.BadZipFile, EOFError, OSError, ValueError) as e:
        print(f"  [WARN] Broken npz removed: {raw_path} ({e})")
        try:
            os.remove(raw_path)
        except OSError:
            pass
        return None


def _compute_frame_scores(fr, dataset_name):
    """Pass 1: Compute selection scores for a single frame (lightweight, no heatmaps stored)."""
    data = _safe_np_load(fr['raw_path'])
    if data is None:
        return None
    pred_box_norm = data['pred_box_norm']
    frame_num = int(data['frame_num'])

    stage_heatmaps = {}
    stage_lowres = {}
    for feat_suffix, feat_key, _ in FEATURE_TYPES:
        feat = data[feat_key]
        heatmap = process_feature_map(feat, FEAT_SZ)
        stage_heatmaps[feat_suffix] = heatmap

        if feat.ndim == 3:
            feat = feat[0]
        side = int(np.sqrt(feat.shape[0]))
        # 与 process_feature_map 保持一致：使用逐通道 L2 范数作为能量图，
        # 保证 SNR/CNR 与可视化热力图的能量定义完全一致。
        lowres = np.sqrt(
            np.sum(feat.astype(np.float32) ** 2, axis=-1)
        ).reshape(side, side)
        stage_lowres[feat_suffix] = lowres

    target_mask_lr = create_target_mask(FEAT_SZ, pred_box_norm)
    target_mask_up = cv2.resize(
        target_mask_lr.astype(np.uint8),
        (HEATMAP_UPSAMPLE_SIZE, HEATMAP_UPSAMPLE_SIZE),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    record = {
        'dataset': dataset_name,
        'video': fr['video'],
        'frame': frame_num,
        'raw_path': fr['raw_path'],
    }
    for feat_suffix, _, _ in FEATURE_TYPES:
        snr, cnr = compute_snr_cnr(stage_lowres[feat_suffix], target_mask_lr)
        record[f'SNR_{feat_suffix}'] = snr
        record[f'CNR_{feat_suffix}'] = cnr
        record[f'focus_{feat_suffix}'] = compute_focus_score(
            stage_heatmaps[feat_suffix], target_mask_up
        )
    # NOTE: heatmaps are NOT stored to avoid OOM on large datasets
    return record


def _render_paper_figure(rec, dataset_name, paper_dir):
    """Pass 2: Reload a single selected frame and render its paper figure."""
    data = _safe_np_load(rec['raw_path'])
    if data is None:
        return
    pred_box_norm = data['pred_box_norm']
    target_mask_lr = create_target_mask(FEAT_SZ, pred_box_norm)

    feat_maps = []
    for _, feat_key, _ in FEATURE_TYPES:
        feat_maps.append(process_feature_map(data[feat_key], FEAT_SZ))

    frame_name = f"frame_{rec['frame']:06d}"
    base_name = f"{dataset_name}_{rec['video']}_{frame_name}"
    save_path = os.path.join(paper_dir, f"{base_name}_for_paper.png")
    save_paper_figure(
        feat_maps,
        target_mask_lr,
        save_path,
        [label for _, _, label in FEATURE_TYPES]
    )


def stage2_select_paper_frames(dataset_name, experiment_dir):
    """Stage 2: Load saved raw data, select top-5 frames, save paper figures.

    Two-pass design to minimize memory:
      Pass 1 - compute scores only (no heatmaps kept in memory)
      Pass 2 - reload only the selected top-k frames for rendering

    Output: experiment/{dataset_name}/heatmap_for_paper/*.png
    """
    print(f"\n=== Stage 2: Select Paper Frames ===")

    paper_dir = os.path.join(experiment_dir, dataset_name, 'heatmap_for_paper')
    os.makedirs(paper_dir, exist_ok=True)

    # Skip if already completed (output files exist)
    existing = [f for f in os.listdir(paper_dir) if f.endswith('.png')]
    if existing:
        print(f"  Already completed ({len(existing)} figures found), skipping.")
        return

    frames = _load_all_raw_frames(dataset_name, experiment_dir)
    if not frames:
        print("  No frames found. Run Stage 1 first.")
        return

    print(f"  Loaded {len(frames)} frames")
    print(f"  Pass 1: Computing scores (no heatmaps stored) ...")

    # Pass 1: compute scores only — lightweight records
    frame_records = [None] * len(frames)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_idx = {
            executor.submit(_compute_frame_scores, fr, dataset_name): idx
            for idx, fr in enumerate(frames)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(frames),
                           desc="  Scoring", unit="frame"):
            idx = future_to_idx[future]
            frame_records[idx] = future.result()

    # Drop broken frames (returned None by _compute_frame_scores)
    frame_records = [r for r in frame_records if r is not None]
    if not frame_records:
        print("  No valid frames after filtering broken npz files.")
        return

    # Select top-k
    top_k_frames = select_top_k_frames(frame_records, k=TOP_K_FRAMES)

    if not top_k_frames:
        print("  Warning: no frames satisfy monotonic improvement criterion.")
        return

    # Pass 2: reload only k frames for rendering
    print(f"  Pass 2: Rendering {len(top_k_frames)} selected figures ...")
    for rec in tqdm(top_k_frames, desc="  Saving", unit="fig"):
        _render_paper_figure(rec, dataset_name, paper_dir)
    print(f"  Saved {len(top_k_frames)} paper figures -> {paper_dir}")


# -------------------- Stage 3: Compute SNR/CNR --------------------

def _compute_snr_cnr_for_frame(fr, dataset_name):
    """Compute SNR/CNR for a single raw frame, return list of result dicts (thread-safe)."""
    data = _safe_np_load(fr['raw_path'])
    if data is None:
        return []
    pred_box_norm = data['pred_box_norm']
    frame_num = int(data['frame_num'])
    video_name = fr['video']

    target_mask_lr = create_target_mask(FEAT_SZ, pred_box_norm)

    results = []
    for feat_suffix, feat_key, feat_label in FEATURE_TYPES:
        feat = data[feat_key]
        if feat.ndim == 3:
            feat = feat[0]
        side = int(np.sqrt(feat.shape[0]))
        # 与 process_feature_map / _compute_frame_scores 保持一致：
        # 使用逐通道 L2 范数作为能量图。
        lowres = np.sqrt(
            np.sum(feat.astype(np.float32) ** 2, axis=-1)
        ).reshape(side, side)

        snr, cnr = compute_snr_cnr(lowres, target_mask_lr)
        results.append({
            'dataset': dataset_name,
            'video': video_name,
            'frame': frame_num,
            'feature_type': feat_suffix,
            'feature_label': feat_label,
            'SNR': snr,
            'CNR': cnr,
            'pred_box_norm': str(pred_box_norm),
        })
    return results


def stage3_compute_snr_cnr(dataset_name, experiment_dir):
    """Stage 3: Load saved raw data, compute SNR/CNR for entire dataset, save CSVs & charts.

    Output: experiment/{dataset_name}/SNR_CNR/summary.csv, SNR_CNR_paper.csv, img/*.png
    """
    print(f"\n=== Stage 3: Compute SNR/CNR ===")

    snr_cnr_dir = os.path.join(experiment_dir, dataset_name, 'SNR_CNR')
    img_dir = os.path.join(snr_cnr_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Skip if already completed (summary CSV exists)
    summary_path = os.path.join(snr_cnr_dir, 'summary.csv')
    if os.path.isfile(summary_path):
        print(f"  Already completed (summary.csv found), skipping.")
        return

    frames = _load_all_raw_frames(dataset_name, experiment_dir)
    if not frames:
        print("  No frames found. Run Stage 1 first.")
        return

    print(f"  Loaded {len(frames)} frames, computing SNR/CNR ...")

    # Process in batches to bound peak memory
    BATCH_SIZE = 2000
    all_results = []
    for batch_start in tqdm(range(0, len(frames), BATCH_SIZE),
                            desc="  SNR/CNR", unit="batch"):
        batch = frames[batch_start:batch_start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(_compute_snr_cnr_for_frame, fr, dataset_name)
                for fr in batch
            ]
            for future in as_completed(futures):
                all_results.extend(future.result())

    if not all_results:
        print("  No results computed.")
        return

    df = pd.DataFrame(all_results)

    # Detailed CSV
    summary_path = os.path.join(snr_cnr_dir, 'summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"  Summary CSV      : {summary_path}")

    # Aggregated paper CSV
    paper_df = (
        df.groupby(['dataset', 'feature_type', 'feature_label'])[['SNR', 'CNR']]
          .agg(['mean', 'std'])
          .reset_index()
    )
    paper_path = os.path.join(snr_cnr_dir, 'SNR_CNR_paper.csv')
    paper_df.to_csv(paper_path, index=False)
    print(f"  Paper CSV        : {paper_path}")

    # Charts
    generate_summary_charts(df, img_dir)
    print(f"  Charts saved to  : {img_dir}")


# -------------------- Main Pipeline --------------------

def main():
    warnings.filterwarnings('ignore', category=UserWarning)

    experiment_dir = os.path.join(OUTPUT_DIR, 'experiment')
    os.makedirs(experiment_dir, exist_ok=True)

    for dataset_idx, dataset_name in enumerate(DATASET_NAMES, start=1):
        print(f"\n{'=' * 60}")
        print(f"[Dataset {dataset_idx}/{len(DATASET_NAMES)}] {dataset_name}")
        print(f"{'=' * 60}")

        if RUN_STAGE_1:
            stage1_predict_and_save(dataset_name, experiment_dir)
        # Stage 3 (整体 SNR/CNR 统计) 先跑，Stage 2 (精选展示帧) 后跑，
        # 便于先看到定量指标是否正常，再基于结果挑选展示帧。
        if RUN_STAGE_3:
            stage3_compute_snr_cnr(dataset_name, experiment_dir)
        if RUN_STAGE_2:
            stage2_select_paper_frames(dataset_name, experiment_dir)

    print("\n=== Analysis complete! ===")


if __name__ == '__main__':
    main()
