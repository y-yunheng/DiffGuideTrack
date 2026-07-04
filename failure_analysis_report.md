# 失败分析报告 (Failure & Robustness Analysis)

> 回应审稿人：报告 **逐序列跟踪失败率** 与 **TRAC 鲁棒性指标**（首次故障前的跟踪长度），支撑 DiffGuideTrack 相对 ODTrack 的优越性。

- 单帧成功阈值 IoU ≥ 0.5
- 序列级成功阈值 SuccessRate@0.5 ≥ 0.5
- 跟丢判定 = 连续 5 帧 IoU=0

## 1. 数据集级汇总（Failure Rate / Win-Loss / 平均鲁棒性）

| Dataset | #Seq | ODTrack Fail Rate | Ours Fail Rate | Ours_Win | Ours_Lose | Both_Success | Both_Fail | ODTrack AUC | Ours AUC | ODTrack Prec@20 | Ours Prec@20 | ODTrack TRAC-len | Ours TRAC-len | ODTrack TRAC-ratio | Ours TRAC-ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anti-UAV410 | 80 | 0.312 | 0.188 | 10 | 0 | 55 | 15 | 0.5162 | 0.5664 | 0.6934 | 0.7629 | 616.2 | 630.6 | 0.570 | 0.587 |
| CST-AntiUAV | 60 | 0.750 | 0.733 | 2 | 1 | 14 | 43 | 0.2780 | 0.2986 | 0.4538 | 0.4918 | 374.6 | 402.4 | 0.365 | 0.389 |

> **Ours_Win**: DiffGuideTrack 序列成功而 ODTrack 失败的序列数；**Ours_Lose**: 反之。**TRAC-len**: 首次连续跟丢前的跟踪帧数（越大越鲁棒）。

## 2. 精选展示序列（每数据集 Top-15)

> 优先级：`Ours_Win` > `Both_Success`(Δ 大) > `Both_Fail`(Ours 更晚跟丢) > `Ours_Lose`。

### 2.1 Anti-UAV410

| Dataset | Sequence | Frames | ODTrack_AUC | Ours_AUC | ΔAUC | ODTrack_Prec@20 | Ours_Prec@20 | ODTrack_TRAC_len | Ours_TRAC_len | ODTrack_TRAC_ratio | Ours_TRAC_ratio | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anti-UAV410 | 20190926_111509_1_2 | 1000 | 0.049 | 0.604 | 0.556 | 0.002 | 0.821 | 3 | 148 | 0.003 | 0.148 | Ours_Win |
| Anti-UAV410 | 20190925_193610_1_4 | 943 | 0.093 | 0.565 | 0.472 | 0.067 | 0.705 | 66 | 70 | 0.070 | 0.074 | Ours_Win |
| Anti-UAV410 | new10_train_newfix | 1491 | 0.054 | 0.429 | 0.375 | 0.013 | 0.579 | 19 | 19 | 0.013 | 0.013 | Ours_Win |
| Anti-UAV410 | 20190925_193610_1_5 | 948 | 0.067 | 0.427 | 0.360 | 0.028 | 0.527 | 28 | 28 | 0.030 | 0.030 | Ours_Win |
| Anti-UAV410 | 20190925_200805_1_1 | 950 | 0.175 | 0.522 | 0.347 | 0.205 | 0.736 | 15 | 101 | 0.016 | 0.106 | Ours_Win |
| Anti-UAV410 | 20190925_193610_1_3 | 950 | 0.261 | 0.558 | 0.296 | 0.311 | 0.761 | 54 | 106 | 0.057 | 0.112 | Ours_Win |
| Anti-UAV410 | 20190925_193610_1_8 | 921 | 0.322 | 0.612 | 0.290 | 0.379 | 0.817 | 355 | 355 | 0.385 | 0.385 | Ours_Win |
| Anti-UAV410 | 20190925_193610_1_1 | 926 | 0.269 | 0.499 | 0.230 | 0.298 | 0.625 | 73 | 73 | 0.079 | 0.079 | Ours_Win |
| Anti-UAV410 | 20190925_200805_1_6 | 862 | 0.413 | 0.564 | 0.152 | 0.510 | 0.748 | 84 | 108 | 0.097 | 0.125 | Ours_Win |
| Anti-UAV410 | 20190925_200805_1_4 | 940 | 0.356 | 0.414 | 0.057 | 0.494 | 0.559 | 148 | 148 | 0.157 | 0.157 | Ours_Win |
| Anti-UAV410 | 20190925_200805_1_9 | 475 | 0.411 | 0.568 | 0.157 | 0.514 | 0.731 | 20 | 82 | 0.042 | 0.173 | Both_Success |
| Anti-UAV410 | 20190925_134301_1_6 | 1000 | 0.379 | 0.486 | 0.106 | 0.587 | 0.731 | 41 | 41 | 0.041 | 0.041 | Both_Success |
| Anti-UAV410 | 20190925_134301_1_7 | 1000 | 0.634 | 0.722 | 0.088 | 0.846 | 0.984 | 632 | 632 | 0.632 | 0.632 | Both_Success |
| Anti-UAV410 | 20190925_200805_1_3 | 1000 | 0.478 | 0.565 | 0.087 | 0.698 | 0.839 | 416 | 298 | 0.416 | 0.298 | Both_Success |
| Anti-UAV410 | 20190925_134301_1_8 | 1000 | 0.664 | 0.735 | 0.070 | 0.888 | 0.999 | 868 | 1000 | 0.868 | 1.000 | Both_Success |

### 2.2 CST-AntiUAV

| Dataset | Sequence | Frames | ODTrack_AUC | Ours_AUC | ΔAUC | ODTrack_Prec@20 | Ours_Prec@20 | ODTrack_TRAC_len | Ours_TRAC_len | ODTrack_TRAC_ratio | Ours_TRAC_ratio | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CST-AntiUAV | cn_mountains_38 | 1039 | 0.166 | 0.585 | 0.419 | 0.208 | 0.772 | 9 | 9 | 0.009 | 0.009 | Ours_Win |
| CST-AntiUAV | jungle_16 | 1295 | 0.123 | 0.417 | 0.294 | 0.114 | 0.639 | 148 | 786 | 0.114 | 0.607 | Ours_Win |
| CST-AntiUAV | urban-areas_38 | 1500 | 0.463 | 0.486 | 0.023 | 0.615 | 0.671 | 924 | 994 | 0.616 | 0.663 | Both_Success |
| CST-AntiUAV | water_23 | 797 | 0.537 | 0.542 | 0.005 | 1.000 | 1.000 | 797 | 797 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | water_21 | 952 | 0.644 | 0.645 | 0.001 | 0.996 | 0.996 | 952 | 952 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | water_20 | 746 | 0.701 | 0.701 | 0.000 | 1.000 | 1.000 | 746 | 746 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | jungle_7 | 1223 | 0.470 | 0.470 | 0.000 | 0.924 | 0.924 | 349 | 349 | 0.285 | 0.285 | Both_Success |
| CST-AntiUAV | water_22 | 874 | 0.500 | 0.500 | 0.000 | 0.696 | 0.696 | 608 | 608 | 0.696 | 0.696 | Both_Success |
| CST-AntiUAV | jungle_10 | 1012 | 0.562 | 0.561 | -0.002 | 1.000 | 1.000 | 1012 | 1012 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | jungle_11 | 830 | 0.696 | 0.693 | -0.003 | 1.000 | 1.000 | 830 | 830 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | water_25 | 657 | 0.699 | 0.695 | -0.004 | 1.000 | 1.000 | 657 | 657 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | building_69 | 1000 | 0.649 | 0.645 | -0.004 | 1.000 | 1.000 | 1000 | 1000 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | jungle_12 | 1476 | 0.631 | 0.626 | -0.005 | 0.944 | 0.928 | 364 | 364 | 0.247 | 0.247 | Both_Success |
| CST-AntiUAV | jungle_14 | 1295 | 0.526 | 0.520 | -0.006 | 0.997 | 0.997 | 1295 | 1295 | 1.000 | 1.000 | Both_Success |
| CST-AntiUAV | cn_sky_15 | 1000 | 0.489 | 0.482 | -0.006 | 0.939 | 0.939 | 117 | 117 | 0.117 | 0.117 | Both_Success |

## 3. 最终结论 —— 直接回应审稿人 Comment 6

> **Reviewer Comment 6.** *The visualization in Table 6 shows qualitative examples but does not report quantitative tracking failure rates (e.g., number of sequences where ODTrack fails while DiffGuideTrack succeeds, or vice versa); please include a per-sequence success analysis or a tracking robustness metric (e.g., tracking length before first failure) to support the claimed superiority.*

下表给出**所有数据集**上的定量跟踪失败率、逐序列成功/失败对比 (Ours→Win vs. ODTrack→Win) 以及 **TRAC 鲁棒性指标（首次故障前的跟踪长度）**，直接满足审稿人要求：

| Dataset | #Seq | ODTrack Failed Seq | Ours Failed Seq | ODTrack FailRate(%) | Ours FailRate(%) | Ours→Win vs ODTrack | ODTrack→Win vs Ours | Both Success | Both Fail | ODTrack TRAC-len (frames) | Ours TRAC-len (frames) | Δ TRAC-len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anti-UAV410 | 80 | 25 | 15 | 31.25 | 18.75 | 10 | 0 | 55 | 15 | 616.2 | 630.6 | +14.4 |
| CST-AntiUAV | 60 | 45 | 44 | 75.00 | 73.33 | 2 | 1 | 14 | 43 | 374.6 | 402.4 | +27.8 |
| **All (Total)** | 140 | 70 | 59 | 50.00 | 42.14 | 12 | 1 | 69 | 58 | 512.6 | 532.8 | +20.2 |

**列含义：**
- `Ours→Win vs ODTrack`：ODTrack **失败** 而 DiffGuideTrack **成功** 的序列数量 (审稿人所要求的核心统计)
- `ODTrack→Win vs Ours`：反向情况的序列数量
- `TRAC-len (frames)`：首次连续跟丢前的成功跟踪帧数，序列级平均值 (鲁棒性指标)

**Take-away：**

- **Anti-UAV410** (80 seq): ODTrack fails on **25** seqs vs. Ours **15** seqs; Ours→Win = **10**, ODTrack→Win = **0**; TRAC-len: 616.2 → **630.6** frames (+14.4).
- **CST-AntiUAV** (60 seq): ODTrack fails on **45** seqs vs. Ours **44** seqs; Ours→Win = **2**, ODTrack→Win = **1**; TRAC-len: 374.6 → **402.4** frames (+27.8).
- **Overall (140 seq)**: Ours→Win = **12**, ODTrack→Win = **1**, net gain = **+11**; weighted-avg TRAC-len improves from **512.6** to **532.8** frames — **quantitatively confirming DiffGuideTrack's superiority claimed in Table 6.**
