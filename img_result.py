import os
import cv2
import numpy as np

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0

def load_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    boxes.append([x, y, w, h])
                except:
                    boxes.append([0, 0, 0, 0])
            else:
                boxes.append([0, 0, 0, 0])
    return boxes

def load_groundtruth(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                print(f"Error parsing line {line}")
                continue
            parts = line.split(',')
            if len(parts) < 4:
                parts = line.split()
                if len(parts) < 4:
                    boxes.append(None)
                    print(f"Error parsing line {line}")
                    continue
            try:
                x, y, w, h = map(float, parts[:4])
                boxes.append([x, y, w, h])
            except Exception as e:
                print(f"Error parsing line {e}")

                boxes.append(None)
    return boxes

def draw_box(image, box, color=(0, 0, 255), thickness=2):
    # 1. 先四舍五入到整数（关键修复！）
    x, y, w, h = [int(round(coord)) for coord in box]
    
    # 2. 检查w/h是否为正数（避免无效框）
    if w <= 0 or h <= 0:
        return image
    
    # 3. 确保坐标在图像范围内
    x1 = max(0, min(x, image.shape[1]-1))
    y1 = max(0, min(y, image.shape[0]-1))
    x2 = min(image.shape[1], x1 + w)
    y2 = min(image.shape[0], y1 + h)
    
    # 4. 绘制矩形
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image
def main():
    # 路径配置
    dataset_name = "CST-AntiUAV"
    model1_dir = f"/home/workspace/projects/ODtrack/outSave/test/{dataset_name}/tracking_results/odtrack/baseline/lasot"
    model2_dir = f"/home/workspace/projects/ODtrack-1/outSave/odtrack_IDF_TGG/test/{dataset_name}/tracking_results/odtrack/baseline/lasot"
    dataset_dir = f"/home/workspace/projects/datasets/{dataset_name}/lasot/test"
    output_root = f"zimg_tmp/{dataset_name}"
    os.makedirs(output_root, exist_ok=True)

    # 获取所有视频名
    video_names = [f.replace('.txt', '') for f in os.listdir(model1_dir)
                   if f.endswith('.txt') and not f.endswith('_time.txt')]

    candidate_videos = []

    for video_name in video_names:
        m1_path = os.path.join(model1_dir, f"{video_name}.txt")
        m2_path = os.path.join(model2_dir, f"{video_name}.txt")
        gt_path = os.path.join(dataset_dir, video_name, "groundtruth.txt")

        if not (os.path.exists(m1_path) and os.path.exists(m2_path) and os.path.exists(gt_path)):
            continue

        m1_boxes = load_boxes(m1_path)
        m2_boxes = load_boxes(m2_path)
        gt_boxes = load_groundtruth(gt_path)

        min_len = min(len(m1_boxes), len(m2_boxes), len(gt_boxes))
        m1_boxes = m1_boxes[:min_len]
        m2_boxes = m2_boxes[:min_len]
        gt_boxes = gt_boxes[:min_len]

        # 逐帧收集对比明显帧
        good_frames = []  # 保存满足条件的帧索引
        diff_vals = []    # 保存每帧的差值 (iou2 - iou1)

        for i in range(min_len):
            if gt_boxes[i] is None:
                continue
            iou1 = compute_iou(m1_boxes[i], gt_boxes[i])
            iou2 = compute_iou(m2_boxes[i], gt_boxes[i])
            
            # 条件：模型1表现差，模型2表现好，且差值足够大
            if iou1 < iou2:
                good_frames.append(i)
                diff_vals.append(iou2 - iou1)

        # 如果有足够多的对比明显帧（至少4帧），则加入候选列表
        if len(good_frames) >= 4:
            avg_diff = np.mean(diff_vals)
            candidate_videos.append({
                "video_name": video_name,
                "good_frames": good_frames,  # 保存所有满足条件的帧索引
                "avg_diff": avg_diff
            })

    # 按平均差值从大到小排序（对比最明显的视频排前面）
    candidate_videos.sort(key=lambda x: x["avg_diff"], reverse=True)
    selected_videos = candidate_videos[:5]  # 取前5个最明显的视频

    # 处理每个选中视频
    for video in selected_videos:
        video_name = video["video_name"]
        print(f"Processing {video_name} (with {len(video['good_frames'])} good frames)...")

        m1_path = os.path.join(model1_dir, f"{video_name}.txt")
        m2_path = os.path.join(model2_dir, f"{video_name}.txt")
        gt_path = os.path.join(dataset_dir, video_name, "groundtruth.txt")

        m1_boxes = load_boxes(m1_path)
        m2_boxes = load_boxes(m2_path)
        gt_boxes = load_groundtruth(gt_path)

        min_len = min(len(m1_boxes), len(m2_boxes), len(gt_boxes))
        m1_boxes = m1_boxes[:min_len]
        m2_boxes = m2_boxes[:min_len]
        gt_boxes = gt_boxes[:min_len]

        # 从good_frames中均匀选取4帧
        good_frames = video["good_frames"]
        n = len(good_frames)
        selected_frames = []
        for i in range(4):
            idx = i * 12
            selected_frames.append(good_frames[idx])
        
        # 创建输出子目录
        out_dir = os.path.join(output_root, video_name)
        os.makedirs(out_dir, exist_ok=True)

        img_dir = os.path.join(dataset_dir, video_name)
        print(f"Processing {img_dir}...")
        for frame_idx in selected_frames[:4]:
            frame_num = frame_idx + 1
            img_path = os.path.join(img_dir, f"{frame_num:06d}.jpg")
            if not os.path.exists(img_path):
                continue

            orig_img = cv2.imread(img_path)
            if orig_img is None:
                continue

            # === 绘制综合图：GT（蓝）、Model1（红）、Model2（绿） ===
            combined_img = orig_img.copy()

            # 画 Ground Truth（如果存在）
            if gt_boxes[frame_idx] is not None:
                combined_img = draw_box(combined_img, gt_boxes[frame_idx], color=(255, 0, 0), thickness=2)  # BGR: blue
                print(f"GT: {gt_boxes[frame_idx]}")
                # 画模型一（红色）
                combined_img = draw_box(combined_img, m1_boxes[frame_idx], color=(0, 0, 255), thickness=2)
                # 画模型二（绿色）
                combined_img = draw_box(combined_img, m2_boxes[frame_idx], color=(0, 255, 0), thickness=2)

                # 保存综合标注图
                all_save = os.path.join(out_dir, f"{dataset_name}_{video_name}_{frame_num:06d}_all_anno.jpg")
                cv2.imwrite(all_save, combined_img)
    
                # # 保存单独的模型标注图（可选）
                # img_m1 = draw_box(orig_img.copy(), m1_boxes[frame_idx], color=(0, 0, 255))
                # m1_save = os.path.join(out_dir, f"{dataset_name}_{video_name}_{frame_num:06d}_model1_anno.jpg")
                # cv2.imwrite(m1_save, img_m1)
    
                # img_m2 = draw_box(orig_img.copy(), m2_boxes[frame_idx], color=(0, 255, 0))
                # m2_save = os.path.join(out_dir, f"{dataset_name}_{video_name}_{frame_num:06d}_model2_anno.jpg")
                # cv2.imwrite(m2_save, img_m2)
                

            

    print(f"\n✅ Done! Results saved in '{output_root}/'")
    print(f"Selected {len(selected_videos)} videos with clear comparison frames")

if __name__ == "__main__":
    main()