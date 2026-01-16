import os
import subprocess
import random
import run_config
import shutil

import os
import subprocess
import run_config
import shutil


def run_with_logging(cmd, log_file_handle):
    """运行命令，并将输出同时写入 log_file_handle 和控制台"""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # ✅ 修复：删除了 with log_file_handle: 这行
    for line in proc.stdout:
        print(line, end='')
        log_file_handle.write(line)
        log_file_handle.flush()

    return proc.wait()


def train(log_file_handle):
    """
    单卡训练入口。
    内部自动构造命令并调用 tracking/train.py。
    """
    # -------------------------------
    # ✅ 可自定义参数
    # -------------------------------
    script = "odtrack"  # 训练脚本名
    config = "baseline"  # 配置文件名
    save_dir = f"./experiments/{run_config.dataname}"  # 日志、模型保存路径
    use_lmdb = 0  # 是否使用 LMDB 数据
    use_wandb = 0  # 是否启用 wandb
    distill = 0  # 是否启用知识蒸馏
    script_prv = "none"  # 预训练 student 脚本
    config_prv = "none"  # 预训练配置
    script_teacher = "none"  # teacher 脚本
    config_teacher = "none"  # teacher 配置

    # -------------------------------
    # ✅ 构造单卡训练命令
    # -------------------------------
    cmd = [
        "python", "-u", "tracking/train.py",
        "--mode", "single",
        "--script", script,
        "--config", config,
        "--save_dir", save_dir,
        "--use_lmdb", str(use_lmdb),
        "--script_prv", script_prv,
        "--config_prv", config_prv,
        "--distill", str(distill),
        "--script_teacher", script_teacher,
        "--config_teacher", config_teacher,
        "--use_wandb", str(use_wandb)
    ]

    separator = "=" * 80
    print_log = lambda msg: print_and_log(msg, log_file_handle)
    print_log(separator)
    print_log("🚀 [run.py] Launching single-GPU training")
    print_log(str(cmd))
    print_log(separator)

    exit_code = run_with_logging(cmd, log_file_handle)
    if exit_code != 0:
        raise RuntimeError(f"Training failed with exit code {exit_code}")
    print_log("✅ Training finished successfully!")


def test(log_file_handle):
    """
    单卡测试入口。
    内部自动构造命令并调用 tracking/test.py。
    """
    # -------------------------------
    # ✅ 构造测试命令
    # -------------------------------
    cmd = [
        "python", "-u", "tracking/test.py",
        "odtrack",  # tracker_name 位置参数
        "baseline",  # tracker_param 位置参数
        "--dataset_name", "lasot",
        "--debug", "0",
        "--threads", "0",
        "--num_gpus", "0"
    ]

    separator = "=" * 80
    print_log = lambda msg: print_and_log(msg, log_file_handle)
    print_log(separator)
    print_log("🚀 [run.py] Launching single-GPU testing")
    print_log(" ".join(cmd))
    print_log(separator)

    exit_code = run_with_logging(cmd, log_file_handle)
    if exit_code != 0:
        raise RuntimeError(f"Testing failed with exit code {exit_code}")
    print_log("✅ Testing finished successfully!")


def print_and_log(message, log_file_handle):
    """同时打印到控制台和日志文件"""
    print(message)
    log_file_handle.write(str(message) + "\n")
    log_file_handle.flush()


def main():
    datasets = ["Anti-UAV410", "CST-AntiUAV"]
    for dataset in datasets:
        log_dir = f"./experiments/{dataset}"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "run_full_log.txt")

        # 更新 run_config.py
        with open("run_config.py", "w", encoding="utf-8") as f:
            f.write(f"dataname = '{dataset}'\n")
        run_config.dataname = dataset

        # 复制数据划分文件
        if dataset == 'CST-AntiUAV':
            src = 'lib/train/data_specs/lasot_train_split-cts-antiuav.txt'
        else:  # Anti-UAV410
            src = 'lib/train/data_specs/lasot_train_split-antiuav.txt'
        dst = 'lib/train/data_specs/lasot_train_split.txt'
        shutil.copyfile(src, dst)

        # 打开日志文件（每个 dataset 独立）
        with open(log_file_path, "w", encoding="utf-8") as log_f:
            print_and_log(f"📝 Logging all output to: {log_file_path}", log_f)
            print_and_log(f"已将 {src} 的内容复制到 {dst}", log_f)
            print_and_log(f"当前数据集：{dataset}", log_f)

            # 训练
            train(log_f)

            # 测试
            test(log_f)

            # 分析结果
            print_and_log(f"📊 Running analysis for {dataset}...", log_f)
            analysis_cmd = ["python", "-u", "tracking/analysis_results.py"]
            exit_code = run_with_logging(analysis_cmd, log_f)
            if exit_code != 0:
                print_and_log(f"⚠️ Analysis failed with exit code {exit_code}", log_f)
            else:
                print_and_log(f"✅ Finished analysis for {dataset}", log_f)


if __name__ == "__main__":
    main()