import os
import subprocess
import random
import run_config
import shutil
def train():
    """
    单卡训练入口。
    内部自动构造命令并调用 tracking/train.py。
    """
    # -------------------------------
    # ✅ 可自定义参数
    # -------------------------------
    script = "odtrack"                      # 训练脚本名
    config = "baseline"                    # 配置文件名
    save_dir = f"./experiments/{run_config.dataname}"        # 日志、模型保存路径
    use_lmdb = 0                           # 是否使用 LMDB 数据
    use_wandb = 0                          # 是否启用 wandb
    distill = 0                            # 是否启用知识蒸馏
    script_prv = "none"                    # 预训练 student 脚本
    config_prv = "none"                    # 预训练配置
    script_teacher = "none"                # teacher 脚本
    config_teacher = "none"                # teacher 配置

    # -------------------------------
    # ✅ 构造单卡训练命令
    # -------------------------------
    cmd = [
        "python", "tracking/train.py",
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

    print("=" * 80)
    print("🚀 [run.py] Launching single-GPU training")
    print(cmd)
    print("=" * 80)

    subprocess.run(cmd)
    print("✅ Training finished successfully!")


def test():
    """
    单卡测试入口。
    内部自动构造命令并调用 tracking/test.py。
    """
    # -------------------------------
    # ✅ 构造测试命令
    # -------------------------------
    cmd = [
        "python", "tracking/test.py",
        "odtrack",  # tracker_name 位置参数
        "baseline",  # tracker_param 位置参数
        "--dataset_name", "lasot",
        "--debug", "0",
        "--threads", "0",
        "--num_gpus", "0"
    ]

    print("=" * 80)
    print("🚀 [run.py] Launching single-GPU testing")
    print(" ".join(cmd))  # 更好地显示命令
    print("=" * 80)

    subprocess.run(cmd)
    print("✅ Testing finished successfully!")
    



def main():
    """
    主入口，目前仅支持训练。
    后续可扩展 test(), eval(), etc.
    """
    datasets=[
    "Anti-UAV410",
    "CST-AntiUAV"]
    for dataset in datasets:
      with open("run_config.py", "w", encoding="utf-8") as f:
        f.write(f"dataname= '{dataset}'\n")
      run_config.dataname = dataset
      # 源文件路径（要复制内容的文件）
      if run_config.dataname == 'CST-AntiUAV':
         src = 'lib/train/data_specs/lasot_train_split-cts-antiuav.txt'
      elif run_config.dataname == 'Anti-UAV410':
         src = 'lib/train/data_specs/lasot_train_split-antiuav.txt'
      # 目标文件路径（要被覆盖的文件）
      dst = 'lib/train/data_specs/lasot_train_split.txt'
      # 使用 shutil.copyfile 覆盖目标文件内容
      shutil.copyfile(src, dst)
      print(f"已将 {src} 的内容复制到 {dst}")
      
      print(f"当前数据集：{run_config.dataname}")
      train()
      test()
      # 现在开始输出测试结果
      import subprocess
      with open(f"{run_config.dataname}_analysis_results.txt", "w") as log_file:
           subprocess.run(["python", "tracking/analysis_results.py"], stdout=log_file, stderr=subprocess.STDOUT)
      print(f"当前数据集：{run_config.dataname}")


if __name__ == "__main__":
    main()
