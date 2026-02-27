import os
import subprocess
from evaluation import run_evaluation

# 配置区域
SEQUENCES = [f"{i:04d}" for i in range(41)] # 0000 到 0040
DATA_ROOT = "./data"
MARGIN = 3.5
EPOCHS = 20

def run_command(cmd):
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    for seq in SEQUENCES:
        print(f"\n{'='*20} 开始处理序列 {seq} {'='*20}")
        
        # 1. 训练脚本
        train_cmd = [
            "python3", "main.py",
            "--seq_id", seq,
            "--data_root", DATA_ROOT,
            "--margin", str(MARGIN),
            "--epochs", str(EPOCHS)
        ]
        
        # 2. 评估脚本
        lora_path = f"checkpoints/lora_{seq}_final"
        eval_cmd = [
            "python3", "evaluate.py", # 这里建议把 evaluate.py 里的 main 逻辑稍作修改以接收参数
        ]
        
        try:
            # 运行训练
            run_command(train_cmd)
            
            # 运行评估 (可以直接调用 evaluate.py 里的函数，或者通过命令行)
            # 这里演示直接调用 evaluate.py 的方式，你可以在 evaluate.py 末尾增加 argparse
            print(f">>> 正在生成序列 {seq} 的对比图...")

            run_evaluation(seq, DATA_ROOT, lora_path, r=4)
            
        except Exception as e:
            print(f"❌ 序列 {seq} 处理失败: {e}")
            continue

    print("\n✅ 所有序列处理完毕！请在 results/ 文件夹查看对比图。")

if __name__ == "__main__":
    main()