import os
import subprocess
import sys
import os

def setup_environment():
    """设置 nnUNetv2 必须的环境变量"""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "datasets")
    
    env_vars = {
        "nnUNet_wandb_enabled": 1,
        "nnUNet_wandb_project" : "nnunet",
        "nnUNet_raw": os.path.join(output_dir, "nnUNet_raw"),
        "nnUNet_preprocessed": os.path.join(output_dir, "nnUNet_preprocessed"),
        "nnUNet_results": os.path.join(output_dir, "nnUNet_results")
    }
    
    for k, v in env_vars.items():
        os.makedirs(v, exist_ok=True)
        os.environ[k] = v
        print(f"Set {k} = {v}")

def run_command(command):
    """辅助函数，用来安全执行命令行并实时打印输出"""
    print(f"\n>> 执行命令: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy() # 传入已经配置好的环境变量
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
        process.wait()
        if process.returncode != 0:
            print(f"命令执行失败，返回码: {process.returncode}")
            sys.exit(process.returncode)
    except Exception as e:
        print(f"执行时发生错误: {e}")
        sys.exit(1)

def main():
    setup_environment()
    
    task_id = "101"
    
    # 步骤 1：分析数据集并生成最佳前处理计划
    print("\n==============================================")
    print("阶段 1: 数据指纹提取与预处理 (Plan & Preprocess)")
    print("==============================================")
    # `-c 2d 3d_fullres` 指定要预处理的配置，这里先只预处理 2d 节省时间
    run_command(["nnUNetv2_plan_and_preprocess", "-d", task_id, "-c", "2d", "--verify_dataset_integrity"])
    
    # 步骤 2：启动模型训练
    print("\n==============================================")
    print("阶段 2: 启动网络训练 (Training)")
    print("==============================================")
    # `2d` 代表配置，`0` 代表5折交叉验证的第一折 (fold 0)
    # 若需修改 Epoch 数量，可使用官方的快速训练 Trainer 例如：-tr nnUNetTrainer_5epochs (需要官方代码支持)
    # 本脚本仅用于展示完整的管线启动
    run_command(["nnUNetv2_train", task_id, "2d", "0"])

if __name__ == "__main__":
    # 为了避免没有数据直接运行报错，在这里做个简单检查
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_dir = os.path.join(root_dir, "datasets", "nnUNet_raw", "Dataset101_Meningioma")
    if not os.path.exists(task_dir) or not os.listdir(task_dir):
        print(f"未在 {task_dir} 检测到数据！")
        print("请首先运行: python src/prepare_data.py")
        sys.exit(1)
        
    main()