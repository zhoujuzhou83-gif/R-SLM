# main.py
import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
from data_loader import create_data_loaders
from model import ECGReportSystem
from train import ECGTextTrainer


def setup_environment():
    """设置运行环境"""
    # # 设置环境变量以获得准确的CUDA错误位置
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #
    # # 设置随机种子
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)

    # 选择设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    return device


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态ECG诊断文本生成模型')

    # 模型参数
    parser.add_argument('--ecg_dim', type=int, default=512,
                        help='ECG编码维度')
    parser.add_argument('--text_dim', type=int, default=768,
                        help='文本编码维度')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Transformer解码器层数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--max_length', type=int, default=128,
                        help='文本最大长度')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='验证频率（每多少个epoch验证一次）')

    # 其他参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 设置环境
    device = setup_environment()

    # 创建数据加载器
    print("\n" + "=" * 60)
    print("创建数据加载器...")
    print("=" * 60)

    # train_local_loader, val_local_loader, tokenizer = create_data_loaders(
    #     train_txt=r'D:\pycharm\ecg_text\data\train_report.txt',
    #     val_txt=r'D:\pycharm\ecg_text\data\val_report.txt',
    #     data_dir=r'D:\pycharm\LSTM-ECG\important_ecg'
    # )
    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)

    model = ECGReportSystem().to(device)
    train_loader, val_loader, tokenizer = create_data_loaders(
        train_txt='/home/zh/ZhouJu2026/qwen_1.7b/data/MIXED_EN_train_report.txt',
        val_txt='/home/zh/ZhouJu2026/qwen_1.7b/data/MIXED_EN_val_report.txt',
        data_dir='/home/zh/ZhouJu2026/qwen_1.7b/data',
        tokenizer=model.tokenizer
    )

    train_local_loader, val_local_loader, tokenizer = create_data_loaders(
        train_txt='/home/zh/ZhouJu2026/qwen_1.7b/data/MIXED_EN_train_report.txt',
        val_txt='/home/zh/ZhouJu2026/qwen_1.7b/data/MIXED_EN_val_report.txt',
        data_dir='/home/zh/ZhouJu2026/qwen_1.7b/important_ecg/',
        tokenizer=model.tokenizer
    )

    # 创建训练器
    trainer = ECGTextTrainer(
        model=model,
        tokenizer=model.tokenizer,
        device=device,
        save_dir=args.save_dir
    )

    # 恢复训练（如果指定）
    if args.resume and os.path.exists(args.resume):
        print(f"\n恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 训练模型
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq
    )

    # 保存最终模型
    trainer.save_checkpoint(args.num_epochs - 1)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 打印最终指标
    if history['val_metrics']:
        final_metrics = history['val_metrics'][-1]
        print("\n最终验证指标:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")

    # print(f"\n模型保存在: {args.save_dir}")
    # print("可以使用以下命令进行推理:")
    # print(
    #     f"  python inference.py --model_path {args.save_dir}/best_model.pth --ecg_dir {args.ecg_dir} --text_dir {args.text_dir}")


if __name__ == "__main__":
    main()

