# data_loader.py
# coding=utf-8
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

SYSTEM_PROMPT = "You are a professional cardiologist. Analyze the 12-lead ECG and provide a clinical report."
USER_PROMPT = "Here is the ECG signal data. Please generate the diagnosis."


class PTBXLDataReader:
    """PTB-XL数据读取器"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_ecg_record(self, record_path):
        """读取ECG记录文件"""
        full_path = os.path.join(self.data_dir, record_path) + '.npy'
        signals = np.load(full_path)
        return signals


class MultimodalECGDataset(Dataset):
    """多模态ECG数据集"""

    def __init__(self, txt_file, data_dir, tokenizer, signal_length=1000, max_text_len=256,
                 tokenizer_name='HuggingFaceTB/SmolLM2-1.7B'):
        """
        Args:
            txt_file: 包含ECG路径和文本信息的txt文件
            data_dir: PTB-XL数据根目录
            signal_length: ECG信号长度
            max_text_len: 文本最大长度
            tokenizer_name: 用于文本tokenization的模型名称
        """
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.max_text_len = max_text_len
        self.reader = PTBXLDataReader(data_dir)
        # 加载数据记录
        self.records = self._load_txt_file(txt_file)

        # # 初始化tokenizer
        self.tokenizer = tokenizer
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # if self.tokenizer.pad_token_id is None:
        #     print(f"Warning: pad_token_id is None, setting to eos_token_id: {self.tokenizer.eos_token_id}")
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_txt_file(self, txt_file):
        """从txt文件加载数据记录"""
        records = []
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # 分割第一列（路径）和第二列（字典信息）
                    parts = line.split(' ', 1)
                    record_path = parts[0].strip()
                    text_data = parts[1].strip()
                    records.append((record_path, text_data))

        except FileNotFoundError:
            print(f"Error: File {txt_file} not found")
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")

        print(f"Loaded {len(records)} records from {txt_file}")
        return records

    def _preprocess_ecg(self, signal):
        """预处理ECG信号"""
        # 标准化每个导联 (z-score normalization)
        for i in range(12):
            mean_val = np.mean(signal[i])
            std_val = np.std(signal[i])
            if std_val > 1e-8:  # 避免除以接近零的标准差
                signal[i] = (signal[i] - mean_val) / std_val
            else:
                signal[i] = signal[i] - mean_val  # 只去均值

        return signal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path, info_dict = self.records[idx]
        ecg_signal = self.reader.read_ecg_record(record_path)
        ecg_signal = self._preprocess_ecg(ecg_signal)

        # -- 构造符合 Qwen2 的 ChatML 模板 ---
        # 格式: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{report}<|im_end|>

        prompt_full = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{USER_PROMPT}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        answer = f"{info_dict}<|im_end|>"
        full_text = prompt_full + answer

        # Tokenize 全文
        encoding = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # --- 精确构造 Labels ---
        labels = input_ids.clone()

        # 只有回答部分的文字需要计算梯度，Prompt部分全部设为 -100
        # 我们先编码 prompt_full 来计算它的长度
        prompt_tokenized = self.tokenizer(prompt_full, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokenized)

        labels[:prompt_len] = -100  # 屏蔽 prompt 部分
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # 屏蔽 padding 部分

        return {
            'ecg_signal': torch.FloatTensor(ecg_signal),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'report': info_dict
        }


def create_data_loaders(train_txt, val_txt, data_dir, tokenizer,
                        batch_size=2, signal_length=1000, max_text_len=256,
                        tokenizer_name='Qwen/Qwen3-1.7B', num_workers=1):
    """
    创建训练、验证和测试数据加载器

    Args:
        train_txt: 训练数据txt文件路径
        val_txt: 验证数据txt文件路径
        data_dir: PTB-XL数据根目录
        batch_size: 批次大小
        signal_length: ECG信号长度
        max_text_len: 文本最大长度
        tokenizer_name: tokenizer名称
        num_workers: 数据加载器工作进程数

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """

    # 创建数据集
    print("Creating training dataset...")
    train_dataset = MultimodalECGDataset(
        txt_file=train_txt,
        data_dir=data_dir,
        signal_length=signal_length,
        max_text_len=max_text_len,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer
    )

    print("Creating validation dataset...")
    val_dataset = MultimodalECGDataset(
        txt_file=val_txt,
        data_dir=data_dir,
        signal_length=signal_length,
        max_text_len=max_text_len,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset.tokenizer


# 测试函数

if __name__ == '__main__':
    pass
