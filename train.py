# trainer.py
import torch
from peft import get_peft_model, LoraConfig
import numpy as np
import os
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import json
import torch

try:
    from bert_score import score as bertscore_score
except ImportError:
    bertscore_score = None


def check_ecg_encoder_grad(model):
    print("\n===== ECG Encoder Gradient Check =====")
    has_grad = False

    for name, param in model.ecg_encoder.named_parameters():
        if param.grad is None:
            print(f"[NO GRAD] {name}")
        else:
            grad_norm = param.grad.abs().mean().item()
            print(f"[OK] {name} | mean_grad={grad_norm:.6e}")
            has_grad = True

    print("======================================")
    return has_grad


# nltk.download('punkt_tab')
class ECGTextTrainer:
    def __init__(self, model, tokenizer, device='cuda:0', save_dir='checkpoints',
                 bertscore_model=None, bertscore_lang='en',
                 bertscore_local_only=True):
        # 【关键】先统一模型精度
        self.device = device
        self.model = model.to(device).to(torch.bfloat16)
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        # Use the already cached English encoder when no model is specified.
        # Override this with any local Hugging Face model directory if needed.
        self.bertscore_model = bertscore_model or os.path.join(
            os.path.expanduser('~'), '.cache', 'huggingface', 'hub',
            'models--distilbert-base-uncased', 'snapshots',
            '12040accade4e8a0f71eabdb258fecc2e7e948be'
        )
        self.bertscore_lang = bertscore_lang
        self.bertscore_local_only = bertscore_local_only
        os.makedirs(save_dir, exist_ok=True)

        # 过滤需要训练的参数 (LoRA + Encoder + Resampler)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-4,  # 微调建议 1e-4
            weight_decay=0.01
        )
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        # bf16 不需要 GradScaler
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        # 打印模型信息
        self._print_model_info()

    def _print_model_info(self):

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("=" * 60)
        print("模型信息:")
        print(f"设备: {self.device}")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"训练参数比例: {trainable_params / total_params:.2%}")
        print("=" * 60)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # 【关键】将 ECG 转换为 bf16
            ecg = batch['ecg_signal'].to(self.device).to(torch.bfloat16)
            ids = batch['input_ids'].to(self.device).long()
            mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device).long()

            self.optimizer.zero_grad()

            # 使用 autocast 保证内部计算安全
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    ecg,
                    ids,
                    labels
                )
                loss = outputs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 每50个batch打印生成示例
            if batch_idx % 100 == 0:
                self._print_generation_example(ecg[:1], batch['report'][:1])

        return total_loss / len(dataloader)

    def _print_generation_example(self, ecg_signal, text):
        """打印生成示例"""
        self.model.eval()
        with torch.no_grad():
            # 生成文本
            generated_ids = self.model.generate_report(ecg_signal)
            # generated_text = self.tokenizer.decode(
            #     generated_ids[0], skip_special_tokens=True
            # )
            print(f"原始报告：{text[:1]}")
            print(f"生成样例 (第一条): {generated_ids[:1]}")

        self.model.train()

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_refs = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                ecg = batch['ecg_signal'].to(self.device).to(torch.bfloat16)
                ids = batch['input_ids'].to(self.device).long()
                labels = batch['labels'].to(self.device).long()
                mask = batch['attention_mask'].to(self.device)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(ecg, ids, labels)
                    loss = outputs
                    total_loss += loss.item()
                # 生成测试 (使用 batch 里的真实数据)
                # 注意：prompt 必须和训练阶段一致
                generated_texts = self.model.generate_report(
                    ecg,
                    max_new_tokens=100
                )
                all_preds.extend(generated_texts)
                all_refs.extend(batch['report'])

        metrics = self.calculate_metrics(all_preds, all_refs)
        metrics['loss'] = total_loss / len(dataloader)

        return metrics

    def calculate_metrics(self, predictions, references):
        """
        计算综合评估指标：BLEU 1-4, ROUGE-1/2/L, METEOR
        """
        metrics = {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
            'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0,
            'meteor': 0.0,
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
        }

        if not predictions or not references:
            return metrics

        if bertscore_score is None:
            raise ImportError(
                "BERTScore is required for report evaluation. Install it with: pip install bert-score"
            )

        # BERTScore compares contextual token embeddings, making it suitable
        # for reports that use different wording but express the same diagnosis.
        try:
            # Avoid repeated Hugging Face network retries in offline training.
            # The model must exist in the local HF cache or in bertscore_model.
            previous_offline = os.environ.get('HF_HUB_OFFLINE')
            if self.bertscore_local_only:
                os.environ['HF_HUB_OFFLINE'] = '1'
            try:
                precision, recall, f1 = bertscore_score(
                    list(predictions), list(references),
                    model_type=self.bertscore_model,
                    lang=self.bertscore_lang,
                    device=str(self.device),
                    verbose=False,
                )
            finally:
                if previous_offline is None:
                    os.environ.pop('HF_HUB_OFFLINE', None)
                else:
                    os.environ['HF_HUB_OFFLINE'] = previous_offline
            metrics['bertscore_precision'] = float(precision.mean().item())
            metrics['bertscore_recall'] = float(recall.mean().item())
            metrics['bertscore_f1'] = float(f1.mean().item())
        except Exception as exc:
            print(f"BERTScore 计算失败（本地模型不可用）: {exc}")
            print(
                "请将 BERTScore 模型放入本地缓存，或传入本地路径，例如："
                "bertscore_model=r'C:\\models\\bert-base-multilingual-cased'"
            )

        # 1. 准备 ROUGE 计算器
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge_1_list, rouge_2_list, rouge_l_list = [], [], []
        meteor_list = []
        bleu_refs = []
        bleu_preds = []

        for pred, ref in zip(predictions, references):
            p = pred.strip()
            r = ref.strip()

            # 分词 (用于 BLEU 和 METEOR)
            p_tokens = nltk.word_tokenize(p)
            r_tokens = nltk.word_tokenize(r)

            # --- 处理 ROUGE ---
            # rouge_score 处理空字符串返回 0，不会崩溃
            scores = scorer.score(r, p)
            rouge_1_list.append(scores['rouge1'].fmeasure)
            rouge_2_list.append(scores['rouge2'].fmeasure)
            rouge_l_list.append(scores['rougeL'].fmeasure)

            # --- 处理 METEOR ---
            # meteor_score 期望参考答案是一个列表的列表（多个可能的参考答案）
            # 如果预测为空，score 为 0
            if len(p_tokens) == 0:
                m_score = 0.0
            else:
                m_score = meteor_score([r_tokens], p_tokens)
            meteor_list.append(m_score)

            # --- 准备 BLEU 数据 ---
            bleu_preds.append(p_tokens)
            bleu_refs.append([r_tokens])

        # 计算均值
        metrics['rouge_1'] = np.mean(rouge_1_list)
        metrics['rouge_2'] = np.mean(rouge_2_list)
        metrics['rouge_l'] = np.mean(rouge_l_list)
        metrics['meteor'] = np.mean(meteor_list)

        # --- 计算 BLEU 1-4 ---
        smoothing = SmoothingFunction().method1
        try:
            for i in range(1, 5):
                weights = tuple([1.0 / i] * i)
                score = corpus_bleu(bleu_refs, bleu_preds, weights=weights, smoothing_function=smoothing)
                metrics[f'bleu_{i}'] = score
        except Exception as e:
            print(f"BLEU 计算失败: {e}")

        return metrics

    def train(self, train_loader, val_loader, num_epochs=20, eval_freq=1):
        """训练模型"""
        print("开始训练...")
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 60}")

            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            print(f"训练损失: {train_loss:.4f}")

            print("进行验证...")
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            print(f"验证损失: {val_loss:.4f}")
            print(f"验证指标: {val_metrics}")
            print(
                "BERTScore | P(精度): {:.4f} | R(召回率): {:.4f} | F-1: {:.4f}".format(
                    val_metrics['bertscore_precision'],
                    val_metrics['bertscore_recall'],
                    val_metrics['bertscore_f1'],
                )
            )

            # 更新学习率
            self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")

            # 保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            # 保存训练历史
            self.save_history()

        print("\n训练完成!")
        return self.history

    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        if is_best:
            filename = f"best_model_avg.pth"
        else:
            filename = f"model_epoch_avg-{epoch + 1}.pth"

        checkpoint_path = os.path.join(self.save_dir, filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            # 'scaler_state_dict': self.scaler.state_dict(),
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'val_metrics': self.history['val_metrics'][-1] if self.history['val_metrics'] else None,
        }, checkpoint_path)

        print(f"保存检查点: {checkpoint_path}")

    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, 'training_history_avg.json')

        # 将numpy数组转换为Python列表
        history_dict = {}
        for key, value in self.history.items():
            if key == 'val_metrics':
                # 处理指标字典
                history_dict[key] = []
                for metrics in value:
                    metrics_dict = {}
                    for k, v in metrics.items():
                        if isinstance(v, (np.float32, np.float64)):
                            metrics_dict[k] = float(v)
                        else:
                            metrics_dict[k] = v
                    history_dict[key].append(metrics_dict)
            else:
                # 处理损失列表
                history_dict[key] = [float(v) for v in value]

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"加载检查点: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"训练损失: {checkpoint.get('train_loss', 'N/A')}")
        print(f"验证损失: {checkpoint.get('val_loss', 'N/A')}")


if __name__ == "__main__":
    # 测试训练器

    print("测试训练器...")

    # 初始化
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # # 创建模型
    # model = MultimodalECGModel(vocab_size=tokenizer.vocab_size)
    #
    # # 创建训练器
    # trainer = ECGTextTrainer(model, tokenizer, save_dir='test_checkpoints')
    #
    # print("训练器测试完成!")
