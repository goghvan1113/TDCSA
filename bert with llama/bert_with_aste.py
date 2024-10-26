import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime

from matplotlib import pyplot as plt
from pandas.core.common import random_state
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoConfig, PreTrainedModel, AutoModel, Trainer, PretrainedConfig, AutoTokenizer, \
    TrainingArguments, TrainerCallback
from transformers.modeling_outputs import ModelOutput
from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Union, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bert.custom_loss import MultiFocalLoss
from bert.plot_results import plot_confusion_matrix


class AspectEnhancedBertConfig(PretrainedConfig):
    """
    多任务模型配置类
    """
    def __init__(
            self,
            num_labels: int = 3,
            loss_type: str = "ce",
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            label_smoothing: float = 0.0,
            backbone_model: str = "roberta-base",
            hidden_size: Optional[int] = None,
            hidden_dropout_prob: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)

        # 获取backbone模型的配置
        backbone_config = AutoConfig.from_pretrained(f'../pretrain_models/{backbone_model}')

        # 设置必要的配置参数
        self.hidden_size = hidden_size or backbone_config.hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob

        # 相关的配置
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.backbone_model = backbone_model


class AspectEnhancedBertModel(PreTrainedModel):
    config_class = AspectEnhancedBertConfig
    def __init__(self, config: AspectEnhancedBertConfig):
        super().__init__(config)

        # 加载backbone模型
        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.triplet_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights() # 初始化权重 model_utils.py

        # Weight sharing between BERT encoders (optional)
        self.triplet_bert.load_state_dict(self.bert.state_dict())

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            triplet_input_ids=None,
            triplet_attention_mask=None,
            labels=None
    ):
        # Main text encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_pooled = outputs.pooler_output

        # Triplet text encoding
        triplet_outputs = self.triplet_bert(
            input_ids=triplet_input_ids,
            attention_mask=triplet_attention_mask,
        )
        triplet_pooled = triplet_outputs.pooler_output

        # Feature fusion
        combined = torch.cat([text_pooled, triplet_pooled], dim=-1)
        fused = self.fusion(combined)

        # Classification
        logits = self.classifier(fused)

        return {'logits': logits}

class AspectAwareTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs['logits']

        focal_loss = MultiFocalLoss(num_class=3, alpha=model.config.focal_alpha, gamma=model.config.focal_gamma)

        def smoothing_loss(logits, labels, smoothing):
            confidence = 1.0 - smoothing
            num_classes = logits.shape[-1]
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), confidence)
            return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))

        if model.config.loss_type == "focal_loss":
            loss = focal_loss(logits, labels)
        else:
            if model.config.label_smoothing > 0:
                loss = smoothing_loss(logits, labels, model.config.label_smoothing)
            else:
                loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'accuracy': acc
    }


@dataclass
class AspectAwareCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            print("Features:", features)

            # Extract main texts - 添加错误处理
            texts = [f.get('text', '') for f in features]  # 使用get方法安全获取
            labels = [f.get('label', 0) for f in features]  # 使用get方法安全获取

            # Encode main texts
            text_encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Process triplets
            triplet_texts = []
            for feature in features:
                if feature.get('label', 0) == 0:  # Neutral
                    triplet_text = "[NEUTRAL]"
                else:
                    triplets = feature.get('triplets', [])
                    if not triplets:
                        triplet_text = "[NEUTRAL]"
                    else:
                        # Format each triplet
                        triplet_parts = []
                        for aspect, sentiment_word, polarity in triplets:
                            pol_text = "positive" if polarity == "positive" else "negative"
                            triplet_parts.append(f"{aspect} is {sentiment_word} [{pol_text}]")
                        triplet_text = " ; ".join(triplet_parts)
                triplet_texts.append(triplet_text)

            # Encode triplet texts
            triplet_encoding = self.tokenizer(
                triplet_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            return {
                "input_ids": text_encoding["input_ids"],
                "attention_mask": text_encoding["attention_mask"],
                "triplet_input_ids": triplet_encoding["input_ids"],
                "triplet_attention_mask": triplet_encoding["attention_mask"],
                "labels": torch.tensor(labels)
            }

        except Exception as e:
            print("Error in collator:", e)
            print("Feature example:", features[0])
            raise


class AspectAwareDataset(torch.utils.data.Dataset):
    def __init__(self, pos_neg_samples, neutral_samples, tokenizer, max_length=512):
        """
        Args:
            pos_neg_samples: List of dictionaries containing positive/negative samples
                           with text, overall_sentiment, and sentiment_triplets
            neutral_samples: List of neutral text samples
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Process positive and negative samples
        for item in pos_neg_samples:
            label = 1 if item['overall_sentiment'] == 'positive' else 2
            self.samples.append({
                'text': item['text'],
                'label': label,
                'triplets': item['sentiment_triplets']
            })

        # Process neutral samples
        for text in neutral_samples:
            self.samples.append({
                'text': text,
                'label': 0,
                'triplets': []
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if 'text' not in item:
            print(f"Warning: Missing 'text' key in sample {idx}")
        return item


class DatasetSplitter:
    def __init__(self,
                 pos_neg_samples: List[Dict],
                 neutral_samples: List[str],
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        """
        Args:
            pos_neg_samples: List of dictionaries containing positive/negative samples with ASTE
            neutral_samples: List of neutral text samples
            test_size: Proportion of the dataset to include in the test split
            val_size: Proportion of the dataset to include in the validation split
            random_state: Random state for reproducibility
        """
        self.pos_neg_samples = pos_neg_samples
        self.neutral_samples = neutral_samples
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def get_sentiment_distribution(self, samples):
        """获取情感标签分布"""
        if isinstance(samples[0], dict):
            return Counter(s['overall_sentiment'] for s in samples)
        return Counter(['neutral'] * len(samples))

    def split_data(self, stratify_by_sentiment: bool = True) -> Tuple[Dict, Dict, Dict]:
        """
        划分数据集，返回训练集、验证集和测试集

        Returns:
            训练集、验证集和测试集的字典，每个字典包含pos_neg_samples和neutral_samples
        """
        # 首先划分带ASTE标注的样本
        pos_neg_labels = [s['overall_sentiment'] for s in self.pos_neg_samples]

        # 第一次划分：分离出测试集
        pos_neg_train_val, pos_neg_test, _, _ = train_test_split(
            self.pos_neg_samples,
            pos_neg_labels,
            test_size=self.test_size,
            stratify=pos_neg_labels if stratify_by_sentiment else None,
            random_state=self.random_state
        )

        # 第二次划分：从剩余数据中分离出验证集
        val_ratio = self.val_size / (1 - self.test_size)
        train_labels = [s['overall_sentiment'] for s in pos_neg_train_val]
        pos_neg_train, pos_neg_val, _, _ = train_test_split(
            pos_neg_train_val,
            train_labels,
            test_size=val_ratio,
            stratify=train_labels if stratify_by_sentiment else None,
            random_state=self.random_state
        )

        # 划分中性样本
        neutral_train, neutral_test = train_test_split(
            self.neutral_samples,
            test_size=self.test_size,
            random_state=self.random_state
        )

        neutral_train, neutral_val = train_test_split(
            neutral_train,
            test_size=val_ratio,
            random_state=self.random_state
        )

        return {
            'train': {
                'pos_neg_samples': pos_neg_train,
                'neutral_samples': neutral_train
            },
            'val': {
                'pos_neg_samples': pos_neg_val,
                'neutral_samples': neutral_val
            },
            'test': {
                'pos_neg_samples': pos_neg_test,
                'neutral_samples': neutral_test
            }
        }


def create_datasets(split_data: Dict, tokenizer, with_aste: bool = True) -> Tuple[
    AspectAwareDataset, AspectAwareDataset, AspectAwareDataset]:
    """
    创建训练集、验证集和测试集的Dataset对象

    Args:
        split_data: 划分好的数据
        tokenizer: BERT tokenizer
        with_aste: 是否在验证集和测试集中使用ASTE标注
    """
    # 训练集始终使用ASTE

    # 验证数据结构
    for split_name, split in split_data.items():
        for sample in split['pos_neg_samples']:
            if 'text' not in sample:
                print(f"Warning: Missing 'text' in {split_name} split sample: {sample}")

    train_dataset = AspectAwareDataset(
        pos_neg_samples=split_data['train']['pos_neg_samples'],
        neutral_samples=split_data['train']['neutral_samples'],
        tokenizer=tokenizer
    )

    # 验证集和测试集可以选择是否使用ASTE
    if not with_aste:
        # 转换为纯文本格式
        val_pos_neg = [{'text': s['text'],
                        'overall_sentiment': s['overall_sentiment'],
                        'sentiment_triplets': []}  # 空triplets
                       for s in split_data['val']['pos_neg_samples']]
        test_pos_neg = [{'text': s['text'],
                         'overall_sentiment': s['overall_sentiment'],
                         'sentiment_triplets': []}  # 空triplets
                        for s in split_data['test']['pos_neg_samples']]
    else:
        val_pos_neg = split_data['val']['pos_neg_samples']
        test_pos_neg = split_data['test']['pos_neg_samples']

    val_dataset = AspectAwareDataset(
        pos_neg_samples=val_pos_neg,
        neutral_samples=split_data['val']['neutral_samples'],
        tokenizer=tokenizer
    )

    test_dataset = AspectAwareDataset(
        pos_neg_samples=test_pos_neg,
        neutral_samples=split_data['test']['neutral_samples'],
        tokenizer=tokenizer
    )

    return train_dataset, val_dataset, test_dataset

def load_aste_data():
    # 加载正负样本和对应的三元组
    pos_neg_samples = []
    with open('../output/sentiment_aste_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            # 添加检查
            if 'text' not in item:
                print(f"Warning: Missing 'text' in item: {item}")
                continue

            pos_neg_samples.append({
                'text': item['text'],
                'overall_sentiment': item['overall_sentiment'],
                'sentiment_triplets': item['sentiment_triplets']
            })

    # 加载中性样本
    neutral_samples = []
    with open('../data/corpus.txt', "r", encoding="utf8") as f:
        file = f.read().split("\n")
        file = [i.split("\t") for i in file]
        for i in file:
            if len(i) == 2:
                sentence = i[1]
                label = int(i[0])
                # Map labels: 2 -> Positive, 1 -> Neutral, 0 -> Negative
                if label == 1:
                    neutral_samples.append(sentence)
    return pos_neg_samples, neutral_samples


class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataset, batch_size=32):
        """评估模型性能"""
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=AspectAwareCollator(self.tokenizer)
        )

        all_preds = []
        all_labels = []

        for batch in dataloader:
            # 将数据移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            preds = torch.argmax(outputs['logits'], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

        # 计算评估指标
        report = classification_report(
            all_labels,
            all_preds,
            target_names=['neutral', 'positive','negative'],
            digits=4
        )

        conf_matrix = plot_confusion_matrix(all_labels, all_preds, ['neutral', 'positive','negative'])

        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'labels': all_labels
        }


def train_aste_model(args, train_data, eval_data):
    device = args.device
    config = AspectEnhancedBertConfig(
        num_labels=3,
        loss_type=args.loss_type,
        focal_alpha=0.8,  # 0.8
        focal_gamma=2.0,  # 2.0
        label_smoothing=0.1,
        backbone_model=args.model_name,
    )
    model = AspectEnhancedBertModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")
    # print(model)

    training_args = TrainingArguments(
        seed=args.seed,
        report_to='none',
        output_dir=f'./results/{args.model_name}',
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type='cosine',
        gradient_accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_strategy='steps',
        logging_dir=f'./logs/{args.model_name}',
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        disable_tqdm=False,
        fp16=True,
        metric_for_best_model='f1_macro',
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    trainer = AspectAwareTrainer(
        model=model,
        args=training_args,
        data_collator=AspectAwareCollator(tokenizer),
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )
    trainer.train()

    return model

def main(args):
    # 加载数据
    pos_neg_samples, neutral_samples = load_aste_data()

    splitter = DatasetSplitter(
        pos_neg_samples=pos_neg_samples,
        neutral_samples=neutral_samples,
        test_size=0.2,
        val_size=0.1,
        random_state = args.seed
    )
    split_data = splitter.split_data(stratify_by_sentiment=True)
    # 打印数据集统计信息
    for split_name, split in split_data.items():
        pos_neg_dist = Counter(s['overall_sentiment'] for s in split['pos_neg_samples'])
        print(f"\n{split_name} set distribution:")
        print(f"Positive: {pos_neg_dist['positive']}")
        print(f"Negative: {pos_neg_dist['negative']}")
        print(f"Neutral: {len(split['neutral_samples'])}")

    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")
    train_dataset, val_dataset, test_dataset = create_datasets(
        split_data,
        tokenizer,
        with_aste=False  # 验证集和测试集不使用ASTE
    )
    model = train_aste_model(args, train_dataset, val_dataset)

    # 评估模型
    evaluator = ModelEvaluator(model, tokenizer)

    print("\nValidation Set Results:")
    val_results = evaluator.evaluate(val_dataset)
    print(val_results['classification_report'])

    print("\nTest Set Results:")
    test_results = evaluator.evaluate(test_dataset)
    print(test_results['classification_report'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="roberta-llama3.1405B-twitter-sentiment")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="ce_loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)