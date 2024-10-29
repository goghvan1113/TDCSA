import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoConfig, PreTrainedModel, AutoModel, Trainer, PretrainedConfig, AutoTokenizer, \
    TrainingArguments, TrainerCallback, DataCollatorWithPadding

from torchviz import make_dot
from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Union, List, Any, Tuple, Mapping
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
            multitask: bool = True,
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
        self.multitask = multitask
        self.label_smoothing = label_smoothing
        self.backbone_model = backbone_model


class AspectEnhancedBertModel(PreTrainedModel):
    config_class = AspectEnhancedBertConfig
    def __init__(self, config: AspectEnhancedBertConfig):
        super().__init__(config)

        # 加载backbone模型
        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.triplet_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        hidden_size = config.hidden_size

        ## 从词级别到句子级别的处理
        self.word_level = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 短语级别的处理 1D卷积网络捕获局部特征
        self.phrase_level = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 句子级别的处理，捕捉句子内的长距离依赖GRU or LSTM
        self.sentence_level = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
            )

        # Aspect-aware attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        # 自注意力层 - 增强特征提取
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        # 特征转换层
        self.text_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.aspect_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 3), # 都是hidden_size * 2 便于残差连接
            nn.LayerNorm(hidden_size * 3),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, config.num_labels)
        )
        self.sentiment_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, 2) # 检测是否有情感
        )

        self.init_weights() # 初始化权重 model_utils.py

        # Weight sharing between BERT encoders (optional) 不共享得分要高一点
        self.triplet_bert.load_state_dict(self.bert.state_dict())

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            triplet_input_ids=None,
            triplet_attention_mask=None,
            labels=None,
            sentiment_labels=None
    ):
        # Main text encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_hidden = outputs.last_hidden_state # [batch_size, seq_len, hidden_size]
        text_pooled = outputs.pooler_output  # [batch_size, hidden_size]

        # Triplet text encoding
        aspect_outputs = self.triplet_bert(
            input_ids=triplet_input_ids,
            attention_mask=triplet_attention_mask,
        )
        aspect_hidden = aspect_outputs.last_hidden_state
        aspect_pooled = aspect_outputs.pooler_output

        # ## 从词级别到句子级别的处理
        # word_features = self.word_level(text_hidden)  # [batch, seq_len, 512]
        # phrase_features = self.phrase_level(word_features.transpose(1, 2)).transpose(1, 2)  # [batch_size, seq_len, 256]
        # sentence_features, _ = self.sentence_level(phrase_features)  # [batch_size, seq_len, 256]

        text_transformed = self.text_transform(text_hidden)  # [batch_size, seq_len, hidden_size]
        aspect_transformed = self.aspect_transform(aspect_hidden)

        # 交互注意力
        cross_attn_output, _ = self.cross_attention(
            query=text_transformed,
            key=aspect_transformed,
            value=aspect_transformed,
            key_padding_mask=~triplet_attention_mask.bool()
        ) # [batch_size, seq_len, hidden_size]
        # 自注意力增强
        self_attn_output, _ = self.self_attention(
            query=cross_attn_output,
            key=cross_attn_output,
            value=cross_attn_output,
            key_padding_mask=~attention_mask.bool()
        ) # [batch_size, seq_len, hidden_size]
        attn_pooled = torch.mean(self_attn_output, dim=1) # [batch_size, hidden_size]

        # Feature fusion
        combined = torch.cat([text_pooled, attn_pooled], dim=-1) # [batch_size, hidden_size * 3]
        fused = self.fusion(combined) # [batch_size, hidden_size * 3]

        # Classification
        sentiment_logits = self.sentiment_detector(text_pooled) # [batch_size, 2]
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        logits = self.classifier(fused) # [batch_size, num_labels]
        logits_probs = torch.softmax(logits, dim=-1)

        # 修改后的逻辑：使用概率加权
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)  # 客观的概率
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)  # 主观的概率
        # 混合预测结果
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)

        # 转换回logits形式
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits,
            'sentiment_logits': sentiment_logits
        }


class AspectAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        sentiment_labels = inputs.pop("sentiment_labels")
        outputs = model(**inputs)
        logits = outputs['logits']
        sentiment_logits = outputs['sentiment_logits'] # 主客观分类

        focal_loss_3labels = MultiFocalLoss(num_class=model.config.num_labels, alpha=model.config.focal_alpha, gamma=model.config.focal_gamma)
        focal_loss_2labels = MultiFocalLoss(num_class=2, alpha=model.config.focal_alpha, gamma=model.config.focal_gamma)

        def smoothing_loss(logits, labels, smoothing):
            confidence = 1.0 - smoothing
            num_classes = logits.shape[-1]
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), confidence)
            return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))

        if model.config.loss_type == "focal_loss":
            loss = focal_loss_3labels(logits, labels)
            loss_2class = focal_loss_2labels(sentiment_logits, sentiment_labels)
        else:
            if model.config.label_smoothing > 0:
                loss = smoothing_loss(logits, labels, model.config.label_smoothing)
                loss_2class = smoothing_loss(sentiment_logits, sentiment_labels, model.config.label_smoothing)
            else:
                loss = F.cross_entropy(logits, labels)
                loss_2class = F.cross_entropy(sentiment_logits, sentiment_labels)

        if model.config.multitask:
            loss = loss + loss_2class * 0.5
        else:
            loss = loss
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids[0]
    labels_2class = pred.label_ids[1] # Tuple
    preds = pred.predictions[0].argmax(-1)
    preds_2class = pred.predictions[1].argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    _, _, f1_2class, _ = precision_recall_fscore_support(labels_2class, preds_2class, average='macro')
    acc = accuracy_score(labels, preds)
    acc_2class = accuracy_score(labels_2class, preds_2class)
    return {
        'Precision_Macro': precision,
        'Recall_Macro': recall,
        'F1_Macro': f1,
        'Accuracy': acc,
        'F1_Macro_2class': f1_2class,
        'Accuracy_2class': acc_2class
    }

# default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
class AspectDataProcessor:
    """
    数据预处理器，在准备数据集时处理ASTE数据
    """
    def __init__(self, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        分别整个处理主文本和triplets，返回处理后的特征
        """
        # 处理主文本
        texts = [f.get('text', '') for f in features]  # 使用get方法安全获取
        labels = [f.get('label', 0) for f in features]  # 使用get方法安全获取
        sentiment_labels = [1 if label > 0 else 0 for label in labels]  # 有情感(积极/消极)为1, 无情感(中性)为0

        text_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Process triplets 在这里面加入"类别“的文本影响分类效果
        triplet_texts = []
        for feature in features:
            if feature.get('label', 0) == 0:  # Neutral
                triplet_text = "[NEUTRAL]"
                # 使用[MASK]符号和随机噪声填充中性样本的三元组
                num_masks = random.randint(1, 3)  # 随机选择[MASK]数量
                triplet_text = " ".join(["[MASK]"] * num_masks)
                # 添加少量随机噪声
                noise_tokens = [
                    "neutral", "sample", "info", "example", "text", "generic", "data", "reference", "placeholder",
                    "context", "statement", "source", "comment", "topic", "subject", "entry", "note", "point",
                    "sentence", "item", "mention", "instance", "case", "illustration", "abstract", "token",
                    "term", "phrase", "description", "note", "segment", "section", "passage", "piece", "part",
                    "word", "snippet", "fragment", "content", "neutral_text", "general", "entity", "element"
                ]
                random.shuffle(noise_tokens)
                triplet_text += " " + " ".join(noise_tokens[:random.randint(1, len(noise_tokens))]) # 长度不一定为3
            else:
                triplets = feature.get('triplets', [])
                triplet_parts = []
                for aspect, sentiment_word, polarity in triplets:
                    pol_text = "positive" if polarity == "positive" else "negative"
                    # triplet_parts.append(f"{aspect} is {sentiment_word} ")
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

        # 返回处理后的特征
        return {
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "triplet_input_ids": triplet_encoding["input_ids"],
            "triplet_attention_mask": triplet_encoding["attention_mask"],
            "labels": torch.tensor(labels),
            "sentiment_labels": torch.tensor(sentiment_labels)
        }


class AspectAwareDataset(torch.utils.data.Dataset):
    def __init__(self, pos_neg_samples, neutral_samples, tokenizer, processor, max_length=512):
        """
        Args:
            pos_neg_samples: List of dictionaries containing positive/negative samples
                           with text, overall_sentiment, and sentiment_triplets
            neutral_samples: List of neutral text samples
            tokenizer: BERT tokenizer
            processor: AspectDataProcessor instance
            max_length: Maximum sequence length
        """
        self.samples = []
        self.samples_tokenized = {}  # Changed to dictionary for better error tracking
        self.tokenizer = tokenizer
        self.processor = processor
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
        # Shuffle the samples
        random.shuffle(self.samples)
        processed_features = self.processor.process_features(self.samples)

        # Convert tensor data to dictionary with indices
        for key in processed_features:
            if isinstance(processed_features[key], torch.Tensor):
                for idx in range(len(processed_features[key])):
                    if idx not in self.samples_tokenized:
                        self.samples_tokenized[idx] = {}
                    self.samples_tokenized[idx][key] = processed_features[key][idx]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            if idx not in self.samples_tokenized:
                raise KeyError(f"Index {idx} not found in processed samples")

            item = self.samples_tokenized[idx]
            return item

        except Exception as e:
            print(f"Error accessing index {idx}")
            print(f"Available indices: {sorted(self.samples_tokenized.keys())}")
            raise


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
    processor = AspectDataProcessor(tokenizer)

    for split_name, split in split_data.items():
        for sample in split['pos_neg_samples']:
            if 'text' not in sample:
                print(f"Warning: Missing 'text' in {split_name} split sample: {sample}")

    train_dataset = AspectAwareDataset(
        pos_neg_samples=split_data['train']['pos_neg_samples'],
        neutral_samples=split_data['train']['neutral_samples'],
        tokenizer=tokenizer,
        processor=processor
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
        tokenizer=tokenizer,
        processor=processor
    )

    test_dataset = AspectAwareDataset(
        pos_neg_samples=test_pos_neg,
        neutral_samples=split_data['test']['neutral_samples'],
        tokenizer=tokenizer,
        processor=processor
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

            # Check for sentiment_triplets with more than 3 elements
            for triplet in item['sentiment_triplets']:
                if len(triplet) > 3:
                    print(f"Item with triplet having more than 3 elements: {item}")

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
            collate_fn=DataCollatorWithPadding(self.tokenizer) # 改为默认
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
            'labels': all_labels,
        }

class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.train_metrics = {
            'steps': [],
            'train_loss': []
        }
        self.eval_metrics = {
            'steps': [],
            'eval_loss': [],
            'f1_scores_macro': [],
            'accuracy': [],
            'f1_scores_macro_2class': [],
            'accuracy_2class': []
        }
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_metrics['steps'].append(state.global_step)
            self.train_metrics['train_loss'].append(logs['loss'])

        if 'eval_loss' in logs:
            self.eval_metrics['steps'].append(state.global_step)
            self.eval_metrics['eval_loss'].append(logs['eval_loss'])
            self.eval_metrics['f1_scores_macro'].append(logs['eval_F1_Macro'])
            self.eval_metrics['accuracy'].append(logs['eval_Accuracy'])
            self.eval_metrics['f1_scores_macro_2class'].append(logs['eval_F1_Macro_2class'])
            self.eval_metrics['accuracy_2class'].append(logs['eval_Accuracy_2class'])

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self._plot_metrics()

    def _plot_metrics(self):
        self.ax1.clear()
        self.ax2.clear()

        # Plot losses
        if self.train_metrics['train_loss']:
            self.ax1.plot(self.train_metrics['steps'], self.train_metrics['train_loss'],
                          label='Train Loss', color='blue')

        if self.eval_metrics['eval_loss']:
            self.ax1.plot(self.eval_metrics['steps'], self.eval_metrics['eval_loss'],
                          label='Val Loss', color='red')

        self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training and Validation Losses')
        self.ax1.grid(True)
        self.ax1.legend()

        # Plot metrics
        if self.eval_metrics['accuracy']:
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['accuracy'],
                          label='Accuracy', color='green')
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['f1_scores_macro'],
                          label='F1 Scores Macro', color='lightgreen')
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['accuracy_2class'],
                          label='Accuracy 2class', color='orange')
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['f1_scores_macro_2class'],
                          label='F1 Scores Macro 2class', color='yellow')

        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Score')
        self.ax2.set_title('Validation Metrics')
        self.ax2.grid(True)
        self.ax2.legend()

        plt.tight_layout()
        plt.show()

def train_aste_model(args, train_data, eval_data):
    device = args.device
    config = AspectEnhancedBertConfig(
        num_labels=3,
        loss_type=args.loss_type,
        focal_alpha=0.8,  # 0.8
        focal_gamma=2.0,  # 2.0
        label_smoothing=0.1,
        multitask=True,
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
        metric_for_best_model='F1_Macro',
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    trainer = AspectAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        callbacks=[LossRecorderCallback()]
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
        with_aste=True  # 在这进行tokenize
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

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 启用Cudnn

def plot_model(model, tokenizer):
    inputs_ids = tokenizer("This is a sample input", return_tensors="pt")['input_ids']
    attention_mask = tokenizer("This is a sample input", return_tensors="pt")['attention_mask']
    triplet_input_ids = tokenizer("This is a sample input", return_tensors="pt")['input_ids']
    triplet_attention_mask = tokenizer("This is a sample input", return_tensors="pt")['attention_mask']
    inputs = {
        "input_ids": inputs_ids,
        "attention_mask": attention_mask,
        "triplet_input_ids": triplet_input_ids,
        "triplet_attention_mask": triplet_attention_mask,
        "labels": torch.tensor([1]),
        "sentiment_labels": torch.tensor([1])
    }
    outputs = model(**inputs)
    # make_dot(outputs['logits'], params=dict(model.named_parameters())).render("model", format="png")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="./tensorboard_logs")

    # Log the model's structure to TensorBoard
    writer.add_graph(model, [inputs_ids, attention_mask, triplet_input_ids, triplet_attention_mask])

    # Close the writer
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="roberta-llama3.1405B-twitter-sentiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="ce_loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)

    # plot_model(AspectEnhancedBertModel(AspectEnhancedBertConfig(backbone_model=args.model_name)), AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}"))

    main(args)