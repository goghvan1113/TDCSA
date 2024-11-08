import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, PreTrainedModel, AutoModel, Trainer,
    PretrainedConfig, AutoTokenizer, TrainingArguments,
    TrainerCallback, DataCollatorWithPadding
)

from bert.custom_loss import MultiFocalLoss
from bert.plot_results import plot_confusion_matrix

CATEGORY2ID = {
    'METHODOLOGY': 0,
    'PERFORMANCE': 1,
    'INNOVATION': 2,
    'APPLICABILITY': 3,
    'LIMITATION': 4,
    'COMPARISON': 5
}
ID2CATEGORY = {v: k for k, v in CATEGORY2ID.items()}

class ASQPMultitaskConfig(PretrainedConfig):
    """配置类 - 引文分类、四元组分类和情感预测多任务"""

    def __init__(
            self,
            citation_sentiment_labels: int = 3,  # 引文情感分类标签数(positive,negative,neutral)
            aspect_sentiment_labels: int = 2,  # 四元组情感分类标签数(positive,negative)
            aspect_category_labels: int = 6,  # 方面类别标签数
            loss_weights: dict = {  # 任务权重
                "citation_sentiment": 1.0,
                "aspect_sentiment": 1.0,
                "aspect_category": 1.0
            },
            task_specific_layers: int = 2,  # 任务特定层数
            shared_hidden_size: int = 768,  # 共享特征维度
            task_hidden_size: int = 512,  # 任务特定特征维度
            backbone_model: str = "scibert-scivocab-uncased",  # 预训练模型
            max_aspect_length: int = 32,  # 方面词最大长度
            aspect_fusion: str = "attention",  # 方面词融合方式
            dropout_prob: float = 0.1,
            label_smoothing: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)

        backbone_config = AutoConfig.from_pretrained(f'../pretrain_models/{backbone_model}')

        self.hidden_size = backbone_config.hidden_size
        self.citation_sentiment_labels = citation_sentiment_labels
        self.aspect_sentiment_labels = aspect_sentiment_labels
        self.aspect_category_labels = aspect_category_labels
        self.loss_weights = loss_weights
        self.task_specific_layers = task_specific_layers
        self.shared_hidden_size = shared_hidden_size
        self.task_hidden_size = task_hidden_size
        self.backbone_model = backbone_model
        self.max_aspect_length = max_aspect_length
        self.aspect_fusion = aspect_fusion
        self.dropout_prob = dropout_prob
        self.label_smoothing = label_smoothing


class TaskSpecificEncoder(nn.Module):
    """任务特定编码器"""

    def __init__(self, config: ASQPMultitaskConfig, task_name: str):
        super().__init__()

        layers = []
        input_size = config.shared_hidden_size

        for _ in range(config.task_specific_layers):
            layers.extend([
                nn.Linear(input_size, config.task_hidden_size),
                nn.LayerNorm(config.task_hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob)
            ])
            input_size = config.task_hidden_size

        self.encoder = nn.Sequential(*layers)
        self.task_name = task_name

    def forward(self, x):
        return self.encoder(x)


class AspectFusion(nn.Module):
    """方面词融合模块"""

    def __init__(self, config: ASQPMultitaskConfig):
        super().__init__()

        self.fusion_type = config.aspect_fusion # 方面词融合方式
        hidden_size = config.hidden_size

        if self.fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=config.dropout_prob,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

        elif self.fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

    def forward(self, text_features, aspect_features, attention_mask=None):
        if self.fusion_type == "attention":
            if attention_mask is not None:
                attention_mask = ~attention_mask.bool()
            # 交叉注意力
            attn_output, _ = self.attention(
                query=text_features,
                key=aspect_features,
                value=aspect_features,
                key_padding_mask=attention_mask
            )
            return self.layer_norm(text_features + attn_output)

        elif self.fusion_type == "gate":
            # 计算门控权重
            text_expanded = text_features.unsqueeze(1).expand(-1, aspect_features.size(1), -1)
            gate_input = torch.cat([text_expanded, aspect_features], dim=-1)
            gate = self.gate(gate_input)
            # 应用门控
            gated_aspect = gate * aspect_features
            # 池化得到固定维度特征
            fused = torch.mean(gated_aspect, dim=1)
            return fused

        else:  # concat
            return torch.cat([
                text_features,
                torch.mean(aspect_features, dim=1)
            ], dim=-1)


class ASQPMultitaskModel(PreTrainedModel):
    config_class = ASQPMultitaskConfig

    def __init__(self, config: ASQPMultitaskConfig):
        super().__init__(config)
        # 共享编码器
        self.encoder = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        # 方面词融合模块
        self.aspect_fusion = AspectFusion(config)

        # 特征转换
        fusion_output_size = (
            config.hidden_size * 2
            if config.aspect_fusion == "concat"
            else config.hidden_size
        )

        self.shared_transform = nn.Sequential(
            nn.Linear(fusion_output_size, config.shared_hidden_size),
            nn.LayerNorm(config.shared_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob)
        )

        # 任务特定编码器
        self.citation_sentiment_encoder = TaskSpecificEncoder(config, "citation_sentiment")
        self.aspect_sentiment_encoder = TaskSpecificEncoder(config, "aspect_sentiment")
        self.aspect_category_encoder = TaskSpecificEncoder(config, "aspect_category")

        # 分类器
        self.citation_sentiment_classifier = nn.Linear(
            config.task_hidden_size,
            config.citation_sentiment_labels
        )
        self.aspect_sentiment_classifier = nn.Linear(
            config.task_hidden_size,
            config.aspect_sentiment_labels
        )
        self.aspect_category_classifier = nn.Linear(
            config.task_hidden_size,
            config.aspect_category_labels
        )

        self.init_weights()

    def encode_text(self, input_ids, attention_mask, aspect_ids=None, aspect_mask=None):
        # 保持不变...
        outputs = self.encoder(input_ids, attention_mask)
        text_features = outputs.pooler_output

        if aspect_ids is not None:
            aspect_outputs = self.encoder(aspect_ids, aspect_mask)
            aspect_features = aspect_outputs.last_hidden_state
            text_features = self.aspect_fusion(
                text_features.unsqueeze(1),
                aspect_features,
                aspect_mask
            )
            if text_features.dim() == 3:
                text_features = text_features.squeeze(1)

        return text_features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            aspect_ids=None,
            aspect_mask=None,
            citation_input_ids=None,
            citation_attention_mask=None,
            citation_sentiment_labels=None,
            aspect_sentiment_labels=None,
            aspect_category_labels=None
    ):
        outputs = {}

        # 引文级情感分类
        if citation_input_ids is not None:
            citation_features = self.encode_text(
                citation_input_ids,
                citation_attention_mask
            )
            shared_features = self.shared_transform(citation_features)
            citation_sentiment_specific = self.citation_sentiment_encoder(shared_features)
            citation_sentiment_logits = self.citation_sentiment_classifier(citation_sentiment_specific)
            outputs['citation_sentiment_logits'] = citation_sentiment_logits

        # 方面级任务
        if input_ids is not None and aspect_ids is not None and (aspect_category_labels != -1).any():
            # 编码文本和方面词
            aspect_features = self.encode_text(
                input_ids,
                attention_mask,
                aspect_ids,
                aspect_mask
            )
            shared_features = self.shared_transform(aspect_features)

            # 方面级情感分类
            aspect_sentiment_specific = self.aspect_sentiment_encoder(shared_features)
            aspect_sentiment_logits = self.aspect_sentiment_classifier(aspect_sentiment_specific)
            outputs['aspect_sentiment_logits'] = aspect_sentiment_logits

            # 方面类别分类
            aspect_category_specific = self.aspect_category_encoder(shared_features)
            aspect_category_logits = self.aspect_category_classifier(aspect_category_specific)
            outputs['aspect_category_logits'] = aspect_category_logits

        # 训练时计算损失
        if any(label is not None for label in
               [citation_sentiment_labels, aspect_sentiment_labels, aspect_category_labels]):
            total_loss = 0.0

            # 引文级情感分类损失
            if citation_sentiment_labels is not None and 'citation_sentiment_logits' in outputs:
                citation_sentiment_loss = F.cross_entropy(
                    outputs['citation_sentiment_logits'],
                    citation_sentiment_labels,
                    label_smoothing=self.config.label_smoothing
                )
                total_loss += self.config.loss_weights['citation_sentiment'] * citation_sentiment_loss
                outputs['citation_sentiment_loss'] = citation_sentiment_loss

            # 方面级情感分类损失
            if aspect_sentiment_labels is not None and 'aspect_sentiment_logits' in outputs:
                valid_indices = aspect_sentiment_labels != -1  # Ignore labels with -1
                if valid_indices.any():
                    aspect_sentiment_loss = F.cross_entropy(
                        outputs['aspect_sentiment_logits'][valid_indices],
                        aspect_sentiment_labels[valid_indices],
                        label_smoothing=self.config.label_smoothing
                    )
                    total_loss += self.config.loss_weights['aspect_sentiment'] * aspect_sentiment_loss
                    outputs['aspect_sentiment_loss'] = aspect_sentiment_loss

            # 方面类别分类损失
            if aspect_category_labels is not None and 'aspect_category_logits' in outputs:
                valid_indices = aspect_category_labels != -1  # Ignore labels with -1
                if valid_indices.any():
                    aspect_category_loss = F.cross_entropy(
                        outputs['aspect_category_logits'][valid_indices],
                        aspect_category_labels[valid_indices],
                        label_smoothing=self.config.label_smoothing
                    )
                    total_loss += self.config.loss_weights['aspect_category'] * aspect_category_loss
                    outputs['aspect_category_loss'] = aspect_category_loss

            outputs['loss'] = total_loss

        return outputs


# 修改 compute_metrics 函数，计算三个任务的评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 解包预测结果
    citation_sentiment_logits = predictions['citation_sentiment_logits']
    aspect_sentiment_logits = predictions.get('aspect_sentiment_logits') # get方法，如果不存在返回None
    aspect_category_logits = predictions.get('aspect_category_logits')

    # 解包标签
    citation_sentiment_labels = labels['citation_sentiment_labels']
    aspect_sentiment_labels = labels.get('aspect_sentiment_labels')
    aspect_category_labels = labels.get('aspect_category_labels')

    # 计算引文级情感分类指标
    citation_sentiment_preds = np.argmax(citation_sentiment_logits, axis=1)
    citation_sentiment_metrics = classification_report(
        citation_sentiment_labels,
        citation_sentiment_preds,
        target_names=['Neutral', 'Positive', 'Negative'],
        output_dict=True,
        zero_division=0
    )
    # 从报告中提取需要的指标，例如宏平均 F1 分数
    citation_sentiment_f1 = citation_sentiment_metrics['macro avg']['f1-score']

    # 初始化结果字典
    metrics = {
        'citation_sentiment_f1': citation_sentiment_f1,
        'citation_sentiment_accuracy': citation_sentiment_metrics['accuracy']
    }

    # 如果存在方面级情感标签，计算方面级情感分类指标
    if aspect_sentiment_labels is not None and aspect_sentiment_logits is not None:
        valid_indices = aspect_sentiment_labels != -1  # 过滤无效标签
        if np.any(valid_indices):
            aspect_sentiment_preds = np.argmax(aspect_sentiment_logits[valid_indices], axis=1)
            aspect_sentiment_metrics = classification_report(
                aspect_sentiment_labels[valid_indices],
                aspect_sentiment_preds,
                target_names=['Negative', 'Positive'],
                output_dict=True,
                zero_division=0
            )
            aspect_sentiment_f1 = aspect_sentiment_metrics['macro avg']['f1-score']
            metrics.update({
                'aspect_sentiment_f1': aspect_sentiment_f1,
                'aspect_sentiment_accuracy': aspect_sentiment_metrics['accuracy']
            })

    # 如果存在方面类别标签，计算方面类别分类指标
    if aspect_category_labels is not None and aspect_category_logits is not None:
        valid_indices = aspect_category_labels != -1  # 过滤无效标签
        if np.any(valid_indices):
            aspect_category_preds = np.argmax(aspect_category_logits[valid_indices], axis=1)
            aspect_category_metrics = classification_report(
                aspect_category_labels[valid_indices],
                aspect_category_preds,
                # 使用真实的类别名称替代Category_i
                target_names=list(CATEGORY2ID.keys()),
                output_dict=True,
                zero_division=0
            )
            aspect_category_f1 = aspect_category_metrics['macro avg']['f1-score']
            metrics.update({
                'aspect_category_f1': aspect_category_f1,
                'aspect_category_accuracy': aspect_category_metrics['accuracy']
            })

    return metrics


class ASQPCitationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            texts: List[str],  # 引文文本
            citation_sentiment_labels: List[int],  # 引文级情感标签
            aspects: List[Optional[str]],  # 方面词列表
            opinions: List[Optional[str]],  # 观点词列表
            aspect_categories: List[Optional[int]],  # 方面类别标签
            aspect_sentiments: List[Optional[int]],  # 方面级情感标签
            tokenizer=None,
            max_length: int = 512,
            max_aspect_length: int = 32
    ):
        self.texts = texts
        self.citation_sentiment_labels = citation_sentiment_labels
        self.aspects = aspects
        self.opinions = opinions
        self.aspect_categories = aspect_categories
        self.aspect_sentiments = aspect_sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_aspect_length = max_aspect_length

    def _build_aspect_categories(self):
        # 构建方面类别标签
        aspect_categories = []
        for category in self.aspect_categories:
            if category is not None:
                aspect_categories.append(category)
            else:
                aspect_categories.append(-1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 编码引文文本
        text_encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
        item = {
            'citation_input_ids': text_encoding['input_ids'].squeeze(0),
            'citation_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'citation_sentiment_labels': torch.tensor(self.citation_sentiment_labels[idx])
        }
    
        # 如果存在四元组相关数据，进行编码
        if self.aspects[idx] is not None:
            # 编码方面词
            aspect_encoding = self.tokenizer(
                self.aspects[idx],
                padding='max_length',
                truncation=True,
                max_length=self.max_aspect_length,
                return_tensors='pt'
            )
    
            # 构造输入文本(包含方面词和观点词的上下文)
            aspect_text = f"{self.texts[idx]} [SEP] {self.aspects[idx]} {self.opinions[idx]}"
            aspect_text_encoding = self.tokenizer(
                aspect_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
    
            item.update({
                'input_ids': aspect_text_encoding['input_ids'].squeeze(0),
                'attention_mask': aspect_text_encoding['attention_mask'].squeeze(0),
                'aspect_ids': aspect_encoding['input_ids'].squeeze(0),
                'aspect_mask': aspect_encoding['attention_mask'].squeeze(0),
                'aspect_category_labels': torch.tensor(self.aspect_categories[idx]),
                'aspect_sentiment_labels': torch.tensor(self.aspect_sentiments[idx])
            })
        else:
            # 如果没有方面信息，使用占位符
            item.update({
                'input_ids': text_encoding['input_ids'].squeeze(0),
                'attention_mask': text_encoding['attention_mask'].squeeze(0),
                'aspect_ids': torch.zeros(self.max_aspect_length, dtype=torch.long),
                'aspect_mask': torch.zeros(self.max_aspect_length, dtype=torch.long),
                'aspect_category_labels': torch.tensor(-1),  # 使用 -1 表示无效标签
                'aspect_sentiment_labels': torch.tensor(-1)
            })
    
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

def create_datasets(split_data: Dict, tokenizer):
    """创建训练、验证和测试数据集，包含中性样本。"""
    datasets = {}
    for split_name in ['train', 'val', 'test']:
        pos_neg_samples = split_data[split_name]['pos_neg_samples']
        neutral_texts = split_data[split_name]['neutral_samples']
        
        texts = []
        citation_sentiment_labels = []
        aspects = []
        opinions = []
        aspect_categories = []
        aspect_sentiments = []
        
        # 处理正负样本
        valid_quadruples = []
        for sample in pos_neg_samples:
            valid_quads = []
            for quad in sample['quads']:
                if quad[2] in CATEGORY2ID:
                    valid_quads.append(quad)
            if valid_quads:
                valid_quadruples.append({
                    'text': sample['text'],
                    'overall_sentiment': sample['overall_sentiment'],
                    'quads': valid_quads
                })

        for sample in valid_quadruples:
            text = sample['text']
            overall_sentiment = sample['overall_sentiment']
            sentiment_label = {'positive': 1, 'negative': 2}[overall_sentiment]  # 正类为1，负类为2
            if 'quads' in sample and sample['quads']:
                for quad in sample['quads']:
                    texts.append(text)
                    citation_sentiment_labels.append(sentiment_label)
                    aspects.append(quad[0])  # aspect
                    opinions.append(quad[1])  # opinion
                    # 将方面类别字符串转换为对应的ID
                    aspect_categories.append(CATEGORY2ID[quad[2]])  # 使用映射字典转换类别标签
                    aspect_sentiments.append(1 if quad[3] == 'positive' else 0)  # sentiment
            else:
                # 如果没有四元组，仍然包含该样本
                texts.append(text)
                citation_sentiment_labels.append(sentiment_label)
                aspects.append(None)
                opinions.append(None)
                aspect_categories.append(None)
                aspect_sentiments.append(None)
        
        # 处理中性样本（没有四元组信息）
        for text in neutral_texts:
            texts.append(text)
            citation_sentiment_labels.append(0)  # 中性类标签为0
            aspects.append(None)
            opinions.append(None)
            aspect_categories.append(-1)  # Set to -1 instead of None
            aspect_sentiments.append(-1)  # Set to -1 instead of None

        # 打乱数据顺序
        combined = list(
            zip(texts, citation_sentiment_labels, aspects, opinions, aspect_categories, aspect_sentiments))
        random.shuffle(combined)
        texts, citation_sentiment_labels, aspects, opinions, aspect_categories, aspect_sentiments = zip(*combined)

        # 创建数据集
        datasets[split_name] = ASQPCitationDataset(
            texts=list(texts),
            citation_sentiment_labels=list(citation_sentiment_labels),
            aspects=list(aspects),
            opinions=list(opinions),
            aspect_categories=list(aspect_categories),
            aspect_sentiments=list(aspect_sentiments),
            tokenizer=tokenizer
        )
    return datasets['train'], datasets['val'], datasets['test']

def load_acos_data_verified(quad_file=None, corpus_file=None):
    """加载四元组数据"""
    pos_neg_samples = []
    with open(quad_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            pos_neg_samples.append({
                'text': item['text'],
                'overall_sentiment': item['overall_sentiment'],
                'quads': item['final_quadruples'] # 加载verified的四元组
            })

    # 加载中性样本
    df = pd.read_csv(corpus_file)
    neutral_samples = df[df['Sentiment'] == 'Neutral']['Text'].tolist()

    return pos_neg_samples, neutral_samples


def train_acos_model(args, train_data, eval_data):
    device = args.device
    config = ASQPMultitaskConfig(
        citation_sentiment_labels=3,
        aspect_sentiment_labels=2,
        aspect_category_labels=6,
        loss_weights={"citation_sentiment": 1.0, "aspect_sentiment": 1.0, "aspect_category": 1.0},
        backbone_model=args.model_name,
        label_smoothing=0.1,
    )

    model = ASQPMultitaskModel(config).to(device)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")

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
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='citation_sentiment_f1',
        greater_is_better=True
    )

    # 自定义数据整理器，处理标签中的 -1 值
    def data_collator(features):
        batch = DataCollatorWithPadding(tokenizer)(features)
        # 将标签转换为张量，并将 -1 转换为忽略的索引
        for key in ['citation_sentiment_labels', 'aspect_sentiment_labels', 'aspect_category_labels']:
            if key in batch:
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return model


# 修改 ModelEvaluator 类，评估三个任务的性能
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
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )

        all_citation_sentiment_preds = []
        all_citation_sentiment_labels = []
        all_aspect_sentiment_preds = []
        all_aspect_sentiment_labels = []
        all_aspect_category_preds = []
        all_aspect_category_labels = []

        for batch in dataloader:
            # 将数据移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            # 引文级情感分类
            citation_sentiment_preds = torch.argmax(outputs['citation_sentiment_logits'], dim=1)
            all_citation_sentiment_preds.extend(citation_sentiment_preds.cpu().numpy())
            all_citation_sentiment_labels.extend(batch['citation_sentiment_labels'].cpu().numpy())

            # 方面级情感分类
            if 'aspect_sentiment_logits' in outputs:
                valid_indices = batch['aspect_sentiment_labels'] != -1
                if valid_indices.any():
                    aspect_sentiment_preds = torch.argmax(outputs['aspect_sentiment_logits'], dim=1)
                    all_aspect_sentiment_preds.extend(aspect_sentiment_preds[valid_indices].cpu().numpy())
                    all_aspect_sentiment_labels.extend(batch['aspect_sentiment_labels'][valid_indices].cpu().numpy())

            # 方面类别分类
            if 'aspect_category_logits' in outputs:
                valid_indices = batch['aspect_category_labels'] != -1
                if valid_indices.any():
                    aspect_category_preds = torch.argmax(outputs['aspect_category_logits'], dim=1)
                    all_aspect_category_preds.extend(aspect_category_preds[valid_indices].cpu().numpy())
                    all_aspect_category_labels.extend(batch['aspect_category_labels'][valid_indices].cpu().numpy())

        # 计算引文级情感分类指标
        citation_sentiment_report = classification_report(
            all_citation_sentiment_labels,
            all_citation_sentiment_preds,
            target_names=['Neutral', 'Positive', 'Negative'],
            digits=4
        )
        print("\nCitation Sentiment Classification Report:")
        print(citation_sentiment_report)

        # 计算方面级情感分类指标
        if all_aspect_sentiment_labels:
            aspect_sentiment_report = classification_report(
                all_aspect_sentiment_labels,
                all_aspect_sentiment_preds,
                target_names=['Negative', 'Positive'],
                digits=4
            )
            print("\nAspect Sentiment Classification Report:")
            print(aspect_sentiment_report)

        # 计算方面类别分类指标
        if all_aspect_category_labels:
            aspect_category_report = classification_report(
                all_aspect_category_labels,
                all_aspect_category_preds,
                target_names=list(CATEGORY2ID.keys()),  # 使用真实的类别名称
                digits=4
            )
            print("\nAspect Category Classification Report:")
            print(aspect_category_report)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用Cudnn加速，保证结果可复现

def main(args):
    # 加载数据
    quad_file = f'../output/sentiment_asqp_results_corpus_expand_verified.json'
    corpus_file = '../data/citation_sentiment_corpus_expand.csv'
    pos_neg_samples, neutral_samples = load_acos_data_verified(quad_file, corpus_file)

    # 划分数据集
    splitter = DatasetSplitter(
        pos_neg_samples=pos_neg_samples,
        neutral_samples=neutral_samples,
        test_size=0.2,
        val_size=0.1,
        random_state=args.seed
    )
    split_data = splitter.split_data(stratify_by_sentiment=True)
    # 打印数据集统计信息
    for split_name, split in split_data.items():
        pos_neg_dist = Counter(s['overall_sentiment'] for s in split['pos_neg_samples'])
        print(f"\n{split_name} set distribution:")
        print(f"Positive: {pos_neg_dist.get('positive', 0)}")
        print(f"Negative: {pos_neg_dist.get('negative', 0)}")
        print(f"Neutral: {len(split['neutral_samples'])}")

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(split_data, tokenizer)

    # 训练模型
    model = train_acos_model(args, train_dataset, val_dataset)
    # 评估模型
    # evaluator = ModelEvaluator(model, tokenizer)
    #
    # print("\nValidation Set Results:")
    # evaluator.evaluate(val_dataset)
    #
    # print("\nTest Set Results:")
    # evaluator.evaluate(test_dataset)


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
    parser.add_argument("--loss_type", type=str, default="ce")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)

