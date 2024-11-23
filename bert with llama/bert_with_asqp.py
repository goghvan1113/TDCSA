import argparse
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

from matplotlib import pyplot as plt, gridspec
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoConfig, PreTrainedModel, AutoModel, Trainer, PretrainedConfig, AutoTokenizer, \
    TrainingArguments, TrainerCallback, DataCollatorWithPadding


from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Union, List, Any, Tuple, Mapping, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bert.custom_loss import MultiFocalLoss
from bert.plot_results import plot_confusion_matrix
from AttentionVisualization import visualize_model_attention

class QuadAspectEnhancedBertConfig(PretrainedConfig):
    """配置类 - 增加四元组相关配置"""
    def __init__(
            self,
            num_labels: int = 3,
            loss_type: str = "ce",
            loss_weights: dict = {  # 任务权重
                "citation_sentiment": 1.0,
                "subject_sentiment": 0.5
            },
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            label_smoothing: float = 0.0,
            multitask: bool = True,
            backbone_model: str = "roberta-base",
            hidden_size: Optional[int] = None,
            hidden_dropout_prob: float = 0.1,
            num_categories: int = 5,  # 添加类别数量
            **kwargs
    ):
        super().__init__(**kwargs)

        backbone_config = AutoConfig.from_pretrained(f'../pretrain_models/{backbone_model}')

        self.hidden_size = hidden_size or backbone_config.hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob

        self.num_labels = num_labels
        self.loss_type = loss_type
        self.loss_weights = loss_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.multitask = multitask
        self.label_smoothing = label_smoothing
        self.backbone_model = backbone_model
        self.num_categories = num_categories


class NonLinearFusion(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()

        # 门控机制
        self.gate_text = nn.Linear(hidden_size, hidden_size)
        self.gate_quad = nn.Linear(hidden_size, hidden_size)
        self.gate_attn = nn.Linear(hidden_size, hidden_size)

        # 特征转换
        self.transform_text = nn.Linear(hidden_size, hidden_size)
        self.transform_quad = nn.Linear(hidden_size, hidden_size)
        self.transform_attn = nn.Linear(hidden_size, hidden_size)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 3)

        # 非线性变换
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 4, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3)
        )

    def forward(self, text_features, quad_features, attn_features):
        # 计算门控权重
        text_gate = torch.sigmoid(self.gate_text(text_features))
        quad_gate = torch.sigmoid(self.gate_quad(quad_features))
        attn_gate = torch.sigmoid(self.gate_attn(attn_features))

        # 特征转换
        text_transformed = self.transform_text(text_features)
        quad_transformed = self.transform_quad(quad_features)
        attn_transformed = self.transform_attn(attn_features)

        # 应用门控机制
        text_gated = text_gate * text_transformed
        quad_gated = quad_gate * quad_transformed
        attn_gated = attn_gate * attn_transformed

        # 特征拼接
        combined = torch.cat([text_gated, quad_gated, attn_gated], dim=-1)
        normalized = self.layer_norm(combined)
        fused = self.fusion_layer(normalized)

        return fused

class QuadAspectEnhancedBertModel(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig
    def __init__(self, config: QuadAspectEnhancedBertConfig):
        super().__init__(config)

        # 基础编码器
        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.quad_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        hidden_size = config.hidden_size

        # 类别感知层
        self.category_embedding = nn.Embedding(config.num_categories, hidden_size)

        # 交互注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
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
        self.quad_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        # 融合层
        self.fusion = NonLinearFusion(hidden_size, config.hidden_dropout_prob)

        # 分类器
        self.classifier = nn.Linear(hidden_size * 3, config.num_labels)
        # 情感检测器
        self.sentiment_detector = nn.Linear(hidden_size, 2)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            quad_input_ids=None,
            quad_attention_mask=None,
            category_ids=None,
            labels=None,
            sentiment_labels=None
    ):
        # 文本编码
        text_outputs = self.bert(input_ids, attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output

        # 四元组编码
        quad_outputs = self.quad_bert(quad_input_ids, quad_attention_mask)
        quad_hidden = quad_outputs.last_hidden_state
        quad_pooled = quad_outputs.pooler_output

        # 类别增强
        if category_ids is not None:
            category_embeds = self.category_embedding(category_ids)
            quad_hidden = quad_hidden + category_embeds.unsqueeze(1)

        # 特征转换
        text_transformed = self.text_transform(text_hidden)
        quad_transformed = self.quad_transform(quad_hidden)

        # 交互注意力
        cross_attn_output, _ = self.cross_attention(
            query=text_transformed,
            key=quad_transformed,
            value=quad_transformed,
            key_padding_mask=~quad_attention_mask.bool()
        )
        self_attn_output, _ = self.self_attention(
            query=cross_attn_output,
            key=cross_attn_output,
            value=cross_attn_output,
            key_padding_mask=~attention_mask.bool()
        )
        attn_pooled = torch.mean(self_attn_output, dim=1)

        # 特征融合
        fused = self.fusion(text_pooled, quad_pooled, attn_pooled)

        # 分类预测
        logits = self.classifier(fused)
        logits_probs = torch.softmax(logits, dim=-1)
        # 使用概率加权
        sentiment_logits = self.sentiment_detector(text_pooled)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        # 修改后的逻辑：使用概率加权
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)  # 客观的概率
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)  # 主观的概率
        # 混合预测结果
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)

        # 转换回logits形式
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits, # 这个指标才是影响混淆矩阵最左边一列的数据
            'sentiment_logits': sentiment_logits,
            'embeddings': fused,  # Return embeddings before classification
            'text_pooled': text_pooled  # Return text_pooled outputs
        }

class AspectAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        sentiment_labels = inputs.pop("sentiment_labels")
        outputs = model(**inputs)
        logits = outputs['logits']
        sentiment_logits = outputs['sentiment_logits'] # 主客观分类

        if model.config.loss_type == "focal_loss":
            focal_loss_3labels = MultiFocalLoss(num_class=model.config.num_labels, alpha=model.config.focal_alpha,
                                                gamma=model.config.focal_gamma)
            focal_loss_2labels = MultiFocalLoss(num_class=2, alpha=model.config.focal_alpha,
                                                gamma=model.config.focal_gamma)
            loss_3class = focal_loss_3labels(logits, labels)
            loss_2class = focal_loss_2labels(sentiment_logits, sentiment_labels)
        else:
            loss_3class = F.cross_entropy(logits, labels, label_smoothing=model.config.label_smoothing)
            loss_2class = F.cross_entropy(sentiment_logits, sentiment_labels,
                                          label_smoothing=model.config.label_smoothing)

        loss = model.config.loss_weights["citation_sentiment"] * loss_3class
        if model.config.multitask:
            loss += model.config.loss_weights["subject_sentiment"] * loss_2class

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


class QuadAspectDataProcessor:
    """数据预处理器 - 处理四元组数据"""

    def __init__(self, tokenizer: Any, max_length: int = 512,
                 backbone_model: str = 'bert-base-uncased'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.backbone_model = backbone_model.lower()
        self.category2id = {  # 预定义类别映射
            'METHODOLOGY': 0,
            'PERFORMANCE': 1,
            'INNOVATION': 2,
            'APPLICABILITY': 3,
            'LIMITATION': 4,
        }

        # 中性样本处理的噪声词库
        self.noise_tokens = [
                        "neutral", "sample", "info", "example", "text", "generic", "data", "reference", "analysis",
                        "method", "approach", "context", "details", "review", "background", "overview", "basis",
                        "supporting", "summary", "aspect", "topic", "result", "content", "framework", "definition",
                        "perspective", "framework", "parameter", "hypothesis", "purpose", "context", "concept",
                        "finding", "discussion", "focus", "case", "observation", "outline", "description",
                        "literature", "citation", "source", "statement", "concept", "objective", "insight",
                        "overview", "scope", "narrative", "data", "sampling", "parameters", "model", "application",
                        "approach", "contribution", "aspect", "point", "highlight", "field", "angle", "review",
                        "data", "collection", "survey", "discussion", "observation", "analysis", "phenomenon",
                        "evidence", "evaluation", "factor", "basis", "insight", "record", "statistical", "note",
                        "term", "application", "practice", "theme", "range", "pattern", "structure", "strategy",
                        "background", "core", "survey", "source", "option", "component", "variable", "output",
                        "input", "equation", "notation", "framework", "methodology", "reference", "technique",
                        "context", "standard", "goal", "element", "operation", "material", "topic", "theory",
                        "format", "hypothesis", "data", "overview", "section", "outline", "support", "summary",
                        "literature", "definition", "metric", "aspect", "version", "pathway", "background",
                        "factor", "function", "assessment", "process", "insight", "highlight", "trend",
                        "hypothesis", "design", "view", "procedure", "summary", "representation", "property",
                        "construct", "aspect", "formula", "description", "subsection", "principle", "element",
                        "protocol", "phase", "inference", "statement", "illustration", "application", "framework",
                        "notation", "dataset", "theory", "phenomenon", "measure", "instance", "idea",
                        "proposition", "foundation", "structure", "analysis", "foundation", "term", "description",
                        "idea", "conception", "feature", "element", "hypothesis", "evaluation", "syntax",
                        "concept", "notion", "comparison", "scheme", "basis", "modeling", "pattern", "aspect",
                        "object", "theory", "property", "domain", "notation", "factor", "rationale", "viewpoint"
        ]
        # 设置mask token
        self.mask_token = self._get_mask_token()

    def _get_mask_token(self) -> str:
        """
        根据backbone model确定mask token
        仅判断是否包含roberta或bert字符串

        Returns:
            对应模型的mask token
        """
        if 'roberta' in self.backbone_model:
            return '<mask>'
        # 其他情况默认使用bert的mask token
        return '[MASK]'

    def _process_original_quad(self, feature: Dict) -> Tuple[str, int]:
        """处理原始四元组"""
        quads = feature.get('quads', [])
        quad_parts = []
        feature_categories = []

        for aspect, opinion, category, polarity, confidence in quads:
            quad_template = f"This citation expresses {self.mask_token} sentiment towards {aspect} through {opinion} which belongs to {category} category"
            quad_parts.append(quad_template)
            feature_categories.append(category)

        quad_text = " ; ".join(quad_parts)
        # 使用最常见的类别作为主类别
        category_id = (self.category2id.get(max(set(feature_categories),
                                                key=feature_categories.count)) if feature_categories else 0)

        return quad_text, category_id

    def _process_empty_quad(self, feature: Dict) -> Tuple[str, int]:
        """返回空四元组"""
        return '', 0

    def _process_random_mask_quad(self, feature: Dict) -> Tuple[str, int]:
        """使用随机掩码处理四元组"""
        num_neutral_tags = random.randint(1, 2)
        num_words = random.randint(2, 5)

        # 随机选择一个类别
        category = random.choice(list(self.category2id.keys()))
        # 获取该类别的随机词
        neutral_tags = [self.mask_token] * num_neutral_tags
        random_words = random.sample(self.noise_tokens, num_words)

        # 组合并打乱词序
        combined_list = neutral_tags + random_words
        random.shuffle(combined_list)

        # 确保长度合适
        if len(combined_list) < 5:
            combined_list += [self.mask_token] * (5 - len(combined_list))

        quad_text = " ".join(combined_list[:5])
        return quad_text, self.category2id[category]

    def _process_random_polar_quad(self, feature: Dict, pos_neg_quads: List) -> Tuple[str, int]:
        """使用随机极性处理四元组"""
        if not pos_neg_quads:
            return "", 0

        aspect, opinion, category, polarity, confidence = random.choice(pos_neg_quads)
        quad_text = f"This citation expresses {self.mask_token} sentiment towards {aspect} through {opinion} which belongs to {category} category"
        category_id = self.category2id.get(category, 0)

        return quad_text, category_id

    def _process_random_words_quad(self, feature: Dict) -> Tuple[str, int]:
        """使用随机词处理四元组"""
        text_tokens = feature.get('text', '').split()
        aspect = random.choice(text_tokens) if len(text_tokens) >= 2 else 'aspect'
        opinion = random.choice(text_tokens) if len(text_tokens) >= 2 else 'opinion'
        category = random.choice(list(self.category2id.keys()))

        quad_text = f"This citation expresses {self.mask_token} sentiment towards {aspect} through {opinion} which belongs to {category} category"
        category_id = self.category2id.get(category, 0)

        return quad_text, category_id

    def _get_processor_method(self, method: str, original_prob: float = 0) -> Callable:
        """获取处理方法"""
        method_map = {
            'original': self._process_original_quad,
            'empty': self._process_empty_quad,
            'random_mask': self._process_random_mask_quad,
            'random_polar': self._process_random_polar_quad,
            'random_words': self._process_random_words_quad
        }

        if method == 'random_multi':
            methods = ['random_mask', 'random_polar', 'random_words']
            if random.random() < original_prob:
                return method_map['original']
            return method_map[random.choice(methods)]

        return method_map.get(method, self._process_empty_quad)

    def process_features(self, features: List[Dict], method: str = 'original',
                         original_prob: float = 0) -> Dict[str, torch.Tensor]:
        """
        处理四元组特征

        Args:
            features: 特征列表
            method: 处理方法 ('original', 'empty', 'random_mask', 'random_polar',
                    'random_words', 'random_multi')
            original_prob: 使用原始方法的概率 (仅在method='random_multi'时有效)

        Returns:
            处理后的特征字典
        """
        texts = [f.get('text', '') for f in features]
        labels = [f.get('label', 0) for f in features]
        sentiment_labels = [1 if label > 0 else 0 for label in labels]

        # 编码原始文本
        text_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 收集所有正面和负面的四元组
        pos_neg_quads = []
        for feature in features:
            if feature.get('label', 0) != 0:
                pos_neg_quads.extend(feature.get('quads', []))

        # 处理四元组
        quad_texts = []
        category_ids = []
        processor = self._get_processor_method(method, original_prob)

        for feature in features:
            if feature.get('label', 0) == 0:  # 中性样本
                quad_text, category_id = (processor(feature, pos_neg_quads)
                                          if processor.__name__ == '_process_random_polar_quad'
                                          else processor(feature))
            else:  # 非中性样本使用原始处理方法
                quad_text, category_id = self._process_original_quad(feature)

            quad_texts.append(quad_text)
            category_ids.append(category_id)

        # 编码四元组文本
        quad_encoding = self.tokenizer(
            quad_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "quad_input_ids": quad_encoding["input_ids"],
            "quad_attention_mask": quad_encoding["attention_mask"],
            "category_ids": torch.tensor(category_ids),
            "labels": torch.tensor(labels),
            "sentiment_labels": torch.tensor(sentiment_labels)
        }


class AspectAwareDataset(torch.utils.data.Dataset):
    def __init__(self, pos_neg_samples, neutral_samples, tokenizer, processor, max_length=512, method='random_polar'):
        """
        Args:
            pos_neg_samples: List of dictionaries containing positive/negative samples
            neutral_samples: List of neutral text samples
            tokenizer: BERT tokenizer
            processor: AspectDataProcessor instance
            max_length: Maximum sequence length
            method: Method for processing neutral samples
        """
        self.samples = []
        self.samples_tokenized = {}  # Changed to dictionary for better error tracking
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.CATEGORY2ID = {  # 预定义类别映射CATEGORY2ID
            'METHODOLOGY': 0,
            'PERFORMANCE': 1,
            'INNOVATION': 2,
            'APPLICABILITY': 3,
            'LIMITATION': 4
        }

        # Process positive and negative samples
        for item in pos_neg_samples:
            label = 1 if item['overall_sentiment'] == 'positive' else 2
            self.samples.append({
                'text': item['text'],
                'label': label,
                'quads': item.get('quads', [])
            })

        # Process neutral samples and include their quads
        for item in neutral_samples:
            self.samples.append({
                'text': item['text'],
                'label': 0,
                'quads': item.get('quads', [])
            })
        # Shuffle the samples
        random.shuffle(self.samples)
        processed_features = self.processor.process_features(self.samples, method=method)

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

    def split_data(self, stratify_by_sentiment: bool = True):
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

def create_datasets(split_data: Dict, tokenizer, with_asqp: bool = True, method='random_words') -> Tuple[
    AspectAwareDataset, AspectAwareDataset, AspectAwareDataset]:
    """
    创建训练集、验证集和测试集的Dataset对象

    Args:
        :param method:
        :param split_data:
        :param tokenizer:
        :param with_asqp:
    """
    # 训练集始终使用ASTE

    # 验证数据结构
    processor = QuadAspectDataProcessor(tokenizer)

    for split_name, split in split_data.items():
        for sample in split['pos_neg_samples']:
            if 'text' not in sample:
                print(f"Warning: Missing 'text' in {split_name} split sample: {sample}")

    if not with_asqp:
        # Convert to pure text format
        val_neu = [{'text': s['text'],
                        'overall_sentiment': s['overall_sentiment'],
                        'quads': []}  # Empty quadruples
                       for s in split_data['val']['pos_neg_samples']]
        test_neu = [{'text': s['text'],
                         'overall_sentiment': s['overall_sentiment'],
                         'quads': []}  # Empty quadruples
                        for s in split_data['test']['pos_neg_samples']]
        val_pos_neg = [{'text': s['text'],
                        'overall_sentiment': s['overall_sentiment'],
                        'quads': []}  # Empty quadruples
                       for s in split_data['val']['pos_neg_samples']]
        test_pos_neg = [{'text': s['text'],
                         'overall_sentiment': s['overall_sentiment'],
                         'quads': []}  # Empty quadruples
                        for s in split_data['test']['pos_neg_samples']]
        method = 'empty'
    else:
        val_pos_neg = split_data['val']['pos_neg_samples']
        test_pos_neg = split_data['test']['pos_neg_samples']
        val_neu = split_data['val']['neutral_samples']
        test_neu = split_data['test']['neutral_samples']
        method = method

    train_dataset = AspectAwareDataset(
        pos_neg_samples=split_data['train']['pos_neg_samples'],
        neutral_samples=split_data['train']['neutral_samples'],
        tokenizer=tokenizer,
        processor=processor,
        method=method
    )
    val_dataset = AspectAwareDataset(
        pos_neg_samples=val_pos_neg,
        neutral_samples=val_neu,
        tokenizer=tokenizer,
        processor=processor,
        method=method
    )
    test_dataset = AspectAwareDataset(
        pos_neg_samples=test_pos_neg,
        neutral_samples=test_neu,
        tokenizer=tokenizer,
        processor=processor,
        method=method
    )

    return train_dataset, val_dataset, test_dataset


def load_asqp_data(posnegfile=None, neutral_file=None):
    """加载四元组数据"""
    pos_neg_samples = []
    with open(posnegfile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            pos_neg_samples.append({
                'text': item['text'],
                'overall_sentiment': item['overall_sentiment'],
                'quads': item['sentiment_quadruples']  # 修改为sentiment_quadruples
            })

    # 加载中性样本
    neutral_samples = []
    with open(neutral_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item['overall_sentiment'] in ['positive', 'negative', 'neutral']:
            neutral_samples.append({
                'text': item['text'],
                'overall_sentiment': item['overall_sentiment'],
                'quads': item['sentiment_quadruples']  # 修改为sentiment_quadruples
            })

    return pos_neg_samples, neutral_samples


class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='none')

    def _perform_dimension_reduction(self, embeddings, method='tsne', n_components=2, random_state=42):
        """执行降维"""
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=random_state)
        elif method.lower() == 'umap':
            import umap.umap_ as umap
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        else:
            raise ValueError(f"Unsupported dimension reduction method: {method}")

        return reducer.fit_transform(embeddings)

    def _plot_embeddings(self, embeddings_2d, labels, title, timestamp, method):
        """绘制降维后的嵌入向量"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7
        )

        # 添加图例
        legend_labels = ['neutral', 'positive', 'negative']
        handles = [plt.scatter([], [], c=plt.cm.viridis(i / 2.), label=label)
                   for i, label in enumerate(legend_labels)]
        plt.legend(handles=handles)

        plt.title(f'{title} ({method.upper()} Visualization)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # 保存图片
        save_path = f'output/dimension_reduction_{title.lower().replace(" ", "_")}_{method}_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _visualize_attention(self, text, quad_text, quads, timestamp, output_dir='attention_viz'):
        """
        可视化注意力权重,并在图中展示原始四元组

        Args:
            text: 原始文本
            quad_text: 处理后的四元组文本
            quads: 原始四元组列表
            timestamp: 时间戳
            output_dir: 输出目录
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 创建一个大图,包含注意力热图和四元组显示
            fig = plt.figure(figsize=(20, 10))

            # 设置网格布局
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

            # 左侧放置注意力热图
            ax1 = plt.subplot(gs[0])
            attention_fig = visualize_model_attention(
                model=self.model,
                tokenizer=self.tokenizer,
                text=text,
                quad_text=quad_text
            )

            # 将attention_fig中的内容复制到新图的左侧
            for ax_old in attention_fig.axes:
                # 复制热图
                for im in ax_old.images:
                    ax1.imshow(im.get_array(), cmap=im.get_cmap())
                # 复制标签和刻度
                ax1.set_xticks(ax_old.get_xticks())
                ax1.set_xticklabels(ax_old.get_xticklabels(), rotation=45, ha='right')
                ax1.set_yticks(ax_old.get_yticks())
                ax1.set_yticklabels(ax_old.get_yticklabels())

            ax1.set_title("Attention Weights Heatmap")

            # 右侧显示原始四元组
            ax2 = plt.subplot(gs[1])
            ax2.axis('off')  # 隐藏坐标轴

            # 构建四元组展示文本
            quad_display = "Original Quadruples:\n\n"
            for i, (aspect, opinion, category, polarity, confidence) in enumerate(quads, 1):
                quad_display += f"{i}. Aspect: {aspect}\n"
                quad_display += f"   Opinion: {opinion}\n"
                quad_display += f"   Category: {category}\n"
                quad_display += f"   Polarity: {polarity}\n"
                quad_display += f"   Confidence: {confidence:.2f}\n\n"

            # 添加原始文本
            text_display = f"Original Text:\n\n{text}\n\n"

            # 使用文本框显示内容
            text_box = text_display + quad_display
            ax2.text(0, 1, text_box,
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                     transform=ax2.transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     family='monospace',
                     wrap=True)

            # 调整布局
            plt.tight_layout()

            # 保存可视化结果
            viz_path = os.path.join(output_dir, f'attention_visualization_{timestamp}.png')
            plt.savefig(viz_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            plt.close(attention_fig)  # 关闭原始的attention图
            return viz_path

        except Exception as e:
            print(f"Warning: Attention visualization failed with error: {str(e)}")
            return None

    def _select_samples_for_visualization(self, dataset, predictions, labels, k=5):
        """选择用于可视化的样本"""
        visualization_samples = []

        # 获取每个类别的样本索引
        label_indices = {
            'correct': defaultdict(list),  # 每个标签的正确预测
            'incorrect': defaultdict(list)  # 每个标签的错误预测
        }

        # 按照预测正确性和真实标签对样本进行分类
        for idx, (pred, label) in enumerate(zip(predictions, labels)):
            if pred == label:
                label_indices['correct'][label].append(idx)
            else:
                label_indices['incorrect'][label].append(idx)

        # 为每个类别选择样本
        for category in ['correct', 'incorrect']:
            for label in range(3):  # 0: neutral, 1: positive, 2: negative
                indices = label_indices[category][label]
                if indices:
                    # 按损失值排序选择样本
                    sample_scores = [
                        (idx, len(dataset.samples[idx].get('quads', [])))
                        for idx in indices
                    ]
                    # 优先选择有四元组的样本
                    sample_scores.sort(key=lambda x: x[1], reverse=True)
                    selected_indices = [idx for idx, _ in sample_scores[:k]]

                    for idx in selected_indices:
                        sample = dataset.samples[idx]
                        if sample.get('quads'):  # 只选择有四元组的样本
                            visualization_samples.append({
                                'index': idx,
                                'category': category,
                                'text': sample['text'],
                                'quads': sample['quads'],
                                'true_label': label,
                                'predicted_label': predictions[idx]
                            })

        return visualization_samples

    @torch.no_grad()
    def evaluate(self, dataset, batch_size=32, error_analysis=False,
                 dimension_reduction=None, plot_embeddings=True,
                 attention_visualization=False, viz_samples=5):
        """
        评估模型性能并进行错误分析

        Args:
            dataset: 数据集
            batch_size: 批次大小
            error_analysis: 是否进行错误分析
            dimension_reduction: 降维方法 ('tsne', 'umap', 或 None)
            plot_embeddings: 是否绘制嵌入向量图
            attention_visualization: 是否进行注意力可视化
            viz_samples: 每类(正确/错误)选择的可视化样本数
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )

        all_preds = []
        all_labels = []
        all_losses = []
        all_embeddings = []
        all_text_pooled = []
        error_samples = []

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop('labels')

            outputs = self.model(**batch, labels=labels)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)

            sample_losses = self._compute_loss(logits, labels)

            if error_analysis:
                errors = preds != labels
                for i in range(len(preds)):
                    if errors[i]:
                        sample_idx = batch_idx * batch_size + i
                        if sample_idx < len(dataset.samples):
                            sample = dataset.samples[sample_idx]
                            error_info = {
                                'text': sample['text'],
                                'true_label': labels[i].item(),
                                'predicted_label': preds[i].item(),
                                'loss': sample_losses[i].item(),
                                'quads': sample.get('quads', []),
                                'confidence_scores': torch.softmax(logits[i], dim=0).cpu().numpy().tolist()
                            }
                            error_samples.append(error_info)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_losses.extend(sample_losses.cpu().numpy())

            if 'embeddings' in outputs:
                all_embeddings.append(outputs['embeddings'].cpu().numpy())
            if 'text_pooled' in outputs:
                all_text_pooled.append(outputs['text_pooled'].cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_text_pooled = np.concatenate(all_text_pooled, axis=0)
        all_labels = np.array(all_labels)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualization_paths = {}

        # 降维可视化
        if dimension_reduction and plot_embeddings:
            try:
                embeddings_2d = self._perform_dimension_reduction(all_embeddings, method=dimension_reduction)
                viz_path = self._plot_embeddings(embeddings_2d, all_labels, "Fused Embeddings", timestamp,
                                                 dimension_reduction)
                visualization_paths['fused_embeddings'] = viz_path

                text_pooled_2d = self._perform_dimension_reduction(all_text_pooled, method=dimension_reduction)
                viz_path = self._plot_embeddings(text_pooled_2d, all_labels, "Text Pooled Outputs", timestamp,
                                                 dimension_reduction)
                visualization_paths['text_pooled'] = viz_path

            except Exception as e:
                print(f"Warning: Dimension reduction failed with error: {str(e)}")

        # 注意力可视化
        attention_viz_paths = []
        if attention_visualization:
            print("\nGenerating attention visualizations...")
            viz_samples = self._select_samples_for_visualization(dataset, all_preds, all_labels, k=viz_samples)

            for sample in viz_samples:
                # 构建四元组文本
                quad_parts = []
                for aspect, opinion, category, polarity, confidence in sample['quads']:
                    quad_parts.append(
                        f"This citation expresses [MASK] sentiment towards {aspect} through {opinion} which belongs to {category} category"
                    )
                quad_text = " ; ".join(quad_parts)

                label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
                viz_path = self._visualize_attention(
                    text=sample['text'],
                    quad_text=quad_text,
                    quads=sample['quads'],  # 传入原始四元组
                    timestamp=f"{timestamp}_{label_map[sample['true_label']]}_{label_map[sample['predicted_label']]}"
                )
                if viz_path:
                    attention_viz_paths.append({
                        'path': viz_path,
                        'category': sample['category'],
                        'true_label': label_map[sample['true_label']],
                        'predicted_label': label_map[sample['predicted_label']],
                        'num_quads': len(sample['quads'])
                    })

            visualization_paths['attention'] = attention_viz_paths
            print(f"Generated {len(attention_viz_paths)} attention visualizations")

        # 基础评估指标
        report = classification_report(
            all_labels,
            all_preds,
            target_names=['neutral', 'positive', 'negative'],
            digits=4
        )

        conf_matrix = plot_confusion_matrix(all_labels, all_preds, ['neutral', 'positive', 'negative'])

        # 错误分析
        if error_analysis and error_samples:
            error_samples.sort(key=lambda x: x['loss'], reverse=True)

            label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
            for sample in error_samples:
                sample['true_label'] = label_map[sample['true_label']]
                sample['predicted_label'] = label_map[sample['predicted_label']]

            error_analysis_file = f'output/error_analysis_{timestamp}.json'
            error_stats = self._compute_error_statistics(error_samples)

            error_analysis_results = {
                'error_samples': error_samples,
                'error_statistics': error_stats,
                'model_performance': {
                    'classification_report': report,
                    'total_samples': len(all_labels),
                    'error_samples': len(error_samples),
                    'error_rate': len(error_samples) / len(all_labels)
                },
                'visualization_paths': visualization_paths
            }

            with open(error_analysis_file, 'w', encoding='utf-8') as f:
                json.dump(error_analysis_results, f, ensure_ascii=False, indent=2)

            print(f"\nError analysis saved to: {error_analysis_file}")
            print(f"Total samples: {len(all_labels)}")
            print(f"Error samples: {len(error_samples)}")
            print(f"Error rate: {len(error_samples) / len(all_labels):.2%}")
            print("\nError Statistics:")
            print(json.dumps(error_stats, indent=2))

        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'labels': all_labels,
            'losses': all_losses,
            'error_samples': error_samples if error_analysis else None,
            'visualization_paths': visualization_paths
        }

    def _compute_error_statistics(self, error_samples):
        """计算错误样本的统计信息"""
        stats = {
            'confusion_pairs': {},
            'quad_statistics': {
                'avg_quads_per_sample': 0,
                'samples_without_quads': 0,
                'category_distribution': {}
            },
            'high_confidence_errors': 0,
            'error_types': {
                'false_positive': 0,
                'false_negative': 0,
                'neutral_errors': 0
            }
        }

        total_quads = 0
        for sample in error_samples:
            error_pair = f"{sample['true_label']}->{sample['predicted_label']}"
            stats['confusion_pairs'][error_pair] = stats['confusion_pairs'].get(error_pair, 0) + 1

            quads = sample.get('quads', [])
            total_quads += len(quads)
            if not quads:
                stats['quad_statistics']['samples_without_quads'] += 1

            for quad in quads:
                category = quad[2]
                stats['quad_statistics']['category_distribution'][category] = \
                    stats['quad_statistics']['category_distribution'].get(category, 0) + 1

            pred_confidence = max(sample['confidence_scores'])
            if pred_confidence > 0.9:
                stats['high_confidence_errors'] += 1

            if sample['true_label'] == 'neutral' and sample['predicted_label'] != 'neutral':
                stats['error_types']['false_positive'] += 1
            elif sample['true_label'] != 'neutral' and sample['predicted_label'] == 'neutral':
                stats['error_types']['false_negative'] += 1
            else:
                stats['error_types']['neutral_errors'] += 1

        if error_samples:
            stats['quad_statistics']['avg_quads_per_sample'] = total_quads / len(error_samples)

        return stats


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


def train_asqp_model(args, train_data, eval_data):
    device = args.device
    config = QuadAspectEnhancedBertConfig(
        num_labels=3,
        loss_type=args.loss_type,
        loss_weights={'citation_sentiment': 1.0, 'subject_sentiment': 0.25},
        focal_alpha=0.8,
        focal_gamma=2.0,
        label_smoothing=0.0,
        multitask=True,
        backbone_model=args.model_name,
        num_categories=5 # 类别数量
    )

    from ablation_study import (SimpleConcat, WithoutCrossAttention, FourFusionModel,
                                WithoutNonLinearFusion, BaseTextOnlyModel, SingleBertEncoder,
                                TextAttentionFusion, QuadOnlyModel, CrossAttentionOnlyModel,
                                AttentionVariant)

    # 定义模型保存路径
    model_save_path = f'./finetuned_models/{args.model_name}/best_model'

    model = QuadAspectEnhancedBertModel(config).to(device)
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
        bf16=True, # bf16精度较低但是数值范围更大
        bf16_full_eval=True,
        metric_for_best_model='F1_Macro',
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = AspectAwareTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        callbacks=[LossRecorderCallback()]
    )
    trainer.train()
    trainer.save_model(model_save_path)
    return model_save_path

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 设置CUDA工作区配置

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 启用Cudnn


def main(args):
    # 加载数据
    pos_neg_quad = f'../output/sentiment_asqp_results_corpus_expand_llama405b.json'
    neu_quad = f'../output/sentiment_asqp_results_corpus_expand_llama8b_neutral.json'
    quad_file_verified = f'../output/quad_results_v1/sentiment_asqp_results_corpus_expand_verified_gpt4o.json'
    corpus_file = '../data/citation_sentiment_corpus_expand.csv'
    pos_neg_samples, neutral_samples = load_asqp_data(pos_neg_quad, neu_quad)

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
        print(f"Positive: {pos_neg_dist['positive']}")
        print(f"Negative: {pos_neg_dist['negative']}")
        print(f"Neutral: {len(split['neutral_samples'])}")

    # 初始化处理器和分词器
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")
    train_dataset, val_dataset, test_dataset = create_datasets(
        split_data,
        tokenizer,
        with_asqp=True,  # 在这进行tokenize
        method='random_words'
    )

    # 训练模型并获取最佳模型路径
    best_model_path = train_asqp_model(args, train_dataset, val_dataset)

    best_model_path = f'./finetuned_models/{args.model_name}/best_model'
    config = QuadAspectEnhancedBertConfig.from_pretrained(best_model_path)
    best_model = QuadAspectEnhancedBertModel.from_pretrained(best_model_path, config=config)
    evaluator = ModelEvaluator(best_model, tokenizer)

    # print("\nValidation Set Results:")
    # val_results = evaluator.evaluate(val_dataset)
    # print(val_results['classification_report'])

    print("\nTest Set Results with Full Analysis:")
    test_results = evaluator.evaluate(
        test_dataset,
        error_analysis=True,
        dimension_reduction='tsne',  # 或 'umap'
        plot_embeddings=True,
        attention_visualization=True,
        viz_samples=5  # 每类选择5个样本进行注意力可视化
    )
    print(test_results['classification_report'])
    

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
    parser.add_argument("--loss_type", type=str, default="focal_loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
