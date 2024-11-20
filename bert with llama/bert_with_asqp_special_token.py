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

from transformers_interpret import MultiLabelClassificationExplainer
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


class HierarchicalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )

        self.sent_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )

        self.aspect_specific_layer = nn.ModuleDict({
            'method': nn.Linear(config.hidden_size, config.hidden_size),
            'result': nn.Linear(config.hidden_size, config.hidden_size),
            'contribution': nn.Linear(config.hidden_size, config.hidden_size)
        })

    def forward(self, text_hidden, quad_hidden, attention_mask, quad_attention_mask):
        # Token-level attention
        token_attn_output, _ = self.token_attention(
            query=text_hidden,
            key=quad_hidden,
            value=quad_hidden,
            key_padding_mask=~quad_attention_mask.bool()
        )

        # 根据特殊token分离不同视角的特征
        method_mask = (quad_hidden == self.token_to_id('[METHOD]')).float()
        result_mask = (quad_hidden == self.token_to_id('[RESULT]')).float()
        contribution_mask = (quad_hidden == self.token_to_id('[CONTRIBUTION]')).float()

        # Aspect-specific representation
        method_repr = self.aspect_specific_layer['method'](token_attn_output * method_mask)
        result_repr = self.aspect_specific_layer['result'](token_attn_output * result_mask)
        contribution_repr = self.aspect_specific_layer['contribution'](token_attn_output * contribution_mask)

        # Sentence-level attention
        combined_repr = torch.stack([method_repr, result_repr, contribution_repr], dim=1)
        sent_attn_output, _ = self.sent_attention(
            query=combined_repr,
            key=combined_repr,
            value=combined_repr
        )

        return sent_attn_output

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
        # 添加aspect-aware gates
        self.aspect_gates = nn.ModuleDict({
            'method': nn.Linear(hidden_size, hidden_size),
            'result': nn.Linear(hidden_size, hidden_size),
            'contribution': nn.Linear(hidden_size, hidden_size)
        })

        # 添加表达方式aware gates
        self.expression_gates = nn.ModuleDict({
            'explicit': nn.Linear(hidden_size, hidden_size),
            'implicit': nn.Linear(hidden_size, hidden_size)
        })

    def forward(self, text_features, quad_features, hier_attn_features):
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

        # 添加aspect-aware fusion
        aspect_weights = {
            aspect: torch.sigmoid(gate(hier_attn_features))
            for aspect, gate in self.aspect_gates.items()
        }

        aspect_fusion = sum(weight * features
                            for weight, features in zip(aspect_weights.values(),
                                                        [text_features, quad_features, hier_attn_features]))

        # 最终融合
        final_fusion = self.fusion_layer(
            torch.cat([
                fused,  # 原有的融合结果
                aspect_fusion  # 新增的aspect-aware融合结果
            ], dim=-1)
        )

        return final_fusion


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
        # 替换原有的attention层为hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(config)

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

        # 使用分层注意力
        hier_attn_output = self.hierarchical_attention(
            text_hidden,
            quad_hidden,
            attention_mask,
            quad_attention_mask
        )

        # 使用增强的融合层
        fused = self.fusion(
            text_pooled,
            quad_pooled,
            hier_attn_output
        )
        
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
    def __init__(self, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.category2id = {  # 预定义类别映射
            'METHODOLOGY': 0,
            'PERFORMANCE': 1,
            'INNOVATION': 2,
            'APPLICABILITY': 3,
            'LIMITATION': 4,
        }

    def build_category_vocab(self, features):
        """使用预定义的类别词表"""
        return len(self.category2id)

    def process_features(self, features: Dict[str, Any], method='mask') -> Dict[str, Any]:
        """处理四元组特征"""
        texts = [f.get('text', '') for f in features]
        labels = [f.get('label', 0) for f in features]  # 使用get方法安全获取标签
        sentiment_labels = [1 if label > 0 else 0 for label in labels]

        text_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Collect all positive and negative quadruples
        pos_neg_quads = []
        for feature in features:
            if feature.get('label', 0) != 0:  # Positive or Negative samples
                quads = feature.get('quads', [])
                pos_neg_quads.extend(quads)

        # 处理四元组
        quad_texts = []
        category_ids = []
        for feature in features:
            if feature.get('label', 0) == 0:  # Neutral sample
                if method == 'empty':
                    quad_text = ''
                    category_id = 0
                elif method == 'random_mask':
                    # 使用[MASK]符号和随机噪声填充中性样本的三元组
                    noise_tokens = [
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
                    num_neutral_tags = random.randint(1, 2)  # 随机选择1到2个中性标记
                    num_words = random.randint(2, 5)  # 随机选择2到5个中性词

                    neutral_tags = ["[MASK]"] * num_neutral_tags
                    random_words = random.sample(noise_tokens, num_words)

                    combined_list = neutral_tags + random_words
                    random.shuffle(combined_list)
                    if len(combined_list) < 5:  # 截取或者填充
                        combined_list += ["[MASK]"] * (5 - len(combined_list))
                    quad_text = " ".join(combined_list[:5])
                    category_id = 0
                elif method == 'random_polar':
                    # Randomly select a quadruple from positive and negative samples
                    if pos_neg_quads:
                        aspect, opinion, category, polarity, confidence = random.choice(pos_neg_quads)
                        quad_text = f"This citation expresses [MASK] sentiment towards {aspect} through {opinion} which belongs to {category} category"
                        # quad_text = f"This citation expresses [MASK] sentiment towards [ASPECT] through [OPINION] which belongs to [CATEGORY] category"
                        # quad_text = f"{aspect} is {opinion} in [[MASK]] {category}"
                        category_id = self.category2id.get(category, 0)
                    else:
                        quad_text = ""
                        category_id = 0
                elif method == 'random_words':
                    # 下面代码是随机在中性文本中选择四元组
                    text_tokens = feature.get('text', '').split()
                    if len(text_tokens) >= 2:
                        aspect = random.choice(text_tokens)
                        opinion = random.choice(text_tokens)
                    else:
                        aspect = 'aspect'
                        opinion = 'opinion'
                    category = random.choice(list(self.category2id.keys()))
                    quad_text = f"This citation expresses [MASK] sentiment towards {aspect} through {opinion} which belongs to {category} category"
                    category_id = self.category2id.get(category, 0)
            else:
                quads = feature.get('quads', [])
                quad_parts = []
                feature_categories = []
                for aspect, opinion, category, polarity, confidence in quads:
                    # 判断评价视角
                    view_token = self._get_view_token(aspect, category)
                    # 判断表达方式
                    expr_token = self._get_expression_token(opinion)

                    quad_parts.append(
                        f"{view_token} {expr_token} "
                        f"[ASP]{aspect}[/ASP] "
                        f"[OPN]{opinion}[/OPN] "
                        f"[CAT]{category}[/CAT]"
                    )
                    feature_categories.append(category)
                quad_text = " ; ".join(quad_parts)
            
            quad_texts.append(quad_text)
            category_ids.append(category_id)

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

    def _get_view_token(self, aspect: str, category: str) -> str:
        """根据aspect和category判断评价视角"""
        if category == 'METHODOLOGY':
            return '[METHOD]'
        elif category == 'PERFORMANCE':
            return '[RESULT]'
        elif category == 'INNOVATION':
            return '[CONTRIBUTION]'
        return '[GENERAL]'

    def _get_expression_token(self, opinion: str) -> str:
        """根据opinion判断表达方式"""
        # 这里可以构建一个显式/隐式表达的词典
        explicit_words = {'improve', 'outperform', 'better', 'worse', ...}
        return '[EXPLICIT]' if any(word in opinion.lower() for word in explicit_words) else '[IMPLICIT]'


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
                'quads': item['quads']
            })

        # Process neutral samples
        for text in neutral_samples:
            self.samples.append({
                'text': text,
                'label': 0,
                'quads': []
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

def create_datasets(split_data: Dict, tokenizer, with_asqp: bool = True) -> Tuple[
    AspectAwareDataset, AspectAwareDataset, AspectAwareDataset]:
    """
    创建训练集、验证集和测试集的Dataset对象

    Args:
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
        method = 'random_words'

    train_dataset = AspectAwareDataset(
        pos_neg_samples=split_data['train']['pos_neg_samples'],
        neutral_samples=split_data['train']['neutral_samples'],
        tokenizer=tokenizer,
        processor=processor,
        method=method
    )

    val_dataset = AspectAwareDataset(
        pos_neg_samples=val_pos_neg,
        neutral_samples=split_data['val']['neutral_samples'],
        tokenizer=tokenizer,
        processor=processor,
        method=method
    )

    test_dataset = AspectAwareDataset(
        pos_neg_samples=test_pos_neg,
        neutral_samples=split_data['test']['neutral_samples'],
        tokenizer=tokenizer,
        processor=processor,
        method=method
    )

    return train_dataset, val_dataset, test_dataset


def load_asqp_data(quad_file=None, corpus_file=None):
    """加载四元组数据"""
    pos_neg_samples = []
    with open(quad_file, 'r', encoding='utf-8') as f:
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
    df = pd.read_csv(corpus_file)
    # neutral_samples = df[df['Sentiment'] == 'o']['Citation_Text'].tolist() # 是Athar数据集的中性样本
    neutral_samples = df[df['Sentiment'] == 'Neutral']['Text'].tolist()

    return pos_neg_samples, neutral_samples

def load_asqp_data_verified(quad_file=None, corpus_file=None):
    """加载四元组数据"""
    pos_neg_samples = []
    with open(quad_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            pos_neg_samples.append({
                'text': item['text'],
                'overall_sentiment': item['overall_sentiment'],
                'quads': item['final_quadruples']  # 修改为sentiment_quadruples
            })

    # 加载中性样本
    neutral_samples = []
    df = pd.read_csv(corpus_file)
    neutral_samples = df[df['Sentiment'] == 'Neutral']['Text'].tolist()

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
        all_embeddings = []  # To store embeddings
        all_text_pooled = []

        for batch in dataloader:
            # 将数据移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            preds = torch.argmax(outputs['logits'], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_embeddings.append(outputs['embeddings'].cpu().numpy())
            all_text_pooled.append(outputs['text_pooled'].cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_text_pooled = np.concatenate(all_text_pooled, axis=0)
        all_labels = np.array(all_labels)

        # # Perform dimensionality reduction
        # from sklearn.manifold import TSNE
        # import umap.umap_ as umap
        #
        # # Choose UMAP or t-SNE
        # # reducer = umap.UMAP(n_components=2, random_state=42)
        # reducer = TSNE(n_components=2, random_state=42)
        # embeddings_2d = reducer.fit_transform(all_embeddings)
        # text_pooled_2d = reducer.fit_transform(all_text_pooled)
        #
        # # Plot the embeddings side by side
        # fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        # scatter1 = ax[0].scatter(
        #     embeddings_2d[:, 0],
        #     embeddings_2d[:, 1],
        #     c=all_labels,
        #     cmap='viridis',
        #     alpha=0.7
        # )
        # ax[0].set_title('Embeddings Before Classification')
        # ax[0].set_xlabel('Dimension 1')
        # ax[0].set_ylabel('Dimension 2')
        #
        # scatter2 = ax[1].scatter(
        #     text_pooled_2d[:, 0],
        #     text_pooled_2d[:, 1],
        #     c=all_labels,
        #     cmap='viridis',
        #     alpha=0.7
        # )
        # ax[1].set_title('Text Pooled Outputs')
        # ax[1].set_xlabel('Dimension 1')
        # ax[1].set_ylabel('Dimension 2')
        #
        # # Add legends
        # handles, labels = scatter1.legend_elements()
        # ax[0].legend(handles, ['neutral', 'positive', 'negative'])
        # ax[1].legend(handles, ['neutral', 'positive', 'negative'])
        #
        # plt.tight_layout()
        # plt.show()

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


def train_asqp_model(args, train_data, eval_data):
    device = args.device
    config = QuadAspectEnhancedBertConfig(
        num_labels=3,
        loss_type=args.loss_type,
        loss_weights={'citation_sentiment': 1.0, 'subject_sentiment': 0.25},
        focal_alpha=0.8,
        focal_gamma=2.0,
        label_smoothing=0.0,
        multitask=False,
        backbone_model=args.model_name,
        num_categories=5 # 类别数量
    )

    from ablation_study import (SimpleConcat, WithoutCrossAttention,
                                WithoutNonLinearFusion, BaseTextOnlyModel, SingleBertEncoder,
                                TextAttentionFusion, QuadOnlyModel, CrossAttentionOnlyModel)

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
        metric_for_best_model='F1_Macro',
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=True,
        dataloader_num_workers=0,
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
    return model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 启用Cudnn


def main(args):
    # 加载数据
    quad_file = f'../output/sentiment_asqp_results_corpus_expand_llama.json'
    quad_file_verified = f'../output/quad_results_v1/sentiment_asqp_results_corpus_expand_verified_gpt4o.json'
    corpus_file = '../data/citation_sentiment_corpus_expand.csv'
    # pos_neg_samples, neutral_samples = load_asqp_data_verified(quad_file_verified, corpus_file)
    pos_neg_samples, neutral_samples = load_asqp_data(quad_file, corpus_file)

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
        with_asqp=True  # 在这进行tokenize
    )

    model = train_asqp_model(args, train_dataset, val_dataset)

    # 评估模型
    evaluator = ModelEvaluator(model, tokenizer)
    
    # print("\nValidation Set Results:")
    # val_results = evaluator.evaluate(val_dataset)
    # print(val_results['classification_report'])

    print("\nTest Set Results:")
    test_results = evaluator.evaluate(test_dataset)
    print(test_results['classification_report'])
    
    # # Add visualization code
    # # 只选择积极或消极的样本(label为1或2的样本)
    # pos_neg_samples = [i for i, sample in enumerate(test_dataset.samples) if sample['label'] in [1, 2]]
    # if pos_neg_samples:
    #     random_idx = random.choice(pos_neg_samples)
    #     sample = test_dataset.samples[random_idx]
    #
    #     text = sample['text']
    #     quads = sample['quads']
    #     sentiment = "positive" if sample['label'] == 1 else "negative"
    #
    #     # 构建四元组文本
    #     quad_parts = []
    #     for aspect, opinion, category, polarity, confidence in quads:
    #         quad_parts.append(f"This citation expresses [MASK] sentiment towards {aspect} through {opinion} which belongs to {category} category")
    #     quad_text = " ; ".join(quad_parts)
    #
    #     print("\nVisualization Sample:")
    #     print(f"Text: {text}")
    #     print(f"Sentiment: {sentiment}")
    #     print(f"Quads: {quad_text}")
    #
    #     # 生成注意力可视化图
    #     fig = visualize_model_attention(
    #         model=model,
    #         tokenizer=tokenizer,
    #         text=text,
    #         quad_text=quad_text
    #     )
    #
    #     # 保存可视化结果
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     fig.savefig(f'attention_visualization_{sentiment}_{timestamp}.png',
    #                 bbox_inches='tight',
    #                 dpi=300)
    #     plt.close(fig)
    # else:
    #     print("No positive/negative samples found in test set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model_name", type=str, default="scibert-scivocab-uncased")
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
    main(args)
