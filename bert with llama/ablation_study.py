from typing import Optional
import torch.nn.functional as F
import torch
import torch.nn as nn
from modelscope import AutoConfig
from transformers import PreTrainedModel, AutoModel, PretrainedConfig


class AdditiveAttentionFusion(nn.Module):
    """加性注意力融合
    使用加性注意力机制来动态融合不同特征
    """

    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.Dropout(dropout_prob)
        )

    def forward(self, text_features, quad_features, attn_features):
        # 堆叠特征
        features = torch.stack([text_features, quad_features, attn_features], dim=1)
        # 计算注意力权重
        weights = self.attention(features)
        weights = F.softmax(weights, dim=1)
        # 加权融合
        weighted_sum = torch.sum(weights * features, dim=1)
        return self.output_layer(weighted_sum)


class BilinearFusion(nn.Module):
    """双线性特征融合
    使用双线性变换来建模特征间的高阶交互
    """

    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.bilinear1 = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.bilinear2 = nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, text_features, quad_features, attn_features):
        # 双线性交互
        interaction1 = self.bilinear1(text_features, quad_features)
        interaction2 = self.bilinear2(interaction1, attn_features)
        # 连接所有特征
        fused = torch.cat([interaction2, text_features, quad_features], dim=-1)
        return self.fusion_layer(fused)


class HierarchicalFusion(nn.Module):
    """层次化特征融合
    通过多层次的方式逐步融合不同特征
    """

    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.level1_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.level2_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, text_features, quad_features, attn_features):
        # 第一层融合
        level1 = self.level1_fusion(torch.cat([text_features, quad_features], dim=-1))
        # 第二层融合
        level2 = self.level2_fusion(torch.cat([level1, attn_features], dim=-1))
        return level2


class CrossModalFusion(nn.Module):
    """跨模态特征融合
    使用交叉注意力和门控机制进行特征融合
    """

    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_prob,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 3),
            nn.Sigmoid()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, text_features, quad_features, attn_features):
        # 准备查询、键和值
        features = torch.stack([text_features, quad_features, attn_features], dim=1)
        # 交叉注意力
        attended_features, _ = self.cross_attention(features, features, features)
        # 计算门控权重
        gates = self.gate(attended_features.reshape(attended_features.size(0), -1))
        # 应用门控并输出
        gated_features = gates * attended_features.reshape(attended_features.size(0), -1)
        return self.output_layer(gated_features)


class ResidualFusion(nn.Module):
    """残差特征融合
    使用残差连接来保持原始特征信息
    """

    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 3, hidden_size * 3)
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 6, hidden_size * 3),
            nn.Sigmoid()
        )

    def forward(self, text_features, quad_features, attn_features):
        # 连接原始特征
        original = torch.cat([text_features, quad_features, attn_features], dim=-1)
        # 特征变换
        transformed = self.transform(original)
        # 计算融合门控
        gate = self.fusion_gate(torch.cat([original, transformed], dim=-1))
        # 残差连接
        return gate * transformed + (1 - gate) * original

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
            num_categories: int = 6,  # 添加类别数量
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


# 1. 简单拼接版本
class SimpleConcat(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        # 基础编码器
        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.quad_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        hidden_size = config.hidden_size

        # 简单的线性分类器
        self.classifier = nn.Linear(hidden_size * 2, config.num_labels)
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
        text_pooled = text_outputs.pooler_output

        # 四元组编码
        quad_outputs = self.quad_bert(quad_input_ids, quad_attention_mask)
        quad_pooled = quad_outputs.pooler_output

        # 简单拼接
        concatenated = torch.cat([text_pooled, quad_pooled], dim=-1)

        # 分类预测
        logits = self.classifier(concatenated)

        # 情感检测
        sentiment_logits = self.sentiment_detector(text_pooled)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        # 概率加权
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': concatenated,
            'text_pooled': text_pooled
        }


# 2. 移除Cross-attention版本
class WithoutCrossAttention(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.quad_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        hidden_size = config.hidden_size

        # 类别感知层
        self.category_embedding = nn.Embedding(config.num_categories, hidden_size)

        # 只保留自注意力层
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

        self.classifier = nn.Linear(hidden_size * 3, config.num_labels)
        self.sentiment_detector = nn.Linear(hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, quad_input_ids=None,
                quad_attention_mask=None, category_ids=None, labels=None, sentiment_labels=None):
        text_outputs = self.bert(input_ids, attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output

        quad_outputs = self.quad_bert(quad_input_ids, quad_attention_mask)
        quad_hidden = quad_outputs.last_hidden_state
        quad_pooled = quad_outputs.pooler_output

        if category_ids is not None:
            category_embeds = self.category_embedding(category_ids)
            quad_hidden = quad_hidden + category_embeds.unsqueeze(1)

        text_transformed = self.text_transform(text_hidden)
        quad_transformed = self.quad_transform(quad_hidden)

        # 直接对文本做自注意力
        self_attn_output, _ = self.self_attention(
            query=text_transformed,
            key=text_transformed,
            value=text_transformed,
            key_padding_mask=~attention_mask.bool()
        )
        attn_pooled = torch.mean(self_attn_output, dim=1)

        fused = self.fusion(text_pooled, quad_pooled, attn_pooled)

        logits = self.classifier(fused)
        sentiment_logits = self.sentiment_detector(text_pooled)

        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': fused,
            'text_pooled': text_pooled
        }


# 3. 移除非线性融合层版本(使用简单的线性层替代)
class WithoutNonLinearFusion(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.quad_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        hidden_size = config.hidden_size

        self.category_embedding = nn.Embedding(config.num_categories, hidden_size)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )

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

        # 简单的线性融合层
        self.simple_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.classifier = nn.Linear(hidden_size * 3, config.num_labels)
        self.sentiment_detector = nn.Linear(hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, quad_input_ids=None,
                quad_attention_mask=None, category_ids=None, labels=None, sentiment_labels=None):
        text_outputs = self.bert(input_ids, attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output

        quad_outputs = self.quad_bert(quad_input_ids, quad_attention_mask)
        quad_hidden = quad_outputs.last_hidden_state
        quad_pooled = quad_outputs.pooler_output

        if category_ids is not None:
            category_embeds = self.category_embedding(category_ids)
            quad_hidden = quad_hidden + category_embeds.unsqueeze(1)

        text_transformed = self.text_transform(text_hidden)
        quad_transformed = self.quad_transform(quad_hidden)

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

        # 简单拼接并通过线性层
        combined = torch.cat([text_pooled, quad_pooled, attn_pooled], dim=-1)
        fused = self.simple_fusion(combined)

        logits = self.classifier(fused)
        sentiment_logits = self.sentiment_detector(text_pooled)

        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': fused,
            'text_pooled': text_pooled
        }


# 1. Single BERT Encoder Version
class SingleBertEncoder(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        # 使用单个BERT编码器
        self.shared_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
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
        # 自注意力层
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
        # 共享BERT编码文本和四元组
        text_outputs = self.shared_bert(input_ids, attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output

        quad_outputs = self.shared_bert(quad_input_ids, quad_attention_mask)
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

        # 情感检测和概率加权
        sentiment_logits = self.sentiment_detector(text_pooled)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        # 使用概率加权
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': fused,
            'text_pooled': text_pooled
        }


# 2. Base Model (Text Only)
class BaseTextOnlyModel(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        # 只使用一个BERT编码器处理文本
        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        hidden_size = config.hidden_size

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

        # 分类器 - 直接使用BERT输出
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        # 情感检测器
        self.sentiment_detector = nn.Linear(hidden_size, 2)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            quad_input_ids=None,  # 不使用四元组输入
            quad_attention_mask=None,
            category_ids=None,  # 不使用类别信息
            labels=None,
            sentiment_labels=None
    ):
        # 文本编码
        text_outputs = self.bert(input_ids, attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output

        # 特征转换
        text_transformed = self.text_transform(text_hidden)

        # 自注意力增强
        self_attn_output, _ = self.self_attention(
            query=text_transformed,
            key=text_transformed,
            value=text_transformed,
            key_padding_mask=~attention_mask.bool()
        )
        attn_pooled = torch.mean(self_attn_output, dim=1)

        # 分类预测 - 使用注意力增强的特征
        logits = self.classifier(text_pooled)

        # 情感检测和概率加权
        sentiment_logits = self.sentiment_detector(text_pooled)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        # 使用概率加权
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': attn_pooled,
            'text_pooled': text_pooled
        }

# 1. Text-Attention Fusion Only (Without Quad Fusion)
class TextAttentionFusion(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.quad_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        hidden_size = config.hidden_size

        self.category_embedding = nn.Embedding(config.num_categories, hidden_size)

        # 交互注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
        )
        # 自注意力层
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

        # 只融合text和attention的融合层
        class TextAttentionFusionLayer(nn.Module):
            def __init__(self, hidden_size, dropout_prob):
                super().__init__()

                # 门控机制
                self.gate_text = nn.Linear(hidden_size, hidden_size)
                self.gate_attn = nn.Linear(hidden_size, hidden_size)

                # 特征转换
                self.transform_text = nn.Linear(hidden_size, hidden_size)
                self.transform_attn = nn.Linear(hidden_size, hidden_size)

                # Layer Normalization
                self.layer_norm = nn.LayerNorm(hidden_size * 2)

                # 非线性变换
                self.fusion_layer = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size * 3),
                    nn.LayerNorm(hidden_size * 3),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_size * 3, hidden_size * 2),
                    nn.LayerNorm(hidden_size * 2)
                )

            def forward(self, text_features, attn_features):
                # 计算门控权重
                text_gate = torch.sigmoid(self.gate_text(text_features))
                attn_gate = torch.sigmoid(self.gate_attn(attn_features))

                # 特征转换
                text_transformed = self.transform_text(text_features)
                attn_transformed = self.transform_attn(attn_features)

                # 应用门控机制
                text_gated = text_gate * text_transformed
                attn_gated = attn_gate * attn_transformed

                # 特征拼接
                combined = torch.cat([text_gated, attn_gated], dim=-1)
                normalized = self.layer_norm(combined)
                fused = self.fusion_layer(normalized)

                return fused

        self.fusion = TextAttentionFusionLayer(hidden_size, config.hidden_dropout_prob)

        # 分类器 - 因为只融合了两个特征，所以输入维度要相应调整
        self.classifier = nn.Linear(hidden_size * 2, config.num_labels)
        self.sentiment_detector = nn.Linear(hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, quad_input_ids=None,
                quad_attention_mask=None, category_ids=None, labels=None, sentiment_labels=None):
        text_outputs = self.bert(input_ids, attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output

        quad_outputs = self.quad_bert(quad_input_ids, quad_attention_mask)
        quad_hidden = quad_outputs.last_hidden_state

        if category_ids is not None:
            category_embeds = self.category_embedding(category_ids)
            quad_hidden = quad_hidden + category_embeds.unsqueeze(1)

        text_transformed = self.text_transform(text_hidden)
        quad_transformed = self.quad_transform(quad_hidden)

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

        # 只融合text和attention特征
        fused = self.fusion(text_pooled, attn_pooled)

        logits = self.classifier(fused)
        sentiment_logits = self.sentiment_detector(text_pooled)

        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': fused,
            'text_pooled': text_pooled
        }


# 2. Quad Only Model
class QuadOnlyModel(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        # 只使用处理四元组的BERT
        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.quad_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        hidden_size = config.hidden_size

        # 类别感知层
        self.category_embedding = nn.Embedding(config.num_categories, hidden_size)

        # 自注意力层用于四元组特征增强
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )

        # 四元组特征转换层
        self.quad_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        # 分类器 - 直接使用四元组特征
        self.sentiment_detector = nn.Linear(hidden_size, 2)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, quad_input_ids=None,
                quad_attention_mask=None, category_ids=None, labels=None, sentiment_labels=None):
        # 只处理四元组
        text_outputs = self.bert(input_ids, attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output

        quad_outputs = self.quad_bert(quad_input_ids, quad_attention_mask)
        quad_hidden = quad_outputs.last_hidden_state
        quad_pooled = quad_outputs.pooler_output

        # 类别增强
        if category_ids is not None:
            category_embeds = self.category_embedding(category_ids)
            quad_hidden = quad_hidden + category_embeds.unsqueeze(1)

        # 特征转换
        quad_transformed = self.quad_transform(quad_hidden)

        # 自注意力增强四元组特征
        self_attn_output, _ = self.self_attention(
            query=quad_transformed,
            key=quad_transformed,
            value=quad_transformed,
            key_padding_mask=~quad_attention_mask.bool()
        )

        # 使用平均池化得到最终特征
        final_hidden = torch.mean(self_attn_output, dim=1)

        # 分类预测
        logits = self.classifier(final_hidden)

        # 情感检测和概率加权
        sentiment_logits = self.sentiment_detector(text_pooled)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        # 使用概率加权
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': final_hidden,
            'text_pooled': quad_pooled
        }



class CrossAttentionOnlyModel(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config):
        super().__init__(config)

        # 基础编码器
        self.bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.quad_bert = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        hidden_size = config.hidden_size

        # 类别感知层
        self.category_embedding = nn.Embedding(config.num_categories, hidden_size)

        # 只保留交互注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
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

        # 分类器 - 使用交叉注意力的输出
        self.classifier = nn.Linear(hidden_size, config.num_labels)
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

        # 使用平均池化获取最终特征
        cross_pooled = torch.mean(cross_attn_output, dim=1)

        # 分类预测
        logits = self.classifier(cross_pooled)

        # 情感检测和概率加权
        sentiment_logits = self.sentiment_detector(text_pooled)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        # 使用概率加权
        neutral_weight = sentiment_probs[:, 0].unsqueeze(-1)
        subjective_weight = sentiment_probs[:, 1].unsqueeze(-1)
        logits_probs = torch.softmax(logits, dim=-1)
        refined_logits = (neutral_weight * torch.tensor([[1.0, 0.0, 0.0]], device=logits.device).expand_as(logits) +
                          subjective_weight * logits_probs)
        refined_logits = torch.log(refined_logits + 1e-10)

        return {
            'logits': refined_logits if self.config.multitask else logits,
            'sentiment_logits': sentiment_logits,
            'embeddings': cross_pooled,
            'text_pooled': text_pooled
        }


class FourFusionModel(PreTrainedModel):
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

        # 添加一个新的交互注意��层
        self.reverse_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
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
        self.classifier = nn.Linear(hidden_size * 2, config.num_labels)
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

        # 交互注意力：文本作为查询，四元组作为键和值
        cross_attn_output, _ = self.cross_attention(
            query=text_transformed,
            key=quad_transformed,
            value=quad_transformed,
            key_padding_mask=~quad_attention_mask.bool()
        )

        # 交互注意力：四元组作为查询，文本作为键和值
        reverse_cross_attn_output, _ = self.reverse_cross_attention(
            query=quad_transformed,
            key=text_transformed,
            value=text_transformed,
            key_padding_mask=~attention_mask.bool()
        )

        # 自注意力层
        self_attn_output, _ = self.self_attention(
            query=cross_attn_output,
            key=cross_attn_output,
            value=cross_attn_output,
            key_padding_mask=~attention_mask.bool()
        )

        # 池化操作
        attn_pooled = torch.mean(self_attn_output, dim=1)
        reverse_attn_pooled = torch.mean(reverse_cross_attn_output, dim=1)

        # 特征融合（现在包含四个输入）
        fused = self.fusion(text_pooled, quad_pooled, attn_pooled, reverse_attn_pooled)

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
            'logits': refined_logits if self.config.multitask else logits,  # 这个指标才是影响混淆矩阵最左边一列的数据
            'sentiment_logits': sentiment_logits,
            'embeddings': fused,  # Return embeddings before classification
            'text_pooled': text_pooled  # Return text_pooled outputs
        }

class AttentionVariant(PreTrainedModel):

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

        # 变体1：反转query和key/value的角色
        def variant1_attention(self):
            # quad作为query: [batch_size, quad_len, hidden_size]
            cross_attn_output, _ = self.cross_attention(
                query=quad_transformed,
                key=text_transformed,
                value=text_transformed,
                key_padding_mask=~attention_mask.bool()
            )  # 输出: [batch_size, quad_len, hidden_size]

            self_attn_output, _ = self.self_attention(
                query=cross_attn_output,
                key=cross_attn_output,
                value=cross_attn_output,
                key_padding_mask=~quad_attention_mask.bool()
            )  # 输出: [batch_size, quad_len, hidden_size]

            # 确保输出维度与原始模型一致
            attn_pooled = torch.mean(self_attn_output, dim=1)  # [batch_size, hidden_size]
            return attn_pooled

        # 变体3：双向注意力
        def variant3_attention(self):
            # Text to Quad attention
            text2quad_output, _ = self.cross_attention(
                query=text_transformed,
                key=quad_transformed,
                value=quad_transformed,
                key_padding_mask=~quad_attention_mask.bool()
            )  # [batch_size, text_len, hidden_size]

            # Quad to Text attention
            quad2text_output, _ = self.cross_attention(
                query=quad_transformed,
                key=text_transformed,
                value=text_transformed,
                key_padding_mask=~attention_mask.bool()
            )  # [batch_size, quad_len, hidden_size]

            # 分别池化后再组合
            text2quad_pooled = torch.mean(text2quad_output, dim=1)  # [batch_size, hidden_size]
            quad2text_pooled = torch.mean(quad2text_output, dim=1)  # [batch_size, hidden_size]

            # 合并双向注意力的结果
            attn_pooled = (text2quad_pooled + quad2text_pooled) / 2  # [batch_size, hidden_size]
            return attn_pooled

        # 变体5：双向注意力 + 自注意力
        def variant5_bidirectional_self_attention(self):
            """双向注意力后接自注意力机制
            1. 先计算双向的交叉注意力
            2. 融合双向注意力的结果
            3. 对融合后的特征进行自注意力处理
            """
            # Text to Quad attention
            text2quad_output, _ = self.cross_attention(
                query=text_transformed,
                key=quad_transformed,
                value=quad_transformed,
                key_padding_mask=~quad_attention_mask.bool()
            )  # [batch_size, text_len, hidden_size]

            # Quad to Text attention
            quad2text_output, _ = self.cross_attention(
                query=quad_transformed,
                key=text_transformed,
                value=text_transformed,
                key_padding_mask=~attention_mask.bool()
            )  # [batch_size, quad_len, hidden_size]

            # 融合双向注意力的结果
            # 将两个序列长度对齐（通过平均池化）
            text2quad_avg = torch.mean(text2quad_output, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
            quad2text_avg = torch.mean(quad2text_output, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]

            # 拼接双向注意力的结果
            combined_features = torch.cat([text2quad_avg, quad2text_avg], dim=1)  # [batch_size, 2, hidden_size]

            # 应用自注意力机制
            self_attn_output, _ = self.self_attention(
                query=combined_features,
                key=combined_features,
                value=combined_features
            )  # [batch_size, 2, hidden_size]

            # 最终池化
            attn_pooled = torch.mean(self_attn_output, dim=1)  # [batch_size, hidden_size]

            return attn_pooled

        # 变体6：级联注意力 + 自注意力
        def variant6_cascaded_self_attention(self):
            """级联注意力后接自注意力机制
            1. 先进行两阶段的级联注意力
            2. 保留中间状态和最终状态
            3. 对所有状态进行自注意力处理
            """
            # First cross-attention
            text2quad_output, _ = self.cross_attention(
                query=text_transformed,
                key=quad_transformed,
                value=quad_transformed,
                key_padding_mask=~quad_attention_mask.bool()
            )  # [batch_size, text_len, hidden_size]

            # Second cross-attention
            cascaded_output, _ = self.cross_attention(
                query=quad_transformed,
                key=text2quad_output,
                value=text2quad_output,
                key_padding_mask=~attention_mask.bool()
            )  # [batch_size, quad_len, hidden_size]

            # 收集所有中间状态
            # 将所有序列长度统一（通过平均池化）
            text2quad_avg = torch.mean(text2quad_output, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
            cascaded_avg = torch.mean(cascaded_output, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
            original_avg = torch.mean(text_transformed, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]

            # 拼接所有状态
            all_states = torch.cat([
                original_avg,  # 原始特征
                text2quad_avg,  # 第一阶段注意力
                cascaded_avg  # 第二阶段注意力
            ], dim=1)  # [batch_size, 3, hidden_size]

            # 应用自注意力机制整合所有状态
            self_attn_output, _ = self.self_attention(
                query=all_states,
                key=all_states,
                value=all_states
            )  # [batch_size, 3, hidden_size]

            # 最终池化
            attn_pooled = torch.mean(self_attn_output, dim=1)  # [batch_size, hidden_size]

            return attn_pooled

        # 变体4：级联注意力
        def variant4_attention(self):
            # First cross-attention
            text2quad_output, _ = self.cross_attention(
                query=text_transformed,
                key=quad_transformed,
                value=quad_transformed,
                key_padding_mask=~quad_attention_mask.bool()
            )  # [batch_size, text_len, hidden_size]

            # Second cross-attention
            final_output, _ = self.cross_attention(
                query=quad_transformed,
                key=text2quad_output,
                value=text2quad_output,
                key_padding_mask=~attention_mask.bool()
            )  # [batch_size, quad_len, hidden_size]

            attn_pooled = torch.mean(final_output, dim=1)  # [batch_size, hidden_size]
            return attn_pooled

        # 选择使用哪个变体（
        attn_pooled = variant3_attention(self)  # [batch_size, hidden_size]

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
            'logits': refined_logits if self.config.multitask else logits,  # 这个指标才是影响混淆矩阵最左边一列的数据
            'sentiment_logits': sentiment_logits,
            'embeddings': fused,  # Return embeddings before classification
            'text_pooled': text_pooled  # Return text_pooled outputs
        }
