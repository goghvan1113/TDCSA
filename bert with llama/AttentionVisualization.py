import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class AttentionVisualization(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        # 判断模型类型
        self.model_type = self._get_model_type()
        self.attention_scores = {
            'bert_text': None,
            'bert_quad': None,
            'text_transform': None,
            'quad_transform': None,
            'cross_attention': None,
            'self_attention': None,
            'fusion': None
        }

    def _get_model_type(self):
        """根据模型配置确定模型类型"""
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'model_type'):
                return self.model.config.model_type
            # 对于自定义模型，检查backbone_model配置
            elif hasattr(self.model.config, 'backbone_model'):
                if 'roberta' in self.model.config.backbone_model.lower():
                    return 'roberta'
                elif 'bert' in self.model.config.backbone_model.lower():
                    return 'bert'
        # 默认返回bert
        return 'bert'

    def get_special_tokens(self):
        """根据模型类型返回特殊token的表示"""
        if self.model_type == 'roberta':
            return {
                'cls_token': '<s>',
                'sep_token': '</s>',
                'cls_display': '[CLS]',  # 显示用
                'sep_display': '[SEP]',  # 显示用
                'pad_token': '<pad>',
                'mask_token': '<mask>',
            }
        else:  # bert
            return {
                'cls_token': '[CLS]',
                'sep_token': '[SEP]',
                'cls_display': '[CLS]',
                'sep_display': '[SEP]',
                'pad_token': '[PAD]',
                'mask_token': '[MASK]',
            }

    def get_attention_weights(self, layer_outputs):
        if hasattr(layer_outputs, 'attentions'):
            return layer_outputs.attentions[-1]
        return None

    def decode_tokens(self, input_ids):
        """将input_ids解码为更易读的形式，处理不同模型的特殊token"""
        # 获取原始token
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        special_tokens = self.get_special_tokens()

        readable_tokens = []
        for token in tokens:
            # 处理特殊token
            if token == special_tokens['cls_token']:
                readable_tokens.append(special_tokens['cls_display'])
            elif token == special_tokens['sep_token']:
                readable_tokens.append(special_tokens['sep_display'])
            elif token == special_tokens['pad_token']:
                readable_tokens.append('[PAD]')
            elif token == special_tokens['mask_token']:
                readable_tokens.append('[MASK]')
            else:
                # 处理RoBERTa的子词前缀
                if self.model_type == 'roberta':
                    if token.startswith('Ġ'):  # RoBERTa的子词前缀
                        token = token[1:]  # 移除'Ġ'前缀
                # 处理BERT的子词前缀
                elif token.startswith('##'):  # BERT的子词前缀
                    token = token[2:]  # 移除'##'前缀
                readable_tokens.append(token)

        return readable_tokens

    def forward(
            self,
            input_text,
            quad_text,
            attention_mask=None,
            quad_attention_mask=None,
            category_ids=None,
            visualize=True
    ):
        # Tokenize输入
        text_encodings = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.model.bert.device)

        quad_encodings = self.tokenizer(
            quad_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.model.bert.device)

        input_ids = text_encodings['input_ids']
        attention_mask = text_encodings['attention_mask']
        quad_input_ids = quad_encodings['input_ids']
        quad_attention_mask = quad_encodings['attention_mask']

        # 存储解码后的tokens
        self.text_tokens = self.decode_tokens(input_ids)
        self.quad_tokens = self.decode_tokens(quad_input_ids)

        # 文本编码
        text_outputs = self.model.bert(
            input_ids,
            attention_mask,
            output_attentions=True
        )
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output
        self.attention_scores['bert_text'] = self.get_attention_weights(text_outputs)

        # 四元组编码
        quad_outputs = self.model.quad_bert(
            quad_input_ids,
            quad_attention_mask,
            output_attentions=True
        )
        quad_hidden = quad_outputs.last_hidden_state
        quad_pooled = quad_outputs.pooler_output
        self.attention_scores['bert_quad'] = self.get_attention_weights(quad_outputs)

        # 类别增强
        if category_ids is not None:
            category_embeds = self.model.category_embedding(category_ids)
            quad_hidden = quad_hidden + category_embeds.unsqueeze(1)

        # 特征转换
        text_transformed = self.model.text_transform(text_hidden)
        quad_transformed = self.model.quad_transform(quad_hidden)

        self.attention_scores['text_transform'] = torch.norm(text_transformed, dim=-1)
        self.attention_scores['quad_transform'] = torch.norm(quad_transformed, dim=-1)

        # 交互注意力
        cross_attn_output, cross_attn_weights = self.model.cross_attention(
            query=text_transformed,
            key=quad_transformed,
            value=quad_transformed,
            key_padding_mask=~quad_attention_mask.bool(),
            need_weights=True
        )
        self.attention_scores['cross_attention'] = cross_attn_weights

        # 自注意力
        self_attn_output, self_attn_weights = self.model.self_attention(
            query=cross_attn_output,
            key=cross_attn_output,
            value=cross_attn_output,
            key_padding_mask=~attention_mask.bool(),
            need_weights=True
        )
        self.attention_scores['self_attention'] = self_attn_weights

        attn_pooled = torch.mean(self_attn_output, dim=1)

        # 特征融合
        fused = self.model.fusion(text_pooled, quad_pooled, attn_pooled)
        self.attention_scores['fusion'] = torch.norm(fused, dim=-1)

        if visualize:
            self.visualize_attention_comparison()

        return fused

    def visualize_attention_comparison(self):
        """创建两行对比的注意力热力图"""
        # 设置图形尺寸和风格
        plt.style.use('default')
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        plt.subplots_adjust(hspace=0.3)

        # 获取注意力分数
        bert_attention = self.attention_scores['bert_text'].mean(dim=1).squeeze(0)  # 平均所有头的注意力
        self_attention = self.attention_scores['self_attention'].squeeze(0)

        # 确保tokens长度与attention矩阵匹配
        tokens = self.text_tokens[:bert_attention.shape[0]]

        # 为每个注意力层创建热力图
        self._create_single_row_heatmap(
            axes[0],
            bert_attention.mean(dim=0),  # 取平均得到每个token的整体注意力分数
            tokens,
            "Text Encoder Attention\n(Before Aspect-Opinion Awareness)"
        )

        self._create_single_row_heatmap(
            axes[1],
            self_attention.mean(dim=0),  # 取平均得到每个token的整体注意力分数
            tokens,
            "Self Attention\n(After Aspect-Opinion Awareness)"
        )

        plt.tight_layout()
        return fig

    def _create_single_row_heatmap(self, ax, attention_weights, tokens, title):
        """创建单行的注意力热力图"""
        # 将注意力权重转换为numpy数组并重塑为单行
        weights = attention_weights.cpu().detach().numpy()
        weights = weights.reshape(1, -1)

        # 创建热力图
        sns.heatmap(
            weights,
            ax=ax,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=tokens,
            yticklabels=['Attention'],
            square=False
        )

        # 设置标题和标签
        ax.set_title(title, pad=20, fontsize=12)

        # 调整x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    def visualize_attention_distributions(self):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # 增大图像尺寸
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # 绘制 BERT/RoBERTa 编码器的注意力分布
        self.plot_attention_heatmap(
            axes[0, 0],
            self.attention_scores['bert_text'],
            f'{self.model_type.upper()} Text Encoder Attention',
            text_tokens=self.text_tokens,
            quad_tokens=self.text_tokens
        )

        # 绘制 BERT/RoBERTa Quad 编码器的注意力分布
        self.plot_attention_heatmap(
            axes[0, 1],
            self.attention_scores['bert_quad'],
            f'{self.model_type.upper()} Quad Encoder Attention',
            text_tokens=self.quad_tokens,
            quad_tokens=self.quad_tokens
        )

        # 绘制交互注意力的注意力分布
        self.plot_attention_heatmap(
            axes[1, 0],
            self.attention_scores['cross_attention'],
            'Cross Attention',
            text_tokens=self.text_tokens,
            quad_tokens=self.quad_tokens
        )

        # 绘制自注意力的注意力分布
        self.plot_attention_heatmap(
            axes[1, 1],
            self.attention_scores['self_attention'],
            'Self Attention',
            text_tokens=self.text_tokens,
            quad_tokens=self.text_tokens
        )

        plt.tight_layout()
        return fig

    def plot_attention_heatmap(self, ax, attention_scores, title, text_tokens=None, quad_tokens=None):
        if attention_scores is None:
            ax.axis('off')
            return

        if len(attention_scores.shape) == 4:
            attention_scores = attention_scores.mean(dim=1).squeeze(0)
        else:
            attention_scores = attention_scores.squeeze(0)

        # 获取实际的attention scores尺寸
        attn_size = attention_scores.shape[0]

        # 根据attention矩阵的大小选择合适的tokens
        if text_tokens is None:
            text_tokens = self.text_tokens[:attn_size]
        else:
            text_tokens = text_tokens[:attn_size]

        if quad_tokens is None:
            quad_tokens = text_tokens
        else:
            quad_tokens = quad_tokens[:attn_size]

        # 确保tokens长度与attention矩阵匹配
        text_tokens = text_tokens[:attention_scores.shape[0]]
        quad_tokens = quad_tokens[:attention_scores.shape[1]]

        # 创建DataFrame
        df = pd.DataFrame(attention_scores.cpu().detach().numpy(),
                          index=text_tokens,
                          columns=quad_tokens)

        # 设置图形样式
        sns.heatmap(df,
                    cmap='YlOrRd',
                    ax=ax,
                    cbar_kws={'label': 'Attention Score'},
                    square=True)  # 使单元格为正方形

        # 设置标题和标签
        ax.set_title(title, pad=20, fontsize=12)
        ax.set_xlabel('Token', fontsize=10)
        ax.set_ylabel('Token', fontsize=10)

        # 调整刻度标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, va='center', fontsize=8)

        # 添加网格线
        ax.set_xticks(np.arange(len(quad_tokens)))
        ax.set_yticks(np.arange(len(text_tokens)))
        ax.grid(False)


# 使用示例：
def visualize_model_attention(model, tokenizer, text, quad_text, category_ids=None):
    """
    Args:
        model: 预训练模型
        tokenizer: 分词器
        text (str): 输入文本
        quad_text (str): 四元组文本
        category_ids (torch.Tensor, optional): 类别ID

    Returns:
        matplotlib.figure.Figure: 可视化图像
    """
    attention_vis = AttentionVisualization(model, tokenizer)
    with torch.no_grad():
        _ = attention_vis(
            input_text=text,
            quad_text=quad_text,
            category_ids=category_ids,
            visualize=True
        )
    # return attention_vis.visualize_attention_distributions()
    return attention_vis.visualize_attention_comparison()