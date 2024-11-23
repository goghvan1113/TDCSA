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
        }

    def _get_model_type(self):
        """根据模型配置确定模型类型"""
        if hasattr(self.model, 'config'):
            # 对于自定义模型，检查backbone_model配置
            if hasattr(self.model.config, 'backbone_model'):
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

    def merge_subwords(self, tokens, attention_weights):
        """将子词合并为完整词，并相应地合并注意力权重"""
        special_tokens = self.get_special_tokens()
        merged_tokens = []
        merged_weights = []

        i = 0
        while i < len(tokens):
            if tokens[i] in [special_tokens['cls_display'], special_tokens['sep_display'], '[PAD]']:
                merged_tokens.append(tokens[i])
                merged_weights.append(attention_weights[:, i:i + 1])
                i += 1
                continue

            # 收集属于同一个词的所有子词
            current_word = [tokens[i]]
            current_indices = [i]
            i += 1
            while i < len(tokens):
                if (self.model_type == 'bert' and tokens[i].startswith('##')) or \
                        (self.model_type == 'roberta' and not tokens[i].startswith('Ġ')):
                    current_word.append(tokens[i].replace('##', ''))
                    current_indices.append(i)
                    i += 1
                else:
                    break

            # 合并子词
            merged_token = ''.join(current_word)
            if self.model_type == 'roberta' and merged_token.startswith('Ġ'):
                merged_token = merged_token[1:]
            merged_tokens.append(merged_token)

            # 合并注意力权重（取平均）
            merged_weight = attention_weights[:, current_indices].mean(dim=1, keepdim=True)
            merged_weights.append(merged_weight)

        return merged_tokens, torch.cat(merged_weights, dim=1)

    def decode_tokens(self, input_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        special_tokens = self.get_special_tokens()

        readable_tokens = []
        for token in tokens:
            if token == special_tokens['cls_token']:
                readable_tokens.append(special_tokens['cls_display'])
            elif token == special_tokens['sep_token']:
                readable_tokens.append(special_tokens['sep_display'])
            elif token == special_tokens['pad_token']:
                readable_tokens.append('[PAD]')
            elif token == special_tokens['mask_token']:
                readable_tokens.append('[MASK]')
            else:
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

        self.text_tokens = self.decode_tokens(input_ids)
        self.quad_tokens = self.decode_tokens(quad_input_ids)

        text_outputs = self.model.bert(
            input_ids,
            attention_mask,
            output_attentions=True
        )
        text_hidden = text_outputs.last_hidden_state
        self.attention_scores['bert_text'] = self.get_attention_weights(text_outputs)

        quad_outputs = self.model.quad_bert(
            quad_input_ids,
            quad_attention_mask,
            output_attentions=True
        )
        quad_hidden = quad_outputs.last_hidden_state
        self.attention_scores['bert_quad'] = self.get_attention_weights(quad_outputs)

        if category_ids is not None:
            category_embeds = self.model.category_embedding(category_ids)
            quad_hidden = quad_hidden + category_embeds.unsqueeze(1)

        text_transformed = self.model.text_transform(text_hidden)
        quad_transformed = self.model.quad_transform(quad_hidden)

        cross_attn_output, cross_attn_weights = self.model.cross_attention(
            query=text_transformed,
            key=quad_transformed,
            value=quad_transformed,
            key_padding_mask=~quad_attention_mask.bool(),
            need_weights=True
        )
        self.attention_scores['cross_attention'] = cross_attn_weights

        self_attn_output, self_attn_weights = self.model.self_attention(
            query=cross_attn_output,
            key=cross_attn_output,
            value=cross_attn_output,
            key_padding_mask=~attention_mask.bool(),
            need_weights=True
        )
        self.attention_scores['self_attention'] = self_attn_weights

        if visualize:
            self.visualize_attention_comparison()

        return self_attn_output

    def visualize_attention_comparison(self):
        plt.style.use('default')
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        plt.subplots_adjust(hspace=0.3)

        # 获取注意力分数并归一化
        bert_attention = self.attention_scores['bert_text'].mean(dim=1).squeeze(0)
        self_attention = self.attention_scores['self_attention'].squeeze(0)

        # 归一化处理
        bert_attention = bert_attention / bert_attention.max()
        self_attention = self_attention / self_attention.max()

        # 合并子词
        tokens = self.text_tokens[:bert_attention.shape[0]]
        merged_tokens, bert_merged_attention = self.merge_subwords(tokens, bert_attention)
        _, self_merged_attention = self.merge_subwords(tokens, self_attention)

        # 创建热力图
        self._create_single_row_heatmap(
            axes[0],
            bert_merged_attention.mean(dim=0),
            merged_tokens,
            "Text Encoder Attention\n(Before Aspect-Opinion Awareness)",
            cmap='BuGn'  # 使用蓝绿配色
        )

        self._create_single_row_heatmap(
            axes[1],
            self_merged_attention.mean(dim=0),
            merged_tokens,
            "Self Attention\n(After Aspect-Opinion Awareness)",
            cmap='BuGn'  # 使用蓝绿配色
        )

        plt.tight_layout()
        return fig

    def _create_single_row_heatmap(self, ax, attention_weights, tokens, title, cmap='BuGn'):
        weights = attention_weights.cpu().detach().numpy()
        weights = weights.reshape(1, -1)

        sns.heatmap(
            weights,
            ax=ax,
            cmap=cmap,
            cbar=False,
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=tokens,
            yticklabels=['Attention'],
            square=False
        )

        ax.set_title(title, pad=20, fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)


def visualize_model_attention(model, tokenizer, text, quad_text, category_ids=None):
    attention_vis = AttentionVisualization(model, tokenizer)
    with torch.no_grad():
        _ = attention_vis(
            input_text=text,
            quad_text=quad_text,
            category_ids=category_ids,
            visualize=True
        )
    return attention_vis.visualize_attention_comparison()