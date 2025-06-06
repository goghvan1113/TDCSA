import argparse
import os
import random
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, PreTrainedModel, AutoModel, Trainer, PretrainedConfig, AutoTokenizer, \
    TrainingArguments, TrainerCallback
from transformers.modeling_outputs import ModelOutput
from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Union, List, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bert.custom_loss import MultiFocalLoss
from bert.plot_results import plot_confusion_matrix


def main(args):
    # 1. 设置一些超参数
    sentiment_id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
    sentiment_label2id = {"Neutral": 0, "Positive": 1, "Negative": 2}
    intent_id2label = {0: "background", 1: "method", 2: "result"}
    intent_label2id = {"background": 0, "method": 1, "result": 2}

    # 2. 创建模型
    device = torch.device(args.device)
    config = MultitaskConfig(
        num_sentiment_labels=3,
        num_intent_labels=3,
        loss_type=args.loss_type,
        focal_alpha=0.8, # 0.8
        focal_gamma=2.0, # 2.0
        sentiment_weight=5.0,
        intent_weight=1.0,
        label_smoothing=0.1,
        backbone_model=args.model_name
    )
    model = MultitaskModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")

    # 3. 定义评估指标
    train_data, val_data, test_data = load_multitask_datasets(
        test_size=0.2,
        val_size=0.1,
        seed=args.seed,
        tokenizer=tokenizer,
        stratify_by_sentiment=True
    )
    print(f"Train Dataset Size: {len(train_data)}")
    print(f"Test Dataset Size: {len(test_data)}")
    print(f"Val Dataset Size: {len(val_data)}")

    # 4. 训练和评估
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
        metric_for_best_model='sentiment_f1_macro',
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    # 创建训练器和回调函数
    plotting_callback = MetricsPlottingCallback()

    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model.config),
        callbacks=[plotting_callback]
    )
    trainer.train()

    # 最终评估
    eval_result = trainer.evaluate()
    test_result = trainer.predict(test_data)

    # 打印最终结果
    print("\nFinal Evaluation Results:")
    for key, value in eval_result.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    print("\nTest Results:")
    for key, value in test_result.metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    sentiment_preds = np.argmax(test_result.predictions[0], axis=1)
    intent_preds = np.argmax(test_result.predictions[1], axis=1)

    plot_confusion_matrix(test_data.sentiment_labels, sentiment_preds, list(sentiment_label2id.keys()), title='Sentiment Confusion Matrix')
    plot_confusion_matrix(test_data.intent_labels, intent_preds, list(intent_label2id.keys()), title='Intent Confusion Matrix')


@dataclass
class MultitaskOutput(ModelOutput):
    """
    定义多任务模型的输出格式
    """
    loss: Optional[torch.FloatTensor] = None
    sentiment_logits: torch.FloatTensor = None
    intent_logits: torch.FloatTensor = None
    task_weights: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class MultitaskConfig(PretrainedConfig):
    """
    多任务模型配置类
    """
    def __init__(
            self,
            num_sentiment_labels: int = 3,
            num_intent_labels: int = 3,
            loss_type: str = "ce",
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            sentiment_weight: float = 1.0,
            intent_weight: float = 1.0,
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

        # 多任务相关的配置
        self.num_sentiment_labels = num_sentiment_labels
        self.num_intent_labels = num_intent_labels
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.sentiment_weight = sentiment_weight
        self.intent_weight = intent_weight
        self.label_smoothing = label_smoothing
        self.backbone_model = backbone_model


class LearnableTaskWeights(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # 初始化可学习的任务权重参数
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

    def forward(self):
        # 使用softplus确保权重为正数
        weights =  F.softplus(self.task_weights)
        return weights / weights.sum()


class MultitaskModel(PreTrainedModel):
    """
    多任务模型类
    """
    config_class = MultitaskConfig

    def __init__(self, config: MultitaskConfig):
        super().__init__(config)

        # 加载backbone模型
        self.backbone = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        self.shared_layer = nn.Linear(config.hidden_size, config.hidden_size)

        # 任务特定的特征提取层
        self.sentiment_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.intent_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        # 轻量级软参数共享层
        self.sentiment_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.intent_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        # 分类器
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_sentiment_labels)
        )
        self.intent_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_intent_labels)
        )
        self.task_weights = LearnableTaskWeights(num_tasks=2)


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            sentiment_labels: Optional[torch.LongTensor] = None,
            intent_labels: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Union[tuple, MultitaskOutput]:

        # 获取backbone的输出
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        base_features = output[0][:, 0, :]  # 取CLS位置的特征

        shared_features = self.shared_layer(base_features)

        # 任务特定特征
        sentiment_features = self.sentiment_extractor(base_features)
        intent_features = self.intent_extractor(base_features)

        # 软参数共享通过门控机制
        sentiment_gate = self.sentiment_gate(
            torch.cat([shared_features, sentiment_features], dim=-1)
        )
        intent_gate = self.intent_gate(
            torch.cat([shared_features, intent_features], dim=-1)
        )

        # 最终特征
        final_sentiment = sentiment_gate * shared_features + (1 - sentiment_gate) * sentiment_features
        final_intent = intent_gate * shared_features + (1 - intent_gate) * intent_features

        # 分类
        sentiment_logits = self.sentiment_classifier(final_sentiment)
        intent_logits = self.intent_classifier(final_intent)

        weights = self.task_weights()
        # 不在forward函数里面计算损失

        return MultitaskOutput(
            sentiment_logits=sentiment_logits,
            intent_logits=intent_logits,
            task_weights=weights
        )

class MetricsPlottingCallback(TrainerCallback):
    """Callback for plotting training metrics at the end of training"""

    def __init__(self):
        self.train_metrics = {
            'steps': [],
            'total_loss': [],
            'sentiment_loss': [],
            'intent_loss': []
        }
        self.eval_metrics = {
            'steps': [],
            'total_loss': [],
            'sentiment_loss': [],
            'intent_loss': [],
            'sentiment_accuracy': [],
            'intent_accuracy': [],
            'sentiment_f1': [],
            'intent_f1': []
        }
        self.weights_metrics = {
            'steps': [],
            'sentiment_weight': [],
            'intent_weight': []
        }
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 15))

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Record training metrics at each step"""
        if state.is_world_process_zero: # one process will be True
            # 训练指标记录
            if all(key in logs for key in ['loss', 'sentiment_loss', 'intent_loss']):
                # 确保是训练日志而不是评估日志
                if not any(key.startswith('eval_') for key in logs.keys()):
                    step = state.global_step
                    # 避免重复记录同一步骤
                    if not self.train_metrics['steps'] or step > self.train_metrics['steps'][-1]:
                        self.train_metrics['steps'].append(step)
                        self.train_metrics['total_loss'].append(logs['loss'])
                        self.train_metrics['sentiment_loss'].append(logs['sentiment_loss'])
                        self.train_metrics['intent_loss'].append(logs['intent_loss'])

            # 评估指标记录
            if 'eval_loss' in logs:
                step = state.global_step
                # 避免重复记录同一步骤
                if not self.eval_metrics['steps'] or step > self.eval_metrics['steps'][-1]:
                    self.eval_metrics['steps'].append(step)
                    self.eval_metrics['total_loss'].append(logs['eval_loss'])
                    self.eval_metrics['sentiment_loss'].append(logs.get('eval_sentiment_loss', 0))
                    self.eval_metrics['intent_loss'].append(logs.get('eval_intent_loss', 0))
                    self.eval_metrics['sentiment_accuracy'].append(logs.get('eval_sentiment_accuracy', 0))
                    self.eval_metrics['intent_accuracy'].append(logs.get('eval_intent_accuracy', 0))
                    self.eval_metrics['sentiment_f1'].append(logs.get('eval_sentiment_f1_macro', 0))
                    self.eval_metrics['intent_f1'].append(logs.get('eval_intent_f1_macro', 0))

                    # 记录权重
                    if 'eval_sentiment_weight' in logs:
                        if not self.weights_metrics['steps'] or step > self.weights_metrics['steps'][-1]:
                            self.weights_metrics['steps'].append(step)
                            self.weights_metrics['sentiment_weight'].append(logs['eval_sentiment_weight'])
                            self.weights_metrics['intent_weight'].append(logs['eval_intent_weight'])


    def on_train_end(self, args, state, control, **kwargs):
        """Plot metrics at the end of training"""
        self._plot_metrics()

    def _plot_metrics(self):
        """Plot training and evaluation metrics"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Plot losses
        if self.train_metrics['total_loss']:
            # min_length = min(len(self.train_metrics['steps']), len(self.train_metrics['sentiment_loss']))
            self.ax1.plot(self.train_metrics['steps'], self.train_metrics['total_loss'],
                          label='Train Total Loss', color='blue')
            if self.train_metrics['sentiment_loss']:
                self.ax1.plot(self.train_metrics['steps'], self.train_metrics['sentiment_loss'],
                              label='Train Sentiment Loss', color='lightblue')
            if self.train_metrics['intent_loss']:
                self.ax1.plot(self.train_metrics['steps'], self.train_metrics['intent_loss'],
                              label='Train Intent Loss', color='navy')

        if self.eval_metrics['total_loss']:
            self.ax1.plot(self.eval_metrics['steps'], self.eval_metrics['total_loss'],
                          label='Val Total Loss', color='red')
            self.ax1.plot(self.eval_metrics['steps'], self.eval_metrics['sentiment_loss'],
                          label='Val Sentiment Loss', color='pink')
            self.ax1.plot(self.eval_metrics['steps'], self.eval_metrics['intent_loss'],
                          label='Val Intent Loss', color='darkred')

        self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training and Validation Losses')
        self.ax1.grid(True)
        self.ax1.legend()

        # Plot metrics
        if self.eval_metrics['sentiment_accuracy']:
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['sentiment_accuracy'],
                          label='Sentiment Accuracy', color='green')
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['sentiment_f1'],
                          label='Sentiment F1', color='lightgreen')
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['intent_accuracy'],
                          label='Intent Accuracy', color='purple')
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['intent_f1'],
                          label='Intent F1', color='plum')

        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Score')
        self.ax2.set_title('Validation Metrics')
        self.ax2.grid(True)
        self.ax2.legend()

        # 添加权重变化图
        if self.weights_metrics['sentiment_weight']:
            self.ax3.plot(self.weights_metrics['steps'],
                          self.weights_metrics['sentiment_weight'],
                          label='Sentiment Weight', color='orange')
            self.ax3.plot(self.weights_metrics['steps'],
                          self.weights_metrics['intent_weight'],
                          label='Intent Weight', color='brown')

        self.ax3.set_xlabel('Steps')
        self.ax3.set_ylabel('Weight Value')
        self.ax3.set_title('Task Weights Evolution')
        self.ax3.grid(True)
        self.ax3.legend()

        plt.tight_layout()
        plt.show()


class MultitaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        sentiment_labels = inputs["sentiment_labels"]
        intent_labels = inputs["intent_labels"]
        outputs = model(**inputs)

        sentiment_logits = outputs.sentiment_logits
        intent_logits = outputs.intent_logits
        task_weights = outputs.task_weights

        focal_loss = MultiFocalLoss(num_class=3, alpha=model.config.focal_alpha, gamma=model.config.focal_gamma)

        if model.config.loss_type == "focal_loss":
            sentiment_loss = focal_loss(sentiment_logits, sentiment_labels)
            intent_loss = focal_loss(intent_logits, intent_labels)
        else:
            sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels, label_smoothing=model.config.label_smoothing)
            intent_loss = F.cross_entropy(intent_logits, intent_labels, label_smoothing=model.config.label_smoothing)

        loss = model.config.sentiment_weight * sentiment_loss + model.config.intent_weight * intent_loss
        # loss = task_weights[0] * sentiment_loss + task_weights[1] * intent_loss

        # 决定了logs里面的步长，画出图来就是这个步长
        # 只在指定的步数记录日志
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss": loss.item(),
                "sentiment_loss": sentiment_loss.item(),
                "intent_loss": intent_loss.item(),
                "sentiment_weight": task_weights[0].item(),
                "intent_weight": task_weights[1].item()
            })

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred, config):
    """
       计算评估指标
       eval_pred: 包含模型预测结果和真实标签的元组，结构为:
       (predictions, labels) 其中
       predictions是一个包含(sentiment_logits, intent_logits, task_weights)的元组
       labels是一个包含(sentiment_labels, intent_labels)的元组
    """
    predictions, labels = eval_pred
    sentiment_logits, intent_logits, task_weights = predictions  # 解包predictions
    sentiment_labels, intent_labels = labels  # 解包labels

    sentiment_logits = torch.tensor(sentiment_logits)
    intent_logits = torch.tensor(intent_logits)
    sentiment_labels = torch.tensor(sentiment_labels)
    intent_labels = torch.tensor(intent_labels)



    if config.loss_type == "focal_loss":
        loss_fct = MultiFocalLoss(num_class=3, alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        loss_fct = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
    intent_loss = loss_fct(intent_logits, intent_labels)

    loss = config.sentiment_weight * sentiment_loss + config.intent_weight * intent_loss

    # 计算预测结果
    sentiment_preds = np.argmax(sentiment_logits, axis=1)
    intent_preds = np.argmax(intent_logits, axis=1)
    # 计算准确率和F1分数
    sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_preds)
    sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='macro')
    intent_accuracy = accuracy_score(intent_labels, intent_preds)
    intent_f1 = f1_score(intent_labels, intent_preds, average='macro')

    # 获取当前任务权重
    current_sentiment_weight = float(task_weights[0])
    current_intent_weight = float(task_weights[1])

    return {
        "eval_loss": loss.item(),
        "eval_sentiment_loss": sentiment_loss.item(),
        "eval_intent_loss": intent_loss.item(),
        "sentiment_accuracy": sentiment_accuracy,
        "sentiment_f1_macro": sentiment_f1,
        "intent_accuracy": intent_accuracy,
        "intent_f1_macro": intent_f1,
        "macro_f1": (sentiment_f1 + intent_f1) / 2,
        "sentiment_weight": current_sentiment_weight,
        "intent_weight": current_intent_weight
    }

@dataclass
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, sentiment_labels, intent_labels):
        self.encodings = encodings
        self.sentiment_labels = sentiment_labels
        self.intent_labels = intent_labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['sentiment_labels'] = torch.tensor(self.sentiment_labels[idx])
        item['intent_labels'] = torch.tensor(self.intent_labels[idx])
        return item

    def __len__(self):
        return len(self.sentiment_labels)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_multitask_datasets(test_size=0.2, val_size=0.1, seed=42, tokenizer=None, stratify_by_sentiment=True):
    df = pd.read_csv('../output/intent_results/corpus_with_intent_bert.csv', encoding='utf-8')
    # df = df[df['intent'] != 'unknown']
    # df = df[df['confidence'] > 0.6] # 处理unknown和低置信度的样本，低置信度的样本会影响分类效果

    # intent_label2id = {"background": 0, "method": 1, "result": 2}
    # df['intent'] = df['intent'].map(intent_label2id).astype(int)

    texts = df['text'].tolist()
    sentiment_labels = df['sentiment'].tolist()
    intent_labels = df['intent'].tolist()

    train_val_texts, test_texts, train_val_labels, test_sentiment_labels, train_val_labels_intent, test_intent_labels = train_test_split(
        texts,
        sentiment_labels,
        intent_labels,
        test_size=test_size,
        stratify=sentiment_labels if stratify_by_sentiment else None,
        random_state=seed
    )

    val_ratio = val_size / (1 - test_size)
    train_texts, val_texts, train_sentiment_labels, val_sentiment_labels, train_intent_labels, val_intent_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        train_val_labels_intent,
        test_size=val_ratio,
        stratify=train_val_labels if stratify_by_sentiment else None,
        random_state=seed)

    train_data = MultiTaskDataset(
        tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt', max_length=512),
        train_sentiment_labels,
        train_intent_labels
    )
    val_data = MultiTaskDataset(
        tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt', max_length=512),
        val_sentiment_labels,
        val_intent_labels
    )
    test_data = MultiTaskDataset(
        tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt', max_length=512),
        test_sentiment_labels,
        test_intent_labels
    )

    return train_data, val_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='roberta-llama3.1405B-twitter-sentiment',
                        help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--loss_type', type=str, default='focal_loss', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)

