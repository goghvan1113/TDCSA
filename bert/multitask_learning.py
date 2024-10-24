import argparse
import os
import random
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
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



def main(args):
    # 1. 设置一些超参数
    test_size = 0.4
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
        focal_alpha=0.25, # 0.8
        focal_gamma=2.0, # 2.0
        sentiment_weight=1.0,
        intent_weight=1.0,
        label_smoothing=0.1,
        backbone_model=args.model_name
    )
    model = MultitaskModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")

    # 3. 定义评估指标
    train_data, val_data, test_data = load_multitask_datasets(test_size=test_size, seed=args.seed, tokenizer=tokenizer)
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
        compute_metrics=compute_metrics,
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


@dataclass
class MultitaskOutput(ModelOutput):
    """
    定义多任务模型的输出格式
    """
    loss: Optional[torch.FloatTensor] = None
    sentiment_logits: torch.FloatTensor = None
    intent_logits: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class MultitaskConfig(PretrainedConfig):
    """
    多任务模型配置类
    """
    model_type = "multitask"

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


class MultitaskModel(PreTrainedModel):
    """
    多任务模型类
    """
    config_class = MultitaskConfig

    def __init__(self, config: MultitaskConfig):
        super().__init__(config)

        # 加载backbone模型
        self.backbone = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')

        # 两个分类头
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_sentiment_labels)
        )

        self.intent_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_intent_labels)
        )

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化分类头的权重"""
        for module in [self.sentiment_classifier, self.intent_classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(mean=0.0, std=0.02)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            sentiment_labels: Optional[torch.LongTensor] = None,
            intent_labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[tuple, MultitaskOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取backbone的输出
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )

        # 获取[CLS]位置的输出
        pooled_output = outputs[0][:, 0, :]

        # 通过分类头获取logits
        sentiment_logits = self.sentiment_classifier(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        # 不在forward函数里面计算损失

        if not return_dict:
            output = (sentiment_logits, intent_logits)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            if output_attentions:
                output += (outputs.attentions,)
            return output

        return MultitaskOutput(
            sentiment_logits=sentiment_logits,
            intent_logits=intent_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
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
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Record training metrics at each step"""
        if state.is_world_process_zero:
            if 'loss' in logs:
                self.train_metrics['steps'].append(state.global_step)
                self.train_metrics['total_loss'].append(logs.get('loss'))
                if 'sentiment_loss' in logs:
                    self.train_metrics['sentiment_loss'].append(logs.get('sentiment_loss'))
                if 'intent_loss' in logs:
                    self.train_metrics['intent_loss'].append(logs.get('intent_loss'))

            if 'eval_loss' in logs:
                self.eval_metrics['steps'].append(state.global_step)
                self.eval_metrics['total_loss'].append(logs.get('eval_loss'))
                self.eval_metrics['sentiment_loss'].append(logs.get('eval_sentiment_loss', 0))
                self.eval_metrics['intent_loss'].append(logs.get('eval_intent_loss', 0))
                self.eval_metrics['sentiment_accuracy'].append(logs.get('eval_sentiment_accuracy', 0))
                self.eval_metrics['intent_accuracy'].append(logs.get('eval_intent_accuracy', 0))
                self.eval_metrics['sentiment_f1'].append(logs.get('eval_sentiment_f1_macro', 0))
                self.eval_metrics['intent_f1'].append(logs.get('eval_intent_f1_macro', 0))

    def on_train_end(self, args, state, control, **kwargs):
        """Plot metrics at the end of training"""
        self._plot_metrics()
        plt.show()

    def _plot_metrics(self):
        """Plot training and evaluation metrics"""
        self.ax1.clear()
        self.ax2.clear()

        # Plot losses
        if self.train_metrics['total_loss']:
            min_length = min(len(self.train_metrics['steps']), len(self.train_metrics['sentiment_loss']))
            self.ax1.plot(self.train_metrics['steps'], self.train_metrics['total_loss'],
                          label='Train Total Loss', color='blue')
            if self.train_metrics['sentiment_loss']:
                self.ax1.plot(self.train_metrics['steps'][:min_length], self.train_metrics['sentiment_loss'][:min_length],
                              label='Train Sentiment Loss', color='lightblue')
            if self.train_metrics['intent_loss']:
                self.ax1.plot(self.train_metrics['steps'][:min_length], self.train_metrics['intent_loss'][:min_length],
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

        plt.tight_layout()

class MultitaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        sentiment_labels = inputs["sentiment_labels"]
        intent_labels = inputs["intent_labels"]
        outputs = model(**inputs)

        sentiment_logits = outputs.sentiment_logits
        intent_logits = outputs.intent_logits

        def focal_loss(logits, labels):
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            return (model.config.focal_alpha * (1 - pt) ** model.config.focal_gamma * ce_loss).mean()

        def smoothing_loss(logits, labels, smoothing):
            confidence = 1.0 - smoothing
            num_classes = logits.shape[-1]
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), confidence)
            return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))

        if model.config.loss_type == "focal_loss":
            sentiment_loss = focal_loss(sentiment_logits, sentiment_labels)
            intent_loss = focal_loss(intent_logits, intent_labels)
        else:
            if model.config.label_smoothing > 0:
                sentiment_loss = smoothing_loss(sentiment_logits, sentiment_labels, model.config.label_smoothing)
                intent_loss = smoothing_loss(intent_logits, intent_labels, model.config.label_smoothing)
            else:
                sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels)
                intent_loss = F.cross_entropy(intent_logits, intent_labels)

        loss = model.config.sentiment_weight * sentiment_loss + model.config.intent_weight * intent_loss

        self.log({
            "loss": loss.item(),
            "sentiment_loss": sentiment_loss.item(),
            "intent_loss": intent_loss.item()
        })

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    (sentiment_logits, intent_logits), (sentiment_labels, intent_labels) = eval_pred

    # 计算预测结果
    sentiment_preds = np.argmax(sentiment_logits, axis=1)
    intent_preds = np.argmax(intent_logits, axis=1)

    # 计算损失
    sentiment_loss = F.cross_entropy(torch.tensor(sentiment_logits),
                                     torch.tensor(sentiment_labels)).item()
    intent_loss = F.cross_entropy(torch.tensor(intent_logits),
                                  torch.tensor(intent_labels)).item()

    # 计算准确率和F1分数
    sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_preds)
    sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='macro')
    intent_accuracy = accuracy_score(intent_labels, intent_preds)
    intent_f1 = f1_score(intent_labels, intent_preds, average='macro')

    return {
        "sentiment_loss": sentiment_loss,
        "intent_loss": intent_loss,
        "sentiment_accuracy": sentiment_accuracy,
        "sentiment_f1_macro": sentiment_f1,
        "intent_accuracy": intent_accuracy,
        "intent_f1_macro": intent_f1,
        "macro_f1": (sentiment_f1 + intent_f1) / 2
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


def load_multitask_datasets(test_size=0.2, seed=42, tokenizer=None):
    df = pd.read_csv('../output/corpus_with_intent.csv')
    df = df[df['intent'] != 'unknown']
    df = df[df['confidence'] >= 0.8] # 处理unknown和低置信度的样本

    intent_label2id = {"background": 0, "method": 1, "result": 2}
    df['intent'] = df['intent'].map(intent_label2id).astype(int)

    texts = df['text'].tolist()
    sentiment_labels = df['sentiment'].tolist()
    intent_labels = df['intent'].tolist()

    train_texts, temp_texts, train_sentiment_labels, temp_sentiment_labels, train_intent_labels, temp_intent_labels = train_test_split(
        texts, sentiment_labels, intent_labels, test_size=test_size, stratify=sentiment_labels, random_state=seed
    )

    val_texts, test_texts, val_sentiment_labels, test_sentiment_labels, val_intent_labels, test_intent_labels = train_test_split(
        temp_texts, temp_sentiment_labels, temp_intent_labels, test_size=0.5, stratify=temp_sentiment_labels, random_state=seed
    )

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
    parser.add_argument('--loss_type', type=str, default='ce_loss', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)

