import argparse
import json
import random
from typing import List, Dict, Tuple
from collections import Counter

import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoModel, PreTrainedModel,
    PretrainedConfig, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorWithPadding, TrainerCallback
)
import matplotlib.pyplot as plt
import seaborn as sns

from bert.custom_loss import MultiFocalLoss
from bert.plot_results import plot_confusion_matrix

CATEGORY2ID = {
    'METHODOLOGY': 0,
    'PERFORMANCE': 1,
    'INNOVATION': 2,
    'APPLICABILITY': 3,
    'LIMITATION': 4
}
ID2CATEGORY = {v: k for k, v in CATEGORY2ID.items()}

class ASQPFinetuningConfig(PretrainedConfig):
    """Configuration for aspect category and sentiment classification."""

    def __init__(
        self,
        aspect_category_labels: int = 6,  # Number of aspect categories
        aspect_sentiment_labels: int = 2,  # Positive, Negative
        loss_type: str = "ce_loss",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        loss_weight: dict = {'aspect_category': 1.0, 'aspect_sentiment': 1.0},
        backbone_model: str = "roberta-base",
        max_length: int = 512,
        dropout_prob: float = 0.1,
        label_smoothing: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.aspect_category_labels = aspect_category_labels
        self.aspect_sentiment_labels = aspect_sentiment_labels
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_weight = loss_weight
        self.backbone_model = backbone_model
        self.max_length = max_length
        self.dropout_prob = dropout_prob
        self.label_smoothing = label_smoothing

class ASQPFinetuningModel(PreTrainedModel):
    config_class = ASQPFinetuningConfig

    def __init__(self, config: ASQPFinetuningConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(config.dropout_prob)
        self.aspect_category_classifier = nn.Linear(hidden_size, config.aspect_category_labels)
        self.aspect_sentiment_classifier = nn.Linear(hidden_size, config.aspect_sentiment_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        aspect_category_labels=None,
        aspect_sentiment_labels=None
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        aspect_category_logits = self.aspect_category_classifier(pooled_output)
        aspect_sentiment_logits = self.aspect_sentiment_classifier(pooled_output)

        # 将所有输出放入一个字典
        output = {
            'logits': (aspect_category_logits, aspect_sentiment_logits),  # 确保返回logits元组
        }
        return output  # 返回字典，包含loss和logits

# 添加自定义的 Trainer 类
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取标签
        aspect_category_labels = inputs.pop('aspect_category_labels')
        aspect_sentiment_labels= inputs.pop('aspect_sentiment_labels')
        outputs = model(**inputs)

        aspect_category_logits, aspect_sentiment_logits = outputs['logits']

        if model.config.loss_type == "focal_loss":
            loss_fct = MultiFocalLoss(
                num_class=model.config.aspect_category_labels,
                alpha=model.config.focal_alpha,
                gamma=model.config.focal_gamma
            )
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=model.config.label_smoothing)

        aspect_category_loss = loss_fct(aspect_category_logits, aspect_category_labels)
        aspect_sentiment_loss = loss_fct(aspect_sentiment_logits, aspect_sentiment_labels)

        loss = model.config.loss_weight['aspect_category'] * aspect_category_loss +\
                model.config.loss_weight['aspect_sentiment'] * aspect_sentiment_loss

        # 在日志中记录各个损失
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'loss': loss.item(),
                'aspect_category_loss': aspect_category_loss.item(),
                'aspect_sentiment_loss': aspect_sentiment_loss.item()
            })

        return (loss, outputs) if return_outputs else loss

class ASQPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        aspects: List[str],
        opinions: List[str],
        aspect_categories: List[int],
        aspect_sentiments: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for text, aspect, opinion, category, sentiment in zip(texts, aspects, opinions, aspect_categories, aspect_sentiments):
            input_text = f"{text} <s> </s> {aspect} is {opinion}" # roberta是<s> </s>，bert是[CLS] [SEP]
            encoding = self.tokenizer(
                input_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.samples.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'aspect_category_labels': torch.tensor(category, dtype=torch.long),
                'aspect_sentiment_labels': torch.tensor(sentiment, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def compute_metrics(eval_pred, config):
    logits, labels = eval_pred
    aspect_category_logits, aspect_sentiment_logits = logits  # 解包 logits，第一个是总的logits
    aspect_category_labels, aspect_sentiment_labels = labels  # 解包 labels

    # 转换为张量 ，下面代码是计算验证集的多任务loss的
    aspect_category_logits = torch.tensor(aspect_category_logits)
    aspect_sentiment_logits = torch.tensor(aspect_sentiment_logits)
    aspect_category_labels = torch.tensor(aspect_category_labels)
    aspect_sentiment_labels = torch.tensor(aspect_sentiment_labels)
    # 选择损失函数
    if config.loss_type == "focal_loss":
        loss_fct = MultiFocalLoss(
            num_class=config.aspect_category_labels,
            alpha=config.focal_alpha,
            gamma=config.focal_gamma
        )
    else:
        loss_fct = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    # 计算损失
    aspect_category_loss = loss_fct(aspect_category_logits, aspect_category_labels).item()
    aspect_sentiment_loss = loss_fct(aspect_sentiment_logits, aspect_sentiment_labels).item()
    total_loss = (
            config.loss_weight['aspect_category'] * aspect_category_loss +
            config.loss_weight['aspect_sentiment'] * aspect_sentiment_loss
    )

    aspect_category_preds = np.argmax(aspect_category_logits, axis=1)
    aspect_sentiment_preds = np.argmax(aspect_sentiment_logits, axis=1)

    aspect_category_labels = aspect_category_labels.flatten()
    aspect_sentiment_labels = aspect_sentiment_labels.flatten()

    aspect_category_report = classification_report(
        aspect_category_labels,
        aspect_category_preds,
        target_names=list(CATEGORY2ID.keys()),
        output_dict=True,
        zero_division=0
    )
    aspect_sentiment_report = classification_report(
        aspect_sentiment_labels,
        aspect_sentiment_preds,
        target_names=['Negative', 'Positive'],
        output_dict=True, # 返回字典格式对应下面的字典取值
        zero_division=0
    )

    return {
        'eval_loss': total_loss,
        'eval_aspect_category_loss': aspect_category_loss,
        'eval_aspect_sentiment_loss': aspect_sentiment_loss,
        'aspect_category_f1': aspect_category_report['macro avg']['f1-score'],
        'aspect_sentiment_f1': aspect_sentiment_report['macro avg']['f1-score'],
        'aspect_category_accuracy': aspect_category_report['accuracy'],
        'aspect_sentiment_accuracy': aspect_sentiment_report['accuracy'],
    }


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

    def split_data(self, stratify_by_categories: bool = True):
        """
        划分数据集 返回训练集、验证集和测试集

        Returns:
            Dictionary containing train, validation, and test splits with quad-level samples.
        """
        # Extract quad-level samples and category labels
        quad_samples = []
        categories = []
        for sample in self.pos_neg_samples:
            text = sample['text']
            overall_sentiment = sample['overall_sentiment']
            for quads in sample['quads']: # 遍历单个文本的多个四元组
                if quads[2] in CATEGORY2ID:  # Check if category is valid
                    quad_samples.append({
                        'text': text,
                        'overall_sentiment': overall_sentiment,
                        'quads': quads
                    })
                    categories.append(CATEGORY2ID[quads[2]])

        # First split: train_val and test
        quad_train_val, quad_test, categories_train_val, categories_test = train_test_split(
            quad_samples,
            categories,
            test_size=self.test_size,
            stratify=categories if stratify_by_categories else None,
            random_state=self.random_state
        )

        # Second split: train and val
        val_ratio = self.val_size / (1 - self.test_size)
        quad_train, quad_val, _, _ = train_test_split(
            quad_train_val,
            categories_train_val,
            test_size=val_ratio,
            stratify=categories_train_val if stratify_by_categories else None,
            random_state=self.random_state
        )

        # Print category distribution for each split
        print("Train set category distribution:")
        print(Counter(CATEGORY2ID[quad['quads'][2]] for quad in quad_train))
        print("Validation set category distribution:")
        print(Counter(CATEGORY2ID[quad['quads'][2]] for quad in quad_val))
        print("Test set category distribution:")
        print(Counter(CATEGORY2ID[quad['quads'][2]] for quad in quad_test))

        return {
            'train': quad_train,
            'val': quad_val,
            'test': quad_test
        }


def create_datasets(split_data: Dict, tokenizer):
    """创建训练、验证和测试数据集，包含中性样本。"""
    datasets = {}
    for split_name in ['train', 'val', 'test']:
        pos_neg_samples = split_data[split_name] # 只返回了正负样本

        texts = []
        citation_sentiment_labels = []
        aspects = []
        opinions = []
        aspect_categories = []
        aspect_sentiments = []

        # 处理正负样本
        for sample in pos_neg_samples:
            text = sample['text']
            overall_sentiment = sample['overall_sentiment']
            sentiment_label = {'positive': 1, 'negative': 2}[overall_sentiment]  # 正类为1，负类为2
            if 'quads' in sample and sample['quads']:
                quad = sample['quads'] # 在split_data中已经处理了四元组
                if quad[2] in CATEGORY2ID:  # Check if category is valid
                    texts.append(text)
                    citation_sentiment_labels.append(sentiment_label)
                    aspects.append(quad[0])  # aspect
                    opinions.append(quad[1])  # opinion
                    # 将方面类别字符串转换为对应的ID
                    aspect_categories.append(CATEGORY2ID[quad[2]])  # 使用映射字典转换类别标签
                    aspect_sentiments.append(1 if quad[3] == 'positive' else 0)  # sentiment
            else:
                print(f"Warning: No quadruples found for sample: {sample}")

        # 打乱数据顺序
        combined = list(
            zip(texts, citation_sentiment_labels, aspects, opinions, aspect_categories, aspect_sentiments))
        random.shuffle(combined)
        texts, citation_sentiment_labels, aspects, opinions, aspect_categories, aspect_sentiments = zip(*combined)

        # 创建数据集
        datasets[split_name] = ASQPDataset(
            texts=list(texts),
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
                'quads': item['sentiment_quadruples']  # 加载verified的四元组
            })

    # 加载中性样本
    df = pd.read_csv(corpus_file)
    neutral_samples = df[df['Sentiment'] == 'Neutral']['Text'].tolist()

    return pos_neg_samples, neutral_samples


def train_model(args, train_dataset, eval_dataset):
    device = args.device
    config = ASQPFinetuningConfig(
        backbone_model=args.model_name,
        aspect_category_labels=len(CATEGORY2ID),
        loss_type='focal_loss',
        focal_alpha=0.25,
        focal_gamma=2.0,
        loss_weight={'aspect_category': 1.0, 'aspect_sentiment': 1.0},
        label_smoothing=0.0
    )
    model = ASQPFinetuningModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")

    training_args = TrainingArguments(
        report_to='none',
        output_dir=f'./results/{args.model_name}',
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='steps',
        eval_steps=50,
        logging_strategy='steps',
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='aspect_category_f1',
        greater_is_better=True,
        seed=args.seed,
        bf16=True
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model.config),
        callbacks=[LossRecorderCallback()]
    )
    trainer.train()
    return model

class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.train_metrics = {
            'steps': [],
            'aspect_category_loss': [],
            'aspect_sentiment_loss': [],
            'total_loss': []
        }
        self.eval_metrics = {
            'steps': [],
            'eval_aspect_category_loss': [],
            'eval_aspect_sentiment_loss': [],
            'eval_total_loss': [],
            'aspect_category_f1': [],
            'aspect_category_accuracy': [],
            'aspect_sentiment_f1': [],
            'aspect_sentiment_accuracy': []
        }
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:  # one process will be True
            if all(key in logs for key in ['loss', 'aspect_category_loss', 'aspect_sentiment_loss']):
                if not any(key.startswith('eval_') for key in logs.keys()):
                    step = state.global_step
                    if not self.train_metrics['steps'] or step > self.train_metrics['steps'][-1]:
                        self.train_metrics['steps'].append(step)
                        self.train_metrics['aspect_category_loss'].append(logs['aspect_category_loss'])
                        self.train_metrics['aspect_sentiment_loss'].append(logs['aspect_sentiment_loss'])
                        self.train_metrics['total_loss'].append(logs['loss'])

            if 'eval_loss' in logs:
                step = state.global_step
                if not self.eval_metrics['steps'] or step > self.eval_metrics['steps'][-1]:
                    self.eval_metrics['steps'].append(step)
                    self.eval_metrics['eval_aspect_category_loss'].append(logs.get('eval_aspect_category_loss', 0))
                    self.eval_metrics['eval_aspect_sentiment_loss'].append(logs.get('eval_aspect_sentiment_loss', 0))
                    self.eval_metrics['eval_total_loss'].append(logs['eval_loss'])
                    self.eval_metrics['aspect_category_f1'].append(logs.get('eval_aspect_category_f1', 0))
                    self.eval_metrics['aspect_category_accuracy'].append(logs.get('eval_aspect_category_accuracy', 0))
                    self.eval_metrics['aspect_sentiment_f1'].append(logs.get('eval_aspect_sentiment_f1', 0))
                    self.eval_metrics['aspect_sentiment_accuracy'].append(logs.get('eval_aspect_sentiment_accuracy', 0)) # 要加上eval_前缀

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        self._plot_metrics()

    def _plot_metrics(self):
        self.ax1.clear()
        self.ax2.clear()

        # 绘制训练和验证损失曲线
        self.ax1.plot(self.train_metrics['steps'], self.train_metrics['aspect_category_loss'],
                      label='Aspect Category Loss (Train)')
        self.ax1.plot(self.train_metrics['steps'], self.train_metrics['aspect_sentiment_loss'],
                      label='Aspect Sentiment Loss (Train)')
        self.ax1.plot(self.train_metrics['steps'], self.train_metrics['total_loss'],
                      label='Total Loss (Train)')
        self.ax1.plot(self.eval_metrics['steps'], self.eval_metrics['eval_aspect_category_loss'],
                      label='Aspect Category Loss (Eval)')
        self.ax1.plot(self.eval_metrics['steps'], self.eval_metrics['eval_aspect_sentiment_loss'],
                      label='Aspect Sentiment Loss (Eval)')
        self.ax1.plot(self.eval_metrics['steps'], self.eval_metrics['eval_total_loss'],
                      label='Total Loss (Eval)')
        self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training and Evaluation Losses')
        self.ax1.grid(True)
        self.ax1.legend()

        # 绘制验证指标曲线
        self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['aspect_category_f1'],
                      label='Aspect Category F1')
        self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['aspect_category_accuracy'],
                      label='Aspect Category Accuracy')
        self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['aspect_sentiment_f1'],
                      label='Aspect Sentiment F1')
        self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['aspect_sentiment_accuracy'],
                      label='Aspect Sentiment Accuracy')
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Score')
        self.ax2.set_title('Evaluation Metrics')
        self.ax2.grid(True)
        self.ax2.legend()

        plt.tight_layout()
        plt.show()


def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32)
    all_preds_category = []
    all_labels_category = []
    all_preds_sentiment = []
    all_labels_sentiment = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs and labels to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels_category = batch['aspect_category_labels'].to(model.device)
            labels_sentiment = batch['aspect_sentiment_labels'].to(model.device)
            # Model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_category, logits_sentiment = outputs['logits']
            preds_category = torch.argmax(logits_category, dim=1)
            preds_sentiment = torch.argmax(logits_sentiment, dim=1)
            # Collect predictions and labels
            all_preds_category.extend(preds_category.cpu().numpy())
            all_labels_category.extend(labels_category.cpu().numpy())
            all_preds_sentiment.extend(preds_sentiment.cpu().numpy())
            all_labels_sentiment.extend(labels_sentiment.cpu().numpy())

    # Classification reports
    report_category = classification_report(
        all_labels_category,
        all_preds_category,
        target_names=list(CATEGORY2ID.keys())
    )
    report_sentiment = classification_report(
        all_labels_sentiment,
        all_preds_sentiment,
        target_names=['Negative', 'Positive']
    )
    print("Aspect Category Classification Report:\n", report_category)
    print("Aspect Sentiment Classification Report:\n", report_sentiment)

    # Plot confusion matrices
    plot_confusion_matrix(
        all_labels_category,
        all_preds_category,
        labels=list(CATEGORY2ID.keys()),
        title='Aspect Category Confusion Matrix'
    )
    plot_confusion_matrix(
        all_labels_sentiment,
        all_preds_sentiment,
        labels=['Negative', 'Positive'],
        title='Aspect Sentiment Confusion Matrix'
    )


def main(args):
    # Load data and prepare datasets
    quad_file = f'../output/sentiment_asqp_results_corpus_expand_llama.json'
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
    split_data = splitter.split_data(stratify_by_categories=True)

    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")
    train_dataset, val_dataset, test_dataset = create_datasets(split_data, tokenizer)

    # Train the model
    model = train_model(args, train_dataset, val_dataset)
    # Evaluate the model on the test set
    evaluate_model(model, test_dataset)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用Cudnn加速，保证结果可复现

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="scibert_scivocab_uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)