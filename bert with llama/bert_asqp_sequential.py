# ...existing code...
import argparse
import json
import random
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig,
    AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
)
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

# 定义类别映射
CATEGORY2ID = {
    'METHODOLOGY': 0,
    'PERFORMANCE': 1,
    'INNOVATION': 2,
    'APPLICABILITY': 3,
    'LIMITATION': 4,
    'COMPARISON': 5
}
ID2CATEGORY = {v: k for k, v in CATEGORY2ID.items()}

class ASQPSequentialConfig(PretrainedConfig):
    """配置类 - 主观性、情感和四元组分类多任务模型"""
    def __init__(
            self,
            subjectivity_labels: int = 2,  # 主观性标签数（客观、主观）
            sentiment_labels: int = 2,     # 情感标签数（正、负）
            aspect_category_labels: int = 6,  # 方面类别标签数
            loss_weights: dict = {
                "subjectivity": 1.0,
                "sentiment": 1.0,
                "aspect_category": 1.0
            },
            backbone_model: str = "scibert-scivocab-uncased",
            dropout_prob: float = 0.1,
            label_smoothing: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        backbone_config = AutoConfig.from_pretrained(f'../pretrain_models/{backbone_model}')
        self.hidden_size = backbone_config.hidden_size
        self.subjectivity_labels = subjectivity_labels
        self.sentiment_labels = sentiment_labels
        self.aspect_category_labels = aspect_category_labels
        self.loss_weights = loss_weights
        self.backbone_model = backbone_model
        self.dropout_prob = dropout_prob
        self.label_smoothing = label_smoothing

class ASQPSequentialModel(PreTrainedModel):
    config_class = ASQPSequentialConfig

    def __init__(self, config: ASQPSequentialConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(f'../pretrain_models/{config.backbone_model}')
        self.dropout = nn.Dropout(config.dropout_prob)

        # 主观性分类器
        self.subjectivity_classifier = nn.Linear(config.hidden_size, config.subjectivity_labels)
        # 情感分类器
        self.sentiment_classifier = nn.Linear(config.hidden_size, config.sentiment_labels)
        # 方面类别分类器
        self.aspect_category_classifier = nn.Linear(config.hidden_size, config.aspect_category_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            subjectivity_labels=None,
            sentiment_labels=None,
            aspect_category_labels=None
    ):
        outputs = self.encoder(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 主观性分类
        subjectivity_logits = self.subjectivity_classifier(pooled_output)

        loss = 0.0
        outputs = {'subjectivity_logits': subjectivity_logits}

        if subjectivity_labels is not None:
            subjectivity_loss = F.cross_entropy(
                subjectivity_logits,
                subjectivity_labels,
                label_smoothing=self.config.label_smoothing
            )
            loss += self.config.loss_weights['subjectivity'] * subjectivity_loss
            outputs['subjectivity_loss'] = subjectivity_loss

        # 获取预测为主观的样本索引
        with torch.no_grad():
            subjectivity_preds = torch.argmax(subjectivity_logits, dim=1)
        subjective_indices = (subjectivity_preds == 1)  # 假设标签1表示主观

        if subjective_indices.any():
            subjective_outputs = pooled_output[subjective_indices]

            # 情感分类
            sentiment_logits = self.sentiment_classifier(subjective_outputs)
            outputs['sentiment_logits'] = sentiment_logits

            if sentiment_labels is not None:
                sentiment_loss = F.cross_entropy(
                    sentiment_logits,
                    sentiment_labels[subjective_indices],
                    label_smoothing=self.config.label_smoothing
                )
                loss += self.config.loss_weights['sentiment'] * sentiment_loss
                outputs['sentiment_loss'] = sentiment_loss

            # 方面类别分类
            aspect_category_logits = self.aspect_category_classifier(subjective_outputs)
            outputs['aspect_category_logits'] = aspect_category_logits

            if aspect_category_labels is not None:
                aspect_category_loss = F.cross_entropy(
                    aspect_category_logits,
                    aspect_category_labels[subjective_indices],
                    label_smoothing=self.config.label_smoothing
                )
                loss += self.config.loss_weights['aspect_category'] * aspect_category_loss
                outputs['aspect_category_loss'] = aspect_category_loss

        outputs['loss'] = loss if loss > 0 else None
        return outputs

# 定义数据集类
class ASTECitationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            texts: List[str],
            subjectivity_labels: List[int],
            sentiment_labels: List[Optional[int]],
            aspect_categories: List[Optional[int]],
            tokenizer=None,
            max_length: int = 512
    ):
        self.texts = texts
        self.subjectivity_labels = subjectivity_labels
        self.sentiment_labels = sentiment_labels
        self.aspect_categories = aspect_categories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'subjectivity_labels': torch.tensor(self.subjectivity_labels[idx])
        }

        # 如果是主观样本，添加情感和方面类别标签
        if self.sentiment_labels[idx] is not None:
            item['sentiment_labels'] = torch.tensor(self.sentiment_labels[idx])
        else:
            item['sentiment_labels'] = torch.tensor(-1)  # 使用 -1 表示无效标签

        if self.aspect_categories[idx] is not None:
            item['aspect_category_labels'] = torch.tensor(self.aspect_categories[idx])
        else:
            item['aspect_category_labels'] = torch.tensor(-1)  # 使用 -1 表示无效标签

        return item

def load_acos_data(quad_file=None, corpus_file=None):
    """Load quadruple data and neutral samples."""
    pos_neg_samples = []
    with open(quad_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            pos_neg_samples.append({
                'text': item['text'],
                'overall_sentiment': item['overall_sentiment'],
                'quads': item['final_quadruples']
            })

    # Load neutral samples
    df = pd.read_csv(corpus_file)
    neutral_samples = df[df['Sentiment'] == 'Neutral']['Text'].tolist()

    return pos_neg_samples, neutral_samples

def split_dataset(pos_neg_samples, neutral_samples, test_size=0.2, val_size=0.1, random_state=42):
    """Split the dataset into training, validation, and test sets."""
    # Split positive and negative samples
    pos_neg_labels = [s['overall_sentiment'] for s in pos_neg_samples]
    pos_neg_train_val, pos_neg_test = train_test_split(
        pos_neg_samples,
        test_size=test_size,
        stratify=pos_neg_labels,
        random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    pos_neg_train, pos_neg_val = train_test_split(
        pos_neg_train_val,
        test_size=val_ratio,
        stratify=[s['overall_sentiment'] for s in pos_neg_train_val],
        random_state=random_state
    )

    # Split neutral samples
    neutral_train_val, neutral_test = train_test_split(
        neutral_samples,
        test_size=test_size,
        random_state=random_state
    )
    neutral_train, neutral_val = train_test_split(
        neutral_train_val,
        test_size=val_ratio,
        random_state=random_state
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

def create_datasets(split_data, tokenizer):
    """Create datasets for training, validation, and testing."""
    datasets = {}
    for split_name in ['train', 'val', 'test']:
        pos_neg_samples = split_data[split_name]['pos_neg_samples']
        neutral_texts = split_data[split_name]['neutral_samples']

        texts = []
        subjectivity_labels = []
        sentiment_labels = []
        aspect_categories = []

        # Process positive and negative samples
        for sample in pos_neg_samples:
            text = sample['text']
            texts.append(text)
            subjectivity_labels.append(1)  # Subjective
            sentiment_labels.append(1 if sample['overall_sentiment'] == 'positive' else 0)
            # Extract aspect categories from quadruples
            if sample.get('quads'):
                aspect_category_ids = [CATEGORY2ID[quad[2]] for quad in sample['quads'] if quad[2] in CATEGORY2ID]
                aspect_categories.append(aspect_category_ids[0] if aspect_category_ids else -1)
            else:
                aspect_categories.append(-1)

        # Process neutral samples
        for text in neutral_texts:
            texts.append(text)
            subjectivity_labels.append(0)  # Objective
            sentiment_labels.append(-1)    # Invalid label
            aspect_categories.append(-1)   # Invalid label

        # Shuffle data
        combined = list(zip(texts, subjectivity_labels, sentiment_labels, aspect_categories))
        random.shuffle(combined)
        texts, subjectivity_labels, sentiment_labels, aspect_categories = zip(*combined)

        # Create dataset
        datasets[split_name] = ASTECitationDataset(
            texts=list(texts),
            subjectivity_labels=list(subjectivity_labels),
            sentiment_labels=list(sentiment_labels),
            aspect_categories=list(aspect_categories),
            tokenizer=tokenizer
        )
    return datasets['train'], datasets['val'], datasets['test']

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    subjectivity_logits = logits['subjectivity_logits']
    sentiment_logits = logits.get('sentiment_logits')
    aspect_category_logits = logits.get('aspect_category_logits')

    metrics = {}

    # Subjectivity metrics
    subjectivity_preds = np.argmax(subjectivity_logits, axis=1)
    subjectivity_labels = labels['subjectivity_labels']
    subjectivity_report = classification_report(
        subjectivity_labels,
        subjectivity_preds,
        target_names=['Objective', 'Subjective'],
        output_dict=True,
        zero_division=0
    )
    metrics['subjectivity_f1'] = subjectivity_report['macro avg']['f1-score']
    metrics['subjectivity_accuracy'] = subjectivity_report['accuracy']

    # Sentiment metrics
    if sentiment_logits is not None:
        sentiment_labels = labels['sentiment_labels']
        valid_indices = sentiment_labels != -1
        if np.any(valid_indices):
            sentiment_preds = np.argmax(sentiment_logits, axis=1)
            sentiment_report = classification_report(
                sentiment_labels[valid_indices],
                sentiment_preds,
                target_names=['Negative', 'Positive'],
                output_dict=True,
                zero_division=0
            )
            metrics['sentiment_f1'] = sentiment_report['macro avg']['f1-score']
            metrics['sentiment_accuracy'] = sentiment_report['accuracy']

    # Aspect category metrics
    if aspect_category_logits is not None:
        aspect_category_labels = labels['aspect_category_labels']
        valid_indices = aspect_category_labels != -1
        if np.any(valid_indices):
            aspect_category_preds = np.argmax(aspect_category_logits[valid_indices], axis=1)
            aspect_category_report = classification_report(
                aspect_category_labels[valid_indices],
                aspect_category_preds,
                target_names=list(CATEGORY2ID.keys()),
                output_dict=True,
                zero_division=0
            )
            metrics['aspect_category_f1'] = aspect_category_report['macro avg']['f1-score']
            metrics['aspect_category_accuracy'] = aspect_category_report['accuracy']

    return metrics

def train_model(args, train_dataset, val_dataset):
    device = args.device
    config = ASQPSequentialConfig(
        backbone_model=args.model_name,
        label_smoothing=0.1,
        loss_weights={"subjectivity": 1.0, "sentiment": 1.0, "aspect_category": 1.0}
    )
    model = ASQPSequentialModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")

    training_args = TrainingArguments(
        seed=args.seed,
        report_to='none',
        output_dir=f'./results/{args.model_name}',
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy='steps',
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='subjectivity_f1',
        greater_is_better=True
    )

    def data_collator(features):
        batch = DataCollatorWithPadding(tokenizer)(features)
        for key in ['subjectivity_labels', 'sentiment_labels', 'aspect_category_labels']:
            if key in batch:
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    # trainer.train()
    return model

def evaluate_model(model, dataset, tokenizer):
    device = next(model.parameters()).device
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=DataCollatorWithPadding(tokenizer)
    )
    all_subjectivity_preds = []
    all_subjectivity_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_aspect_category_preds = []
    all_aspect_category_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # Subjectivity
            subjectivity_preds = torch.argmax(outputs['subjectivity_logits'], dim=1)
            all_subjectivity_preds.extend(subjectivity_preds.cpu().numpy())
            all_subjectivity_labels.extend(batch['subjectivity_labels'].cpu().numpy())

            # Determine subjective samples
            subjective_indices = (subjectivity_preds == 1)

            if subjective_indices.any():
                # Sentiment
                sentiment_logits = outputs['sentiment_logits']
                sentiment_preds = torch.argmax(sentiment_logits, dim=1)
                sentiment_labels = batch['sentiment_labels'][subjective_indices]
                all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Aspect Category
                aspect_category_logits = outputs['aspect_category_logits']
                aspect_category_preds = torch.argmax(aspect_category_logits, dim=1)
                aspect_category_labels = batch['aspect_category_labels'][subjective_indices]
                all_aspect_category_preds.extend(aspect_category_preds.cpu().numpy())
                all_aspect_category_labels.extend(aspect_category_labels.cpu().numpy())

    # Subjectivity report
    subjectivity_report = classification_report(
        all_subjectivity_labels,
        all_subjectivity_preds,
        target_names=['Objective', 'Subjective']
    )
    print("\nSubjectivity Classification Report:")
    print(subjectivity_report)

    # Sentiment report
    if all_sentiment_labels:
        sentiment_report = classification_report(
            all_sentiment_labels,
            all_sentiment_preds,
            target_names=['Negative', 'Positive']
        )
        print("\nSentiment Classification Report:")
        print(sentiment_report)

    # Aspect category report
    if all_aspect_category_labels:
        aspect_category_report = classification_report(
            all_aspect_category_labels,
            all_aspect_category_preds,
            target_names=list(CATEGORY2ID.keys())
        )
        print("\nAspect Category Classification Report:")
        print(aspect_category_report)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    quad_file = '../output/sentiment_asqp_results_corpus_expand_verified.json'
    corpus_file = '../data/citation_sentiment_corpus_expand.csv'
    pos_neg_samples, neutral_samples = load_acos_data(quad_file, corpus_file)

    split_data = split_dataset(pos_neg_samples, neutral_samples, test_size=0.2, val_size=0.1, random_state=args.seed)

    # Print dataset statistics
    for split_name, split in split_data.items():
        pos_neg_dist = Counter(s['overall_sentiment'] for s in split['pos_neg_samples'])
        print(f"\n{split_name} set distribution:")
        print(f"Positive: {pos_neg_dist.get('positive', 0)}")
        print(f"Negative: {pos_neg_dist.get('negative', 0)}")
        print(f"Neutral: {len(split['neutral_samples'])}")

    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")
    train_dataset, val_dataset, test_dataset = create_datasets(split_data, tokenizer)

    model = train_model(args, train_dataset, val_dataset)

    print("\nValidation Set Results:")
    evaluate_model(model, val_dataset, tokenizer)

    print("\nTest Set Results:")
    evaluate_model(model, test_dataset, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="scibert_scivocab_uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)

