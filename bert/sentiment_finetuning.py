import os
import time
import json
from collections import Counter

import torch
from torch import nn

import wandb
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from transformers import TrainerCallback, TrainerState, TrainerControl, PreTrainedModel, PretrainedConfig, AutoConfig
from sklearn.model_selection import KFold
from huggingface_hub import notebook_login
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments

from bert.custom_loss import MultiFocalLoss, MultiDSCLoss, AsymmetricLoss
from bert.plot_results import plot_roc_curve, plot_pr_curve, plot_confusion_matrix


def main(args):
    id2label={0:"Neutral", 1:"Positive", 2:"Negative"}
    label2id={"Neutral":0, "Positive":1, "Negative":2}

    device = torch.device(args.device)
    filepath = '../data/citation_sentiment_corpus.csv'
    config = CustomBERTConfig(num_labels=3, backbone_model=args.model_name)
    model = CustomBERTModel(config).to(device)
    model_dir = f"../pretrain_models/{args.model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # print(model)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_sentiment_datasets(
        test_size=0.2,
        val_size=0.1,
        seed=args.seed,
        filepath=filepath
    )
    train_data = SentimentDataset(
        tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt', max_length=512),
        train_labels)
    val_data = SentimentDataset(
        tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt', max_length=512),
        val_labels)
    test_data = SentimentDataset(
        tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt', max_length=512),
        test_labels)

    # 定义训练参数
    training_args = TrainingArguments(
        seed=args.seed,
        report_to='none',
        output_dir=f'./results/{args.model_name}',          # 输出结果目录
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        # warmup_steps=args.warmup_steps,
        logging_strategy='steps',
        logging_dir=f'./logs/{args.model_name}',            # 日志目录
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        bf16=True, # faster and use less memory
        metric_for_best_model='F1_Macro',
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        loss_type=args.loss_type,  # 自定义参数 focal_loss dsc_loss
        callbacks=[LossRecorderCallback()]
    )

    start = time.time()
    trainer.train()
    end = time.time()
    train_time = int(end - start)
    print(f"Training took: {train_time} seconds")

    save_and_plot(trainer, val_data, test_data, label2id)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 启用Cudnn


class SentimentDataset(torch.utils.data.Dataset):
    """
    重构数据集类，使其能够返回字典格式的数据，有标签
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class CustomBERTConfig(PretrainedConfig):
    def __init__(self, num_labels=3, backbone_model: str="roberta-base", hidden_dropout_prob: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        backbone_config = AutoConfig.from_pretrained(f'../pretrain_models/{backbone_model}')
        self.num_labels = num_labels
        self.hidden_size = backbone_config.hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.backbone_model = backbone_model

class CustomBERTModel(PreTrainedModel):
    config_class = CustomBERTConfig
    def __init__(self, config: CustomBERTConfig):
        super().__init__(config)
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

        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = outputs.last_hidden_state
        text_pooled = text_hidden[:, 0, :]

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

        return {'logits': logits}


class CustomTrainer(Trainer):
    def __init__(self, loss_type='focal_loss', *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        if self.loss_type == 'focal_loss':
            loss_fct = MultiFocalLoss(num_class=3, alpha=0.8, gamma=2.0)
        elif self.loss_type == 'dsc_loss':
            loss_fct = MultiDSCLoss(alpha=1.0, smooth=1.0)
        elif self.loss_type == 'asymmetric_loss':
            loss_fct = AsymmetricLoss(gamma_pos=0.5, gamma_neg=3.0)
        elif self.loss_type == 'ce_loss':
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


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
            'accuracy': []
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

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
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

        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Score')
        self.ax2.set_title('Validation Metrics')
        self.ax2.grid(True)
        self.ax2.legend()

        plt.tight_layout()
        plt.show()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')

    return {
        'Accuracy': acc,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'F1_Macro': f1_macro
    }


def load_sentiment_datasets(test_size=0.2, val_size=0.1, seed=42, filepath='../data/corpus.txt', is_split=True):
    sentences, labels = [], []
    if filepath == '../data/citation_sentiment_corpus.csv':
        df = pd.read_csv(filepath)
        label_map = {'o': 0, 'p': 1, 'n': 2}
        df['Sentiment'] = df['Sentiment'].map(label_map)
        sentences = df['Citation_Text'].tolist()
        labels = df['Sentiment'].tolist()
    elif filepath == '../data/citation_sentiment_corpus_balanced.csv':
        df = pd.read_csv(filepath)
        df = df[(df['Source'] == 'new') & (df['Sentiment'].isin([1, 2])) | (df['Source'] == 'original') & (
                    df['Sentiment'] == 0)] # 只选取新数据集中的正负样本和原始数据集中的中性样本
        sentences = df['Citation_Text'].tolist()
        labels = df['Sentiment'].tolist()
    elif filepath == '../data/corpus.txt':
        with open(filepath, "r", encoding="utf8") as f:
            file = f.read().split("\n")
            file = [i.split("\t") for i in file]
            for i in file:
                if len(i) == 2:
                    sentence = i[1]
                    label = int(i[0])
                    # Map labels: 2 -> Positive, 1 -> Neutral, 0 -> Negative
                    if label == 2:
                        label = 1
                    elif label == 1:
                        label = 0
                    elif label == 0:
                        label = 2
                    sentences.append(sentence)
                    labels.append(label)
    elif filepath == '../data/CSA_raw_dataset/augmented_context_full/combined.csv':
        df = pd.read_csv(filepath)
        labelmap = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
        df['Sentiment'] = df['Sentiment'].map(labelmap)
        sentences = df['Text'].tolist()
        labels = df['Sentiment'].tolist()
    elif (filepath == '../data/citation_sentiment_corpus_expand.csv'
          or filepath == '../data/citation_sentiment_corpus_expand_athar.csv'):
        df = pd.read_csv(filepath)
        label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
        df['Sentiment'] = df['Sentiment'].map(label_map)
        sentences = df['Text'].tolist()
        labels = df['Sentiment'].tolist()

    if is_split:
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            sentences,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=seed)

        # df_aug = pd.read_csv('../data/train_data_aug3.csv')
        # train_texts = df_aug['Citation_Text'].tolist()
        # train_labels = df_aug['Sentiment'].tolist() # 替换增强后的整个数据集
        # train_texts, train_labels = shuffle(train_texts, train_labels, random_state=seed) # 打乱新的训练集

        val_ratio = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts,
            train_val_labels,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=seed)

        # Print label distribution
        print("Train set label distribution:", Counter(train_labels))
        print("Validation set label distribution:", Counter(val_labels))
        print("Test set label distribution:", Counter(test_labels))

        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    else:
        return sentences, labels

def save_and_plot(trainer, val_dataset, test_dataset, label2id, output_path='../output/bert_training_details.json'):
    # Evaluate on the validation set
    val_result = trainer.predict(val_dataset)
    val_preds = val_result.predictions.argmax(-1)
    val_labels = val_dataset.labels

    # Evaluate on the test set
    test_result = trainer.predict(test_dataset)
    test_preds = test_result.predictions.argmax(-1)
    test_labels = test_dataset.labels

    # plot_roc_curve(test_labels, test_result.predictions)
    # plot_pr_curve(test_labels, test_result.predictions)
    plot_confusion_matrix(test_labels, test_preds, list(label2id.keys()))

    # Generate classification reports
    val_report = classification_report(val_labels, val_preds, target_names=list(label2id.keys()), digits=4)
    print("\nValidation Set Results:")
    print(val_report)

    test_report = classification_report(test_labels, test_preds, target_names=list(label2id.keys()), digits=4)
    print("\nTest Set Results:")
    print(test_report)

    # Save results to JSON
    results = {
        'validation_report': val_report,
        'test_report': test_report,
        'eval_metrics': val_result.metrics,
        'test_metrics': test_result.metrics
    }

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    existing_results.append(results)

    with open(output_path, 'w') as f:
        json.dump(existing_results, f, indent=4)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()  # 创建命令行解析对象
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased',help='Model name or path')  # 添加命令行参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate') #2e-5
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate scheduler type')
    parser.add_argument('--loss_type', type=str, default='focal_loss', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay') # 0.01
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--validation_type', type=str, default='regular', choices=['regular', 'kfold'], help='Validation type: regular or kfold')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for k-fold cross-validation')
    args = parser.parse_args()  # 解析命令行参数

    seed_everything(args.seed)
    main(args)