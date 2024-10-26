import os
import time
import json
import torch
import wandb
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import TrainerCallback
from huggingface_hub import notebook_login
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments

from custom_loss import MultiFocalLoss, MultiDSCLoss, AsymmetricLoss
from plot_results import plot_roc_curve, plot_pr_curve, plot_confusion_matrix


def main(args):
    num_labels = 3
    label2id = {"background": 0, "method": 1, "result": 2}
    id2label = {0: 'background', 1: 'method', 2: 'result'}

    device = torch.device(args.device)
    model_dir = f"../pretrain_models/{args.model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = CustomBERTModel.from_pretrained(model_dir, num_labels=num_labels, id2label=id2label, label2id=label2id).to(
        device)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_intent_datasets()
    train_data = SentimentDataset(
        tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt', max_length=512), train_labels)
    val_data = SentimentDataset(
        tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt', max_length=512), val_labels)
    test_data = SentimentDataset(
        tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt', max_length=512), test_labels)

    print(f"Train Dataset Size: {len(train_data)}")
    print(f"Test Dataset Size: {len(test_data)}")
    print(f"Val Dataset Size: {len(val_data)}")

    loss_recorder = LossRecorderCallback()
    # 定义训练参数
    training_args = TrainingArguments(
        seed=args.seed,
        report_to='none',
        output_dir=f'./results/{args.model_name}',  # 输出结果目录
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type='cosine',
        gradient_accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        # warmup_steps=args.warmup_steps,
        logging_strategy='steps',
        logging_dir=f'./logs/{args.model_name}',  # 日志目录
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        disable_tqdm=False,
        fp16=True,  # faster and use less memory
        metric_for_best_model='F1',
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=True,
        # push_to_hub=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        loss_type=args.loss_type,  # 自定义参数 focal_loss dsc_loss
        callbacks=[loss_recorder]
    )

    start = time.time()
    trainer.train()
    end = time.time()
    train_time = int(end - start)
    print(f"Training took: {train_time} seconds")
    eval_result = trainer.evaluate()

    # Evaluate on the test set
    test_result = trainer.predict(test_data)
    test_preds = test_result.predictions.argmax(-1)
    test_labels = test_data.labels
    # plot_roc_curve(test_labels, test_result.predictions)
    # plot_pr_curve(test_labels, test_result.predictions)
    plot_confusion_matrix(test_labels, test_preds, list(label2id.keys()))

    results = {
        'cls': 'intent',
        'seed': args.seed,
        'eval_result': eval_result,
        'train_time': train_time,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'accumulation_steps': args.accumulation_steps,
        'warmup_steps': args.warmup_steps,
        'warmup_ratio': args.warmup_ratio,
        'loss_type': args.loss_type
    }

    json_file_path = '../output/bert_training_details.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = []
    existing_results.append(results)
    with open(json_file_path, 'w') as f:
        json.dump(existing_results, f, indent=4)


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


class CustomBERTModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super(CustomBERTModel, self).__init__(config)
        self.bert = AutoModel.from_pretrained(config._name_or_path)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the CLS token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class CustomTrainer(Trainer):
    def __init__(self, loss_type='focal_loss', *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
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
            'f1_scores': [],
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
            self.eval_metrics['f1_scores'].append(logs['eval_F1'])
            self.eval_metrics['accuracy'].append(logs['eval_Accuracy'])

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
            self.ax2.plot(self.eval_metrics['steps'], self.eval_metrics['f1_scores'],
                          label='F1 Scores', color='lightgreen')

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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
    }

def load_intent_datasets():
    label2id = {"background": 0, "method": 1, "result": 2}
    id2label = {v: k for k, v in label2id.items()}

    def read_jsonl(file_path):
        citations = []
        intents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                citation = entry.get('citation')
                citation_intent = entry.get('citation_intent')
                if citation and citation_intent:
                    citation_intent_id = label2id.get(citation_intent, -1)  # Use -1 for unknown labels
                    citations.append(citation)
                    intents.append(citation_intent_id)
        return citations, intents

    train_citations, train_intents = read_jsonl('../data/controllable-citation-generation/train.jsonl')
    test_citations, test_intents = read_jsonl('../data/controllable-citation-generation/test.jsonl')
    val_citations, val_intents = read_jsonl('../data/controllable-citation-generation/val.jsonl')
    return train_citations, train_intents, val_citations, val_intents, test_citations, test_intents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建命令行解析对象
    parser.add_argument('--model_name', type=str, default='roberta-llama3.1405B-twitter-sentiment',
                        help='Model name or path')  # 添加命令行参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--loss_type', type=str, default='focal_loss', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()  # 解析命令行参数

    seed_everything(args.seed)
    main(args)