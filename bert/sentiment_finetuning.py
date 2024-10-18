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
    test_size = 0.2
    id2label={0:"Neutral", 1:"Positive", 2:"Negative"}
    label2id={"Neutral":0, "Positive":1, "Negative":2}

    device = torch.device(args.device)
    model_dir = f"../pretrain_models/{args.model_name}"
    # model_dir = f"../citation_finetuned_models/{args.model_name}_cpt"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = CustomBERTModel.from_pretrained(model_dir, num_labels=num_labels, id2label=id2label, label2id=label2id).to(device)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_sentiment_datasets(test_size=test_size, seed=args.seed, filepath='../data/corpus.txt')
    train_data = SentimentDataset(tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt', max_length=512), train_labels)
    val_data = SentimentDataset(tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt', max_length=512), val_labels)
    test_data = SentimentDataset(tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt', max_length=512), test_labels)
    
    print(f"Train Dataset Size: {len(train_data)}")
    print(f"Test Dataset Size: {len(test_data)}")
    print(f"Val Dataset Size: {len(val_data)}")

    loss_recorder = LossRecorderCallback()
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
        disable_tqdm=False,
        fp16= True, # faster and use less memory
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
    loss_recorder.plot_and_save_metrics(f'../output')

    # Evaluate on the test set
    test_result = trainer.predict(test_data)
    test_preds = test_result.predictions.argmax(-1)
    test_labels = test_data.labels
    plot_roc_curve(test_labels, test_result.predictions)
    plot_pr_curve(test_labels, test_result.predictions)
    plot_confusion_matrix(test_labels, test_preds, list(label2id.keys()))

    results = {
        'cls': 'sentiment',
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
        'loss_type': args.loss_type,
        'lr_scheduler_type': args.lr_scheduler_type,
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

    mytest(args, trainer, tokenizer)


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
        self.train_losses = []
        self.eval_losses = []
        self.f1_scores = []
        self.accuracies = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
        if 'eval_F1' in logs:
            self.f1_scores.append(logs['eval_F1'])
        if 'eval_Accuracy' in logs:
            self.accuracies.append(logs['eval_Accuracy'])

    def plot_and_save_metrics(self, output_dir):
        min_length = min(len(self.train_losses), len(self.eval_losses))
        steps = range(min_length)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(steps, self.train_losses[:min_length], label='Training Loss')
        plt.plot(steps, self.eval_losses[:min_length], label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Plot F1 Score
        plt.subplot(2, 2, 2)
        plt.plot(steps, self.f1_scores[:min_length], label='F1 Score')
        plt.xlabel('Steps')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('F1 Score')

        # Plot Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(steps, self.accuracies[:min_length], label='Accuracy')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')

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

def load_sentiment_datasets(test_size=0.4, seed=42, filepath='../data/corpus.txt', is_spilit=True):
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
    if is_spilit:
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(sentences,
                                                                              labels, test_size=test_size,
                                                                              stratify=labels, random_state=seed)
        val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5,
                                                                          stratify=temp_labels, random_state=seed)
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    else:
        return sentences, labels

def mytest(args, trainer, tokenizer):
    # model_dir = f'../citation_finetuned_models/{args.model_name}'
    # trainer.save_model(model_dir)

    test_texts, test_labels, _, _, _, _, = load_sentiment_datasets(test_size=0.1, seed=args.seed, filepath='../data/citation_sentiment_corpus.csv')
    test_dataset = SentimentDataset(tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt', max_length=512),
                             test_labels)
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)

    # Compute metrics
    accuracy = accuracy_score(test_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average='macro')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    label2id = {"Neutral": 0, "Positive": 1, "Negative": 2}
    plot_confusion_matrix(test_labels, preds, list(label2id.keys()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建命令行解析对象
    parser.add_argument('--model_name', type=str, default='roberta-llama3.1405B-twitter-sentiment',help='Model name or path')  # 添加命令行参数
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
    args = parser.parse_args()  # 解析命令行参数

    seed_everything(args.seed)
    main(args)