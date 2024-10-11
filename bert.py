import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random
import numpy as np
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class CustomModel(torch.nn.Module):

    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size,
                                          num_labels)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:,
                                                  0, :]  # Use the CLS token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return logits, loss


class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 device, accumulation_steps):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_accuracies = []

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            self.optimizer.zero_grad()
            for step, batch in enumerate(tqdm(self.train_loader)):
                inputs = {
                    key: val.to(self.device).squeeze()
                    for key, val in batch.items()
                }
                logits, loss = self.model(**inputs)
                total_loss += loss.item()

                loss = loss / self.accumulation_steps
                loss.backward()

                if (step + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                print("loss: ", loss.item() * self.accumulation_steps)

            avg_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            val_metrics = self.evaluate()
            self.val_losses.append(val_metrics['loss'])
            self.val_f1_scores.append(val_metrics['f1'])
            self.val_accuracies.append(val_metrics['accuracy'])

        self.plot_metrics()
        return self.evaluate()

    def evaluate(self):
        self.model.eval()
        preds, labels = [], []
        scores = []
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {
                    key: val.to(self.device).squeeze()
                    for key, val in batch.items()
                }
                logits, loss = self.model(**inputs)
                total_loss += loss.item()
                scores.extend(
                    logits.softmax(dim=-1)[:, 1].detach().cpu().tolist())
                preds.extend(
                    torch.argmax(logits, dim=-1).detach().cpu().tolist())
                labels.extend(inputs['labels'].detach().cpu().tolist())

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        avg_loss = total_loss / len(self.val_loader)
        json.dump(
            scores,
            open(f"output/{args.model_name.split('/')[-1]}_scores.json", 'w'))
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss
        }

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.val_f1_scores, label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Validation F1 Score')

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')

        plt.tight_layout()
        plt.show()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 启用Cudnn


def load_sentiment_datasets():
    df = pd.read_csv(f'./data/citation_sentiment_corpus_new.csv')
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['Citation_Text'].tolist(), df['Sentiment'].tolist(), test_size=0.4, stratify=df['Sentiment'], random_state=42)
    return train_texts, train_labels, test_texts, test_labels

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

    train_citations, train_intents = read_jsonl('./data/controllable-citation-generation/train.jsonl')
    test_citations, test_intents = read_jsonl('./data/controllable-citation-generation/test.jsonl')
    return train_citations, train_intents, test_citations, test_intents

def main(args):
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = CustomModel(args.model_name, num_labels=3).to(device)

    train_texts, train_labels, test_texts, test_labels = load_intent_datasets()
    train_data = MyDataset(tokenizer(train_texts, truncation=True, padding=True), train_labels)
    test_data = MyDataset(tokenizer(test_texts, truncation=True, padding=True), test_labels)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler("cosine",
                              optimizer=optimizer,
                              num_warmup_steps=0,
                              num_training_steps=len(train_loader) * args.epochs)

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler,
                      device, args.accumulation_steps)
    metrics = trainer.train(args.epochs)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 创建命令行解析对象
    parser.add_argument('--model_name',
                        type=str,
                        default='roberta-llama3.1405B-twitter-sentiment',
                        help='Model name or path') # 添加命令行参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of epochs')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    args = parser.parse_args() # 解析命令行参数
    args.model_name = "pretrain_models/" + args.model_name

    seed = 42
    seed_everything(seed)
    metrics = main(args)
    metrics["seed"] = seed
    metrics["model_name"] = args.model_name.split("/")[-1]
    if os.path.exists("output/bert_results.csv"):
        df = pd.read_csv("output/bert_results.csv")
    else:
        df = pd.DataFrame(columns = metrics.keys())
    df = df._append(metrics, ignore_index=True)
    df.to_csv("output/bert_results.csv", index=False)










