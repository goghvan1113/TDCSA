import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random
import numpy as np
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from collections import Counter


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sentiment_lexicon = sentiment_lexicon

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
        # 应用自定义的attention
        weighted_output = self.apply_sentiment_attention(pooled_output, input_ids)

        pooled_output = self.dropout(weighted_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return logits, loss

    def apply_sentiment_attention(self, sequence_output, input_ids):
        # 创建一个与input_ids相同形状的attention权重矩阵
        attention_weights = torch.ones_like(input_ids, dtype=torch.float)

        # 为情感词典中的词赋予额外的权重
        for i, sent in enumerate(input_ids):
            for j, token_id in enumerate(sent):
                token = self.roberta.tokenizer.convert_ids_to_tokens(token_id.item())
                if token in self.sentiment_lexicon:
                    attention_weights[i, j] = 2.0  # 你可以调整这个权重

        # 将权重应用到sequence_output上
        weighted_output = sequence_output * attention_weights.unsqueeze(-1)

        # 对加权后的输出进行池化
        pooled_output = torch.mean(weighted_output, dim=1)

        return pooled_output


class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, accumulation_steps, learning_rate, warmup_steps):
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

def load_sentiment_datasets():
    df = pd.read_csv(f'./data/citation_sentiment_corpus_new.csv')
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['Citation_Text'].tolist(), df['Sentiment'].tolist(), test_size=0.4, stratify=df['Sentiment'], random_state=42)
    return train_texts, train_labels, test_texts, test_labels

def slm_train(args):
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = CustomModel(args.model_name, num_labels=3).to(device)

    train_texts, train_labels, test_texts, test_labels = load_sentiment_datasets()
    train_data = MyDataset(tokenizer(train_texts, truncation=True, padding=True), train_labels)
    test_data = MyDataset(tokenizer(test_texts, truncation=True, padding=True), test_labels)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler("cosine",
                              optimizer=optimizer,
                              num_warmup_steps=args.warmup_steps,
                              num_training_steps=len(train_loader) * args.epochs)

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler,
                      device, args.accumulation_steps, args.learning_rate, args.warmup_steps)
    metrics = trainer.train(args.epochs)
    return metrics

def get_sentiment_words(text, sentiment, tokenizer, model, device=None):
    """使用Qwen-2.5模型提取情感词"""
    system_prompt = "You are an expert in scientific citation sentiment analysis. Your task is to extract sentiment words from given text. Please provide only the list of words or phrases, separated by commas."
    user_prompt = f"""
        Given the following scientific citation, please list up to 3 words or short phrases that express {sentiment} sentiment: 
        Citation: "{text}"
    """

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=30,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(
            model_input.input_ids, generated_ids)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 提取模型回复中的词列表
    words = generated_text.split("\n")[-1].strip().split(", ")
    return words

def create_sentiment_lexicon(path, top_n=20, tokenizer=None, model=None, device=None):
    """从CSV数据集创建情感词典"""
    df = pd.read_csv(path)
    
    positive_words = []
    negative_words = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Citation_Text']
        sentiment = row['Sentiment']
        
        if sentiment == 'p':  # 正面情感
            positive_words.extend(get_sentiment_words(text, 'positive', tokenizer, model))
        elif sentiment == 'n':  # 负面情感
            negative_words.extend(get_sentiment_words(text, 'negative', tokenizer, model))
        # 忽略中性（0）样本

    # 计算词频并选择最常见的词，选择前N个最常见的词
    positive_counter = Counter(positive_words)
    negative_counter = Counter(negative_words)
    top_positive = [word for word, _ in positive_counter.most_common(top_n)]
    top_negative = [word for word, _ in negative_counter.most_common(top_n)]

    return {'positive': top_positive, 'negative': top_negative}

def llm_generate(args):
    """使用Qwen-2.5模型生成具有特定情感的科学引文文本"""
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_dir, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=device)

    csv_path = './data/citation_sentiment_corpus.csv'
    sentiment_lexicon = create_sentiment_lexicon(csv_path, top_n=20, tokenizer=tokenizer, model=model, device=device)

    with open('output/scientific_citation_sentiment_lexicon.json', 'w') as f:
        json.dump(sentiment_lexicon, f, indent=2)

    return sentiment_lexicon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行解析对象
    parser.add_argument('--model_name', type=str, default='roberta-llama3.1405B-twitter-sentiment',help='Model name or path')  # 添加命令行参数
    parser.add_argument(('--llm_model_dir'), type=str, default='Qwen2.5-32B-Instruct-GPTQ-Int4', help='LLM model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')

    args = parser.parse_args()  # 解析命令行参数
    args.model_name = "pretrain_models/" + args.model_name
    args.llm_model_dir = "pretrain_models/" + args.llm_model_dir

    seed_everything(args.seed)
    sentiment_lexicon=llm_generate(args)
    # metrics = slm_train(args)

