from typing import List, Dict

import pandas as pd
import torch
from numpy.ma.core import negative
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np


class CitationSentimentClassifier:
    def __init__(self, model_name='roberta-llama3.1405B-twitter-sentiment', num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(f'../pretrain_models/{model_name}')
        # 初始化用于继续预训练的MLM模型
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(f'../pretrain_models/{model_name}')
        # 初始化用于分类的模型
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            f'../pretrain_models/{model_name}',
            num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)

    def continue_pretraining(self, critical_texts, batch_size=8, epochs=3):
        """使用负面情感数据集继续预训练"""

        critical_dataset = CitationDataset(critical_texts, np.ones(critical_texts), tokenizer)

        # 准备数据集
        dataset = Dataset.from_dict({'text': negative_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir="./roberta_continue_pretrain",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=2,
            report_to="none"
        )

        # 开始预训练
        trainer = Trainer(
            model=self.mlm_model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()

        # 更新分类器的权重
        self.classifier.roberta = self.mlm_model.roberta

    def knowledge_distillation(self, texts: List[str],
                               llm_predictions: np.ndarray,
                               temperature: float = 2.0,
                               epochs: int = 3,
                               batch_size: int = 8,
                               learning_rate: float = 2e-5) -> Dict[str, List[float]]:
        """
        从LLM蒸馏知识到分类器

        Args:
            texts: 训练文本列表
            llm_predictions: LLM生成的软标签，shape为(n_samples, num_labels)
            temperature: 蒸馏温度参数
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率

        Returns:
            包含训练历史的字典
        """
        self.classifier.train()
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=learning_rate)

        # 记录训练历史
        history = {
            'distillation_loss': [],
            'batch_loss': []
        }

        # 将LLM预测转换为torch tensor
        teacher_probs = torch.tensor(llm_predictions, dtype=torch.float32).to(self.device)

        def compute_kl_loss(student_logits, teacher_probs, temperature):
            """计算KL散度损失"""
            student_logits = student_logits / temperature
            student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)  # 确保teacher概率和为1

            loss = torch.nn.KLDivLoss(reduction='batchmean')(
                student_log_probs,
                teacher_probs
            ) * (temperature ** 2)
            return loss

        # 创建数据加载器
        dataset = Dataset.from_dict({
            'text': texts,
            'teacher_probs': llm_predictions.tolist()
        })

        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            probs = [item['teacher_probs'] for item in batch]

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            return {
                'inputs': inputs,
                'teacher_probs': torch.tensor(probs, dtype=torch.float32)
            }

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                # 将数据移到设备上
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                batch_teacher_probs = batch['teacher_probs'].to(self.device)

                # 前向传播
                outputs = self.classifier(**inputs)

                # 计算损失
                loss = compute_kl_loss(
                    outputs.logits,
                    batch_teacher_probs,
                    temperature
                )

                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录损失
                epoch_loss += loss.item()
                history['batch_loss'].append(loss.item())

            avg_epoch_loss = epoch_loss / len(dataloader)
            history['distillation_loss'].append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

        return history

    def generate_soft_labels(self, texts: List[str]) -> np.ndarray:
        """使用LLM为文本生成软标签"""
        llm_predictions = []
        for text in texts:
            # 调用LLM生成预测概率，例如通过API接口
            # probs = call_llm_api(text)
            # llm_predictions.append(probs)
            pass  # 实际实现中替换为真实的LLM调用
        return np.array(llm_predictions)

    def train_classifier(self, train_texts, train_labels, eval_texts=None, eval_labels=None):
        """训练三分类分类器"""
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True
        )
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })

        training_args = TrainingArguments(
            output_dir='./citation_sentiment_classifier',
            num_train_epochs=5,
            per_device_train_batch_size=8,
            evaluation_strategy="epoch" if eval_texts else "no",
        )

        if eval_texts:
            eval_encodings = self.tokenizer(
                eval_texts,
                truncation=True,
                padding=True
            )
            eval_dataset = Dataset.from_dict({
                'input_ids': eval_encodings['input_ids'],
                'attention_mask': eval_encodings['attention_mask'],
                'labels': eval_labels
            })
        else:
            eval_dataset = None

        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

    def predict(self, texts):
        """预测新样本的情感类别"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.classifier(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

# 定义自定义数据集
class CitationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def load_sentiment_datasets(test_size=0.4, seed=42, filepath='../data/corpus.txt', is_split=True):
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

    if is_split:
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(sentences,
                                                                              labels, test_size=test_size,
                                                                              stratify=labels, random_state=seed)
        val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5,
                                                                          stratify=temp_labels, random_state=seed)
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    else:
        return sentences, labels

def load_negative_texts(filepath='../data/20220206_CORPUS_critical_citations_DATA_PAPER.csv'):
    df = pd.read_csv(filepath)
    return df['Context'].tolist()

def main():
    classifier = CitationSentimentClassifier()

    negative_texts = load_negative_texts()
    # 1. 使用负面情感数据集进行继续预训练
    classifier.continue_pretraining(negative_texts)

    # 使用LLM为negative_texts生成软标签
    llm_predictions = classifier.generate_soft_labels(negative_texts)
    # 使用LLM生成的软标签进行知识蒸馏
    classifier.knowledge_distillation(negative_texts, llm_predictions)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_sentiment_datasets()

    # 2. 使用LLM生成的知识进行蒸馏
    # llm_predictions = get_llm_predictions(texts)  # 需要实现这个函数调用LLM
    # classifier.knowledge_distillation(texts, llm_predictions)
    #
    # # 3. 训练三分类分类器
    # classifier.train_classifier(train_texts, train_labels)
    #
    # # 预测新样本
    # predictions = classifier.predict(new_texts)

if __name__ == '__main__':
    main()