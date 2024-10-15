import os
import time
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments

from custom_loss import MultiFocalLoss, MultiDSCLoss, AsymmetricLoss
from plot_results import plot_roc_curve, plot_pr_curve, plot_confusion_matrix

def main(args):
    num_sentiment_labels = 3
    num_intent_labels = 3
    test_size = 0.2
    sentiment_id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
    sentiment_label2id = {"Neutral": 0, "Positive": 1, "Negative": 2}
    intent_id2label = {0: "background", 1: "method", 2: "result"}
    intent_label2id = {"background": 0, "method": 1, "result": 2}

    device = torch.device(args.device)
    model_dir = f"../pretrain_models/{args.model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = CustomMultiTaskModel.from_pretrained(
        model_dir,
        num_sentiment_labels=num_sentiment_labels,
        num_intent_labels=num_intent_labels,
        sentiment_id2label=sentiment_id2label,
        sentiment_label2id=sentiment_label2id,
        intent_id2label=intent_id2label,
        intent_label2id=intent_label2id
    ).to(device)

    # Load datasets and create DataLoader
    train_data, val_data, test_data = load_multitask_datasets(test_size=test_size, seed=args.seed)
    
    print(f"Train Dataset Size: {len(train_data)}")
    print(f"Test Dataset Size: {len(test_data)}")
    print(f"Val Dataset Size: {len(val_data)}")

    # Set up training arguments and trainer
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
        metric_for_best_model='combined_f1',
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    trainer = CustomMultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_multitask_metrics,
        loss_type=args.loss_type,
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate()

    # Test set evaluation
    test_result = trainer.predict(test_data)
    
    # Save results
    save_results(args, eval_result, test_result)

class CustomMultiTaskModel(AutoModelForSequenceClassification):
    def __init__(self, config, num_sentiment_labels, num_intent_labels):
        super(CustomMultiTaskModel, self).__init__(config)
        self.bert = AutoModel.from_pretrained(config._name_or_path)
        self.dropout = torch.nn.Dropout(0.3)
        self.sentiment_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_sentiment_labels)
        self.intent_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_intent_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, sentiment_labels=None, intent_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return sentiment_logits, intent_logits

class CustomMultiTaskTrainer(Trainer):
    def __init__(self, loss_type='focal_loss', *args, **kwargs):
        super(CustomMultiTaskTrainer, self).__init__(*args, **kwargs)
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False):
        sentiment_labels = inputs.pop("sentiment_labels")
        intent_labels = inputs.pop("intent_labels")
        outputs = model(**inputs)
        sentiment_logits, intent_logits = outputs

        if self.loss_type == 'focal_loss':
            loss_fct = MultiFocalLoss(num_class=3, alpha=0.8, gamma=2.0)
        elif self.loss_type == 'dsc_loss':
            loss_fct = MultiDSCLoss(alpha=1.0, smooth=1.0)
        elif self.loss_type == 'asymmetric_loss':
            loss_fct = AsymmetricLoss(gamma_pos=0.5, gamma_neg=3.0)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
        intent_loss = loss_fct(intent_logits, intent_labels)
        total_loss = sentiment_loss + intent_loss

        return (total_loss, outputs) if return_outputs else total_loss

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

def compute_multitask_metrics(pred):
    sentiment_labels = pred.label_ids[:, 0]
    intent_labels = pred.label_ids[:, 1]
    sentiment_preds = pred.predictions[0].argmax(-1)
    intent_preds = pred.predictions[1].argmax(-1)

    sentiment_acc = accuracy_score(sentiment_labels, sentiment_preds)
    intent_acc = accuracy_score(intent_labels, intent_preds)

    sentiment_precision, sentiment_recall, sentiment_f1, _ = precision_recall_fscore_support(sentiment_labels, sentiment_preds, average='macro')
    intent_precision, intent_recall, intent_f1, _ = precision_recall_fscore_support(intent_labels, intent_preds, average='macro')

    combined_f1 = (sentiment_f1 + intent_f1) / 2

    return {
        'Sentiment_Accuracy': sentiment_acc,
        'Sentiment_Precision': sentiment_precision,
        'Sentiment_Recall': sentiment_recall,
        'Sentiment_F1': sentiment_f1,
        'Intent_Accuracy': intent_acc,
        'Intent_Precision': intent_precision,
        'Intent_Recall': intent_recall,
        'Intent_F1': intent_f1,
        'Combined_F1': combined_f1
    }

def load_multitask_datasets(test_size=0.2, seed=42):
    sentiment_df = pd.read_csv('../data/citation_sentiment_corpus.csv')
    intent_df = pd.read_csv('../data/citation_intent_corpus.csv')

    # Ensure both datasets have the same number of samples
    min_samples = min(len(sentiment_df), len(intent_df))
    sentiment_df = sentiment_df.sample(n=min_samples, random_state=seed)
    intent_df = intent_df.sample(n=min_samples, random_state=seed)

    texts = sentiment_df['Citation_Text'].tolist()
    sentiment_labels = sentiment_df['Sentiment'].map({'o': 0, 'p': 1, 'n': 2}).tolist()
    intent_labels = intent_df['Intent'].map({'background': 0, 'method': 1, 'result': 2}).tolist()

    train_texts, temp_texts, train_sentiment_labels, temp_sentiment_labels, train_intent_labels, temp_intent_labels = train_test_split(
        texts, sentiment_labels, intent_labels, test_size=test_size, stratify=sentiment_labels, random_state=seed
    )

    val_texts, test_texts, val_sentiment_labels, test_sentiment_labels, val_intent_labels, test_intent_labels = train_test_split(
        temp_texts, temp_sentiment_labels, temp_intent_labels, test_size=0.5, stratify=temp_sentiment_labels, random_state=seed
    )

    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/{args.model_name}")

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

def save_results(args, eval_result, test_result):
    results = {
        'seed': args.seed,
        'eval_result': eval_result,
        'test_result': test_result,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'accumulation_steps': args.accumulation_steps,
        'warmup_ratio': args.warmup_ratio,
        'loss_type': args.loss_type
    }

    json_file_path = '../output/multitask_training_details.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = []
    existing_results.append(results)
    with open(json_file_path, 'w') as f:
        json.dump(existing_results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='roberta-llama3.1405B-twitter-sentiment', help='Model name or path')
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

