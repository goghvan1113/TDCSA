import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, f1_score
from tqdm import tqdm
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer


class CitationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        tokens = self.tokenizer(text.lower())

        # Convert tokens to indices
        indices = [self.vocab[token] for token in tokens]

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<pad>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]

        return {
            'indices': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def prepare_data(pos_neg_file: str, neutral_file: str) -> Tuple[List[str], List[int]]:
    """Load and prepare data from JSON files

    Args:
        pos_neg_file (str): Path to the file containing positive/negative samples
        neutral_file (str): Path to the file containing neutral samples

    Returns:
        Tuple[List[str], List[int]]: Lists containing texts and their corresponding labels
    """
    texts = []
    labels = []

    # Load positive/negative samples
    with open(pos_neg_file, 'r', encoding='utf-8') as f:
        pos_neg_data = json.load(f)

    for item in pos_neg_data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            texts.append(item['text'])
            labels.append(1 if item['overall_sentiment'] == 'positive' else 2)

    # Load neutral samples
    with open(neutral_file, 'r', encoding='utf-8') as f:
        neutral_data = json.load(f)

    for item in neutral_data:
        texts.append(item['text'])
        labels.append(0)  # neutral label

    return texts, labels


def prepare_data_and_vocab(pos_neg_file: str, neutral_file: str, min_freq: int = 2):
    """Load data and prepare vocabulary using torchtext

    Args:
        pos_neg_file (str): Path to positive/negative samples
        neutral_file (str): Path to neutral samples
        min_freq (int): Minimum frequency for a word to be included in vocab
    """
    texts = []
    labels = []

    # Load positive/negative samples
    with open(pos_neg_file, 'r', encoding='utf-8') as f:
        pos_neg_data = json.load(f)

    for item in pos_neg_data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            texts.append(item['text'])
            labels.append(1 if item['overall_sentiment'] == 'positive' else 2)

    # Load neutral samples
    with open(neutral_file, 'r', encoding='utf-8') as f:
        neutral_data = json.load(f)

    for item in neutral_data:
        texts.append(item['text'])
        labels.append(0)  # neutral label

    # Initialize tokenizer
    tokenizer = get_tokenizer('basic_english')

    # Build word frequency dictionary from the dataset
    word_freq = {}
    for text in texts:
        tokens = tokenizer(text.lower())
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    glove = GloVe(name='6B', dim=300)

    # Create vocabulary
    # 1. Start with special tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    embedding_matrix = torch.zeros(2, 300)  # Initialize with special tokens
    embedding_matrix[1] = torch.randn(300) * 0.1  # Random init for <unk>

    # 2. Add words that appear in both dataset and GloVe
    print("Building vocabulary...")
    added_words = set()
    for word, freq in word_freq.items():
        # Skip words that don't meet minimum frequency
        if freq < min_freq:
            continue

        # Add words that are in GloVe
        if word in glove.stoi and word not in vocab:
            vocab[word] = len(vocab)
            embedding_matrix = torch.cat([
                embedding_matrix,
                glove.vectors[glove.stoi[word]].unsqueeze(0)
            ])
            added_words.add(word)

    # Print statistics
    total_words = len(word_freq)
    covered_words = len(added_words)
    print(f"Vocabulary statistics:")
    print(f"Total unique words in dataset: {total_words}")
    print(f"Words covered by GloVe: {covered_words}")
    print(f"Coverage rate: {covered_words / total_words * 100:.2f}%")
    print(f"Words that will be mapped to <unk>: {total_words - covered_words}")

    # Create a function to map tokens to indices with proper handling of unknown words
    def tokens_to_indices(tokens):
        return [vocab.get(token, vocab['<unk>']) for token in tokens]

    # Test vocabulary coverage on a few examples
    print("\nTesting vocabulary coverage on first few examples:")
    for text in texts[:3]:
        tokens = tokenizer(text.lower())
        unk_tokens = [t for t in tokens if t not in vocab]
        if unk_tokens:
            coverage = (len(tokens) - len(unk_tokens)) / len(tokens) * 100
            print(f"\nExample text: {text[:100]}...")
            print(f"Unknown tokens: {unk_tokens}")
            print(f"Token coverage: {coverage:.2f}%")

    return texts, labels, vocab, tokenizer, embedding_matrix


class BertDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Models
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int,
                 filter_sizes: List[int], output_dim: int, dropout: float,
                 pad_idx: int, embedding_matrix: Optional[torch.Tensor] = None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, indices):
        embedded = self.embedding(indices)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int, dropout: float, pad_idx: int,
                 embedding_matrix: Optional[torch.Tensor] = None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, indices):
        embedded = self.dropout(self.embedding(indices))
        outputs, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int, dropout: float, pad_idx: int,
                 embedding_matrix: Optional[torch.Tensor] = None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)

        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention_net(self, lstm_output):
        att_weights = torch.tanh(self.attention(lstm_output))
        att_weights = F.softmax(att_weights, dim=1)
        att_out = torch.sum(att_weights * lstm_output, dim=1)
        return att_out

    def forward(self, indices):
        embedded = self.dropout(self.embedding(indices))
        outputs, _ = self.lstm(embedded)
        attention_out = self.attention_net(outputs)
        return self.fc(self.dropout(attention_out))

class CitationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        tokens = self.tokenizer(text.lower())

        # Convert tokens to indices
        indices = [self.vocab[token] for token in tokens]

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<pad>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]

        return {
            'indices': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_data_and_vocab(pos_neg_file: str, neutral_file: str):
    """Load data and prepare vocabulary using torchtext"""
    # Load data
    texts = []
    labels = []

    # Load positive/negative samples
    with open(pos_neg_file, 'r', encoding='utf-8') as f:
        pos_neg_data = json.load(f)

    for item in pos_neg_data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            texts.append(item['text'])
            labels.append(1 if item['overall_sentiment'] == 'positive' else 2)

    # Load neutral samples
    with open(neutral_file, 'r', encoding='utf-8') as f:
        neutral_data = json.load(f)

    for item in neutral_data:
        texts.append(item['text'])
        labels.append(0)  # neutral label

    # Initialize tokenizer
    tokenizer = get_tokenizer('basic_english')

    # Load GloVe embeddings
    glove = GloVe(name='6B', dim=300)

    # Create vocabulary from GloVe
    vocab = glove.stoi.copy()  # Create a copy to avoid modifying original
    # Add special tokens
    special_tokens = ['<pad>', '<unk>']
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    # Create embedding matrix
    embedding_matrix = torch.zeros(len(vocab), 300)
    for word, idx in vocab.items():
        if word in glove.stoi:
            embedding_matrix[idx] = glove.vectors[glove.stoi[word]]
        else:
            embedding_matrix[idx] = torch.randn(300) * 0.1

    return texts, labels, vocab, tokenizer, embedding_matrix


# 3. 新的BERT+BiLSTM+Attention模型(替换原有的BERT模型)
class BertBiLSTMAttention(nn.Module):
    def __init__(self, bert_model: str, hidden_dim: int, output_dim: int,
                 n_layers: int, dropout: float):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model)
        embedding_dim = self.bert.config.hidden_size

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        # BERT output
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]

        # BiLSTM
        lstm_output, _ = self.lstm(bert_output)

        # Attention
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_weights = attention_weights * attention_mask.unsqueeze(-1)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum
        weighted_output = (lstm_output * attention_weights).sum(dim=1)

        # Final classification
        return self.fc(self.dropout(weighted_output))


# Training and Evaluation Functions
class MetricTracker:
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_acc': [],
            'epochs': []
        }

    def update(self, epoch: int, train_loss: float, val_metrics: Dict):
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_metrics['loss'])
        self.metrics['val_f1'].append(val_metrics['f1_macro'])
        self.metrics['val_acc'].append(val_metrics['accuracy'])

    def plot_metrics(self, save_path: str = None):
        plt.figure(figsize=(12, 8))

        # Loss subplot
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics['epochs'], self.metrics['train_loss'], 'b-', label='Training Loss')
        plt.plot(self.metrics['epochs'], self.metrics['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Metrics subplot
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics['epochs'], self.metrics['val_f1'], 'g-', label='Validation F1')
        plt.plot(self.metrics['epochs'], self.metrics['val_acc'], 'y-', label='Validation Accuracy')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, is_bert: bool = False) -> float:
    model.train()
    epoch_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        if is_bert:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            predictions = model(input_ids, attention_mask)
        else:
            indices = batch['indices'].to(device)
            labels = batch['label'].to(device)
            predictions = model(indices)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device, is_bert: bool = False) -> Dict:
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        if is_bert:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            predictions = model(input_ids, attention_mask)
        else:
            indices = batch['indices'].to(device)
            labels = batch['label'].to(device)
            predictions = model(indices)

        loss = criterion(predictions, labels)
        epoch_loss += loss.item()

        preds = torch.argmax(predictions, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        'loss': epoch_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'classification_report': classification_report(
            all_labels, all_preds,
            target_names=['neutral', 'positive', 'negative'],
            digits=4
        )
    }

def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                n_epochs: int, device: torch.device, is_bert: bool = False) -> Dict:
    """Train model with metric tracking"""
    tracker = MetricTracker()

    for epoch in range(n_epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_bert)

        # Validation phase
        val_metrics = evaluate(model, valid_loader, criterion, device, is_bert)

        # Update metric tracker
        tracker.update(epoch + 1, train_loss, val_metrics)

        # Print progress
        print(f'\nEpoch: {epoch + 1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'Val F1 Macro: {val_metrics["f1_macro"]:.4f}')
        print('\nClassification Report:')
        print(val_metrics["classification_report"])

    return tracker


def seed_everything(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    parser = argparse.ArgumentParser(description='Citation Sentiment Analysis')
    parser.add_argument('--model_type', type=str, default='bilstm',
                        choices=['textcnn', 'bilstm', 'bilstm_attention', 'bert_bilstm_attention'])
    parser.add_argument('--bert_model', type=str,
                        default='bert-base-uncased')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    pos_neg_file = '../output/sentiment_asqp_results_corpus_expand_llama405b.json'
    neutral_file = '../output/sentiment_asqp_results_corpus_expand_llama8b_neutral.json'

    if args.model_type == 'bert_bilstm_attention':
        texts, labels = prepare_data(pos_neg_file, neutral_file)
    else:
        texts, labels, vocab, tokenizer, embedding_matrix = prepare_data_and_vocab(pos_neg_file, neutral_file)

    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=args.seed
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels,
        random_state=args.seed
    )

    # Prepare model and data specific components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and model based on model type
    if args.model_type == 'bert_bilstm_attention':
        print("Initializing BERT tokenizer and datasets...")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        train_dataset = BertDataset(train_texts, train_labels, tokenizer, args.max_length)
        val_dataset = BertDataset(val_texts, val_labels, tokenizer, args.max_length)
        test_dataset = BertDataset(test_texts, test_labels, tokenizer, args.max_length)

        model = BertBiLSTMAttention(args.bert_model, args.hidden_dim, 3,
                                    args.n_layers, args.dropout)
        is_bert = True
    else:
        print("Initializing traditional model datasets...")
        train_dataset = CitationDataset(train_texts, train_labels, vocab, tokenizer, args.max_length)
        val_dataset = CitationDataset(val_texts, val_labels, vocab, tokenizer, args.max_length)
        test_dataset = CitationDataset(test_texts, test_labels, vocab, tokenizer, args.max_length)

        if args.model_type == 'textcnn':
            model = TextCNN(
                vocab_size=len(vocab),
                embedding_dim=300,
                n_filters=100,
                filter_sizes=[3, 4, 5],
                output_dim=3,
                dropout=args.dropout,
                pad_idx=vocab['<pad>'],
                embedding_matrix=embedding_matrix
            )
        elif args.model_type == 'bilstm':
            model = BiLSTM(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=args.hidden_dim,
                output_dim=3,
                n_layers=args.n_layers,
                dropout=args.dropout,
                pad_idx=vocab['<pad>'],
                embedding_matrix=embedding_matrix
            )
        else:  # bilstm_attention
            model = BiLSTMAttention(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=args.hidden_dim,
                output_dim=3,
                n_layers=args.n_layers,
                dropout=args.dropout,
                pad_idx=vocab['<pad>'],
                embedding_matrix=embedding_matrix
            )
        is_bert = False

    model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train model
    print(f"\nTraining {args.model_type} model...")
    tracker = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        args.epochs, device, is_bert
    )

    # Plot and save training metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.output_dir,
                             f'{args.model_type}_metrics_{timestamp}.png')
    tracker.plot_metrics(save_path=plot_path)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, is_bert)

    # Save test results
    results = {
        'model_type': args.model_type,
        'test_metrics': test_metrics,
        'args': vars(args)
    }

    results_path = os.path.join(args.output_dir,
                                f'{args.model_type}_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\nTest Set Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(test_metrics['classification_report'])
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()