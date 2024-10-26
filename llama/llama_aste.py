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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_sentiment_triplets(text, sentiment, tokenizer, model, device):
    system_prompt = '''
        You are an AI assistant specialized in analyzing scientific citations and extracting aspect sentiment triplet from them according to the following sentiment elements definition: 
        -An "aspect term" refers to a specific contribution, method, finding, or component of the cited work that appears explicitly as a substring in the citation text
        -An "opinion term" refers to the evaluation or assessment expressed towards a particular aspect of the cited work, appearing explicitly as a substring in the citation text
        -The "sentiment polarity" indicates the author's attitude towards the cited aspect, with three possible values: "positive", "negative", or "neutral" (for mild/moderate evaluations)        
                
        Rules:
        1. Only extract up to 2 sentiment element triplets per citation.
        2. Aspect terms and opinion terms must be exact matches from the text.
        3. Sentiment polarity must be one of: "positive" or "negative".
        4. Ignore purely objective statements without evaluative content.

        Example 1:
        Text: Substantial improvements have been made to parse western language such as English, and many powerful models have been proposed.
        Overall Sentiment: Positive
        Sentiment Triplets:
        [("improvements", "substantial", "positive"), ("models", "powerful", "positive")]

        Example 2:
        Text: However, one of the major limitations of these advances is the structured syntactic knowledge, which is important to global reordering, has not been well exploited.
        Overall Sentiment: Negative
        Sentiment Triplets:
        [("structured syntactic knowledge", "has not been well exploited", "negative"), ("syntactic knowledge", "important", "positive")]

        '''

    user_prompt = f'''
        Analyze the sentiment elements in this scientific citation text. Provide exactly up to 2 triplets containing (aspect term, opinion term, sentiment polarity) for the following citation:
        
        Text: {text}
        Overall Sentiment: {sentiment}
        
        Provide your response in the format of a Python list of tuples:
        Sentiment elements: [("aspect term", "opinion term", "sentiment polarity"), ("aspect term", "opinion term", "sentiment polarity")]
        
        Only include the list of triplets in your response, with no additional text.
        '''
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_input = tokenizer([input_text], return_tensors='pt').to(device)

    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=128,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_input.input_ids, generated_ids)]  # remove the input text
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print(response)
    # 解析模型输出,提取三元组
    try:
        # 找到包含"Sentiment elements:"的行并提取元组列表
        triplets_str = response.split("Sentiment elements:")[-1].strip()
        # 使用eval()将字符串转换为Python列表
        triplets = eval(triplets_str)
        return triplets
    except:
        print("Error in extracting sentiment triplets")
        return []

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
    if is_split:
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(sentences,
                                                                              labels, test_size=test_size,
                                                                              stratify=labels, random_state=seed)
        val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5,
                                                                          stratify=temp_labels, random_state=seed)
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    else:
        return sentences, labels


def process_dataset(file_path, tokenizer, model, device):
    sentences, labels = load_sentiment_datasets(filepath=file_path, is_split=False)
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    labels = [label_map[label] for label in labels]

    df = pd.DataFrame({'Citation_Text': sentences, 'Sentiment': labels})
    df = df[df['Sentiment'].isin(['positive', 'negative'])]  # 只处理正面和负面情感的引文
    # df = df.head(5)  # 测试用

    results = []
    for _, row in tqdm(df.iterrows(), desc="Processing citations", total=len(df)):
        triplets = extract_sentiment_triplets(row['Citation_Text'], row['Sentiment'], tokenizer, model, device)
        results.append({
            'text': row['Citation_Text'],
            'overall_sentiment': row['Sentiment'],
            'sentiment_triplets': triplets
        })
    return results


def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    seed = 42
    seed_everything(seed)

    file_path = '../data/corpus.txt'
    output_dir = '../output/sentiment_aste_results.json'
    model_name = 'Meta-Llama-3-8B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2', device_map=device)

    res = process_dataset(file_path, tokenizer, model, device)
    save_results(res, output_dir)


if __name__ == '__main__':
    main()