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
from bert.sentiment_finetuning import load_sentiment_datasets


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_aspect_pairs(text, sentiment, tokenizer, model, device):
    system_prompt = '''
        You are an AI assistant specialized in analyzing scientific citations and extracting aspect-pairs (aspect and corresponding sentiment) from them. Your task is to identify key aspects discussed in the citation and determine the sentiment associated with each aspect. 

        Guidelines:
        1. Extract between 1 to 3 aspect-pairs from each citation.
        2. An aspect can be any specific element of the research being cited, such as methodology, results, implications, innovation, etc.
        3. The sentiment for each aspect should be clearly positive or negative.
        4. Provide your response in a structured format: Aspect: [aspect], Sentiment: [positive/negative]
        5. Ensure that the extracted aspects are relevant and significant to the overall sentiment of the citation.

        Remember, the overall sentiment of the citation is provided, but individual aspects within the citation may have different sentiments.

        Example 1 (Positive):
        Text: This groundbreaking study provides compelling evidence for the effectiveness of the new treatment, demonstrating significant improvement in patient outcomes across multiple metrics.
        Overall Sentiment: Positive
        Aspect-pairs:
        Aspect: Study quality, Sentiment: Positive
        Aspect: Treatment effectiveness, Sentiment: Positive
        Aspect: Patient outcomes, Sentiment: Positive

        Example 2 (Negative):
        Text: While the paper attempts to address an important issue, its methodology is flawed and the conclusions drawn are not adequately supported by the limited data presented.
        Overall Sentiment: Negative
        Aspect-pairs:
        Aspect: Research topic, Sentiment: Positive
        Aspect: Methodology, Sentiment: Negative
        Aspect: Data support, Sentiment: Negative

        '''

    user_prompt = f'''
        Analyze the following scientific citation and extract 1 to 3 aspect-pairs (aspect and sentiment):

        Text: {text}
        Overall Sentiment: {sentiment}

        Please provide the aspect-pairs in the following format:
        Aspect: [aspect], Sentiment: [positive/negative]
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
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print(response)
    # 解析模型输出，提取 aspect-pairs
    aspect_pairs = []
    for line in response.split('\n'):
        if line.startswith('Aspect:'):
            parts = line.split(',')
            if len(parts) == 2:
                aspect = parts[0].replace('Aspect:', '').strip()
                sentiment = parts[1].replace('Sentiment:', '').strip()
                aspect_pairs.append({'aspect': aspect, 'sentiment': sentiment})

    return aspect_pairs

def process_dataset(file_path, tokenizer, model, device):
    sentences, labels = load_sentiment_datasets(filepath=file_path, is_spilit=False)
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    labels = labels.map(label_map)

    df = pd.DataFrame({'Citation_Text': sentences, 'Sentiment': labels})
    df = df[df['Sentiment'].isin(['positive', 'negative'])]
    df = df.head(5) # 测试用
    results = []
    for _, row in tqdm(df.iterrows(), desc="Processing citations", total=len(df)):
        aspect_pairs = extract_aspect_pairs(row['Citation_Text'], row['Sentiment'], tokenizer, model, device)
        results.append({
            'text': row['Citation_Text'],
            'overall_sentiment': row['Sentiment'],
            'aspect_pairs': aspect_pairs
        })
    return results

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    seed = 42
    seed_everything(seed)

    file_path = '../data/corpus.txt'
    output_dir = '../output/sentiment_absa_results.json'
    model_name = 'Meta-Llama-3.1-8B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2', device_map=device)

    res = process_dataset(file_path, tokenizer, model, device)
    save_results(res, output_dir)

if __name__ == '__main__':
    main()