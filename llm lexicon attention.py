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

seed = 42
model_name = 'Qwen2.5-7B-Instruct'
model_dir = f'./pretrain_models/{model_name}'
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation='flash_attention_2',
                                             device_map=device)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sentiment_words(text, sentiment):
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

def create_sentiment_lexicon(path, top_n=20):
    """从CSV数据集创建情感词典"""
    df = pd.read_csv(path)
    
    positive_words = []
    negative_words = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Citation_Text']
        sentiment = row['Sentiment']
        
        if sentiment == 1:  # 正面情感
            positive_words.extend(get_sentiment_words(text, 'positive'))
        elif sentiment == 2:  # 负面情感
            negative_words.extend(get_sentiment_words(text, 'negative'))
        # 忽略中性（0）样本

    # 计算词频并选择最常见的词，选择前N个最常见的词
    positive_counter = Counter(positive_words)
    negative_counter = Counter(negative_words)
    top_positive = [word for word, _ in positive_counter.most_common(top_n)]
    top_negative = [word for word, _ in negative_counter.most_common(top_n)]

    return {'positive': top_positive, 'negative': top_negative}


if __name__ == "__main__":
    csv_path = './data/citation_sentiment_corpus.csv'  # 替换为你的CSV文件路径
    sentiment_lexicon = create_sentiment_lexicon(csv_path)
    
    # 将词典保存到文件
    with open('scientific_citation_sentiment_lexicon.json', 'w') as f:
        json.dump(sentiment_lexicon, f, indent=2)
