import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random
import numpy as np
import os
import re
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
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

def extract_sentiment_words(text, sentiment, tokenizer, model, device):
    system_prompt = '''
        You are an AI assistant specialized in analyzing the sentiment of scientific citations. Your task is to extract exactly 1-3 sentiment-related words or short phrases (up to 3 words) that best represent the emotional tone of each given scientific citation. Focus on concise, impactful words that capture the essence of the sentiment.

        Guidelines:
        1. Extract 1-3 words or short phrases, no more, each word or phrase should be 1-3 words long.
        2. Do not extract unnecessary or neutral terms if only 1 or 2 word clearly reflects the emotional tone.
        3. Focus on adjectives, adverbs, or short noun phrases that convey sentiment.
        4. Avoid full sentences or long explanations.
        5. Ensure the words/phrases directly relate to the sentiment of the citation.
        6. If you can't find any distinct sentiment words/phrases, you may repeat a word/phrase, but try to avoid this if possible.
    
        Remember, brevity and relevance are key.

        Example:
        Text: This groundbreaking study provides compelling evidence for the effectiveness of the new treatment, demonstrating significant improvement in patient outcomes across multiple metrics.
        Sentiment: Positive
        Extracted Words: groundbreaking, compelling evidence, significant improvement

        Text: While the paper attempts to address an important issue, its methodology is flawed and the conclusions drawn are not adequately supported by the limited data presented.
        Sentiment: Negative
        Extracted Words: flawed, not adequately supported, limited data
        '''

    user_prompt = f'''
        Analyze the sentiment of the following scientific citation. Extract 1-3 words or short phrases (1-3 words each) that best represent the emotional tone of the citation. 
        
        Text: {text}
        Sentiment: {sentiment}

        Please respond with only the extracted words and provide the results in the following format:(word/phrases 2 and 3 are optional)
        [word/phrase 1], [word/phrase 2], [word/phrase 3]
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
        max_new_tokens=64,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)] # remove the input text
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(response)

    # Extract the sentiment words from the response
    sentiment_words = response.split("Assistant:")[-1].strip().split(", ")
    return sentiment_words

def process_dataset(csv_path, tokenizer, model, device):
    df = pd.read_csv(csv_path)
    label_map = {'o': 'neutral', 'p': 'positive', 'n': 'negative'}
    df['Sentiment'] = df['Sentiment'].map(label_map)

    df = df[df['Sentiment'].isin(['positive', 'negative'])]
    # df = df.head(20) # 测试用
    results = []
    sentiment_dictionary = {'positive': Counter(), 'negative': Counter()}

    for _, row in tqdm(df.iterrows(), desc="Processing citations", total=len(df)):
        sentiment_words = extract_sentiment_words(row['Citation_Text'], row['Sentiment'], tokenizer, model, device)
        # print(f"Processed text: {row['Citation_Text']}\nSentiment: {row['Sentiment']}\nExtracted words: {sentiment_words}\n")
        results.append({
            'text': row['Citation_Text'],
            'sentiment': row['Sentiment'],
            'sentiment_words': sentiment_words
        })
        sentiment_dictionary[row['Sentiment']].update(sentiment_words)

    return results, sentiment_dictionary

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_sentiment_dictionary(sentiment_dictionary, output_file):
    # Convert Counter objects to regular dictionaries
    sentiment_dict_json = {
        'positive': dict(sentiment_dictionary['positive']),
        'negative': dict(sentiment_dictionary['negative'])
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sentiment_dict_json, f, ensure_ascii=False, indent=2)

def main():
    seed = 42
    seed_everything(seed)

    csv_path = '../data/citation_sentiment_corpus.csv'
    output_dir = '../output'
    model_name = 'Qwen2.5-32B-Instruct-GPTQ-Int4'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=device)

    results, sentiment_lexicon = process_dataset(csv_path, tokenizer, model, device)
    save_results(results, f"{output_dir}/sentiment_lexicon_results.json")
    save_sentiment_dictionary(sentiment_lexicon, f"{output_dir}/sentiment_lexicon.json")

def clean_lexicon(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        sentiment_dict = json.load(f)

    cleaned_dict = {}
    for sentiment, words in sentiment_dict.items():
        cleaned_dict[sentiment] = {}
        for phrase, count in words.items():
            # 删除包含非英语字符的词
            if not re.match(r'^[a-zA-Z\s\-\[\]]+$', phrase):
                continue

            # 删除中括号并保留内容
            cleaned_phrase = phrase.replace('[', '').replace(']', '')

            # 如果是超过3个单词的短语，跳过
            if len(cleaned_phrase.split()) > 3:
                continue

            # 如果词已经在字典中，累加它们的出现次数
            if cleaned_phrase in cleaned_dict[sentiment]:
                cleaned_dict[sentiment][cleaned_phrase] += count
            else:
                cleaned_dict[sentiment][cleaned_phrase] = count

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_dict, f, ensure_ascii=False, indent=2)

# 计算词汇相似性并进行归并
def merge_similar_words(embeddings, threshold=0.9):
    tokenizer = AutoTokenizer.from_pretrained(f"../pretrain_models/roberta-llama3.1405B-twitter-sentiment")
    merged_dict = {}
    word_list = list(embeddings.keys())

    for i, word1 in enumerate(word_list):
        merged_to = word1
        if word1 in merged_dict:
            continue
        for j, word2 in enumerate(word_list[i+1:], i+1):
            # 计算词嵌入的余弦相似度
            similarity = cosine_similarity([embeddings[word1]], [embeddings[word2]])[0][0]
            if similarity >= threshold:
                # 使用相似度较高的词的BPE分词
                token1 = tokenizer.tokenize(word1)
                token2 = tokenizer.tokenize(word2)
                merged_to = min(token1, token2, key=lambda x: len(x))  # 选择较短的BPE子词
                merged_dict[word2] = merged_to
        merged_dict[word1] = merged_to

    return merged_dict

if __name__=='__main__':
    main()
    # clean_lexicon('../output/sentiment_lexicon.json',  '../output/cleaned_sentiment_lexicon.json')


