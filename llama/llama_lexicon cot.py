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
import re
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


def extract_sentiment_words(text, sentiment, tokenizer, model, device):
    system_prompt = '''
        You are an AI assistant specialized in analyzing the sentiment of scientific citations. Your task is to extract 1-3 sentiment-related words or short phrases (up to 3 words) that best represent the emotional tone of each given scientific citation. Focus on concise, impactful words that capture the essence of the sentiment.

         IMPORTANT: 
        - **You MUST ONLY use words or short phrases that are directly present in the original citation text.**
        - **Do NOT split meaningful multi-word phrases (e.g., "no result", "most effective").**
        - **Preserve such phrases as a single unit if they express a clear sentiment or concept.**

        You can think step-by-step to identify the sentiment words:
        1. Carefully read the given scientific citation text.
        2. Identify key words and phrases that may express sentiment or attitude, focus on adjectives, adverbs, or short noun.
        3. Consider the meaning of these words in a scientific context and the sentiment they might convey.
        4. Based on your analysis, extract 1-3 words or phrases from the text that best represent the overall sentiment or attitude of the text.
        5. If the text seems neutral, consider the author's possible implied attitude or sentiment.

        Examples:
        Positive Example:
        Input: "Our findings demonstrate a significant improvement in efficiency using this novel approach."
        Thought process:

            Step 1: Key phrases: "significant improvement", "novel approach"
            Step 2: In scientific context, "significant improvement" indicates positive research results
            Step 3: "Novel approach" suggests innovation, which is generally viewed positively in research
            Step 4: The author likely feels satisfied and optimistic about the results
            Output: significant, efficiency, novel

        Negative Example:
        Input: "The results failed to support our initial hypothesis, highlighting the limitations of the current methodology."
        Thought process:

            Step 1: Key phrases: "failed to support", "limitations"
            Step 2: "Failed to support" indicates a negative outcome for the research
            Step 3: "Limitations" suggests drawbacks or problems with the method
            Step 4: The author may feel disappointed or critical of the current approach
            Output: failed, limitations
        '''

    user_prompt = f'''
        Analyze the sentiment of the following scientific citation. Extract 1-3 words or short phrases (up to 3 words) that best represent the emotional tone of the citation. 

        Text: {text}
        Sentiment: {sentiment}

        Please provide your step-by-step reasoning, then list your final extracted words or phrases and provide the results in the following format:
        Reasoning: [Step 1: ... 
                    Step 2: ...
                    Step 3: ...
                    Step 4: ...]
        Output: [word1, word2, word3]
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
        max_new_tokens=512,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_input.input_ids, generated_ids)]  # remove the input text
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(response)

    # Extract the sentiment words from the response
    response_parts = response.split("Output:")
    reasoning = response_parts[0].strip()
    sentiment_words = response_parts[1].strip().split(", ")

    return sentiment_words, reasoning


def process_dataset(csv_path, tokenizer, model, device):
    df = pd.read_csv(csv_path)
    label_map = {'o': 'neutral', 'p': 'positive', 'n': 'negative'}
    df['Sentiment'] = df['Sentiment'].map(label_map)

    df = df[df['Sentiment'].isin(['positive', 'negative'])]
    df = df.head(10)
    results = []
    sentiment_dictionary = {'positive': Counter(), 'negative': Counter()}

    for _, row in tqdm(df.iterrows(), desc="Processing citations", total=len(df)):
        sentiment_words, reasoning = extract_sentiment_words(row['Citation_Text'], row['Sentiment'], tokenizer, model,
                                                             device)
        # print(f"text: {row['Citation_Text']}\nSentiment: {row['Sentiment']}\nExtracted words: {sentiment_words}\n")
        results.append({
            'text': row['Citation_Text'],
            'sentiment': row['Sentiment'],
            'sentiment_words': sentiment_words,
            'reasoning': reasoning
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
    model_name = 'Meta-Llama-3.1-8B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2', device_map=device)

    results, sentiment_lexicon = process_dataset(csv_path, tokenizer, model, device)
    save_results(results, f"{output_dir}/sentiment_lexicon_results.json")
    save_sentiment_dictionary(sentiment_lexicon, f"{output_dir}/sentiment_lexicon.json")


if __name__ == '__main__':
    main()


