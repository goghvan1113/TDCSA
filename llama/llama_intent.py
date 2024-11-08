import argparse
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classify_intent(text, tokenizer, model, device):
    system_prompt = '''
    You are an AI assistant specialized in analyzing scientific citations and classifying their intent. Your task is to categorize each citation into one of four categories: background, method, result, or unknown.

    Guidelines:
    1. Classify each citation as either 'background', 'method', 'result', or 'unknown'.
    2. Provide a confidence score between 0 and 1 for your classification.
    3. Use 'unknown' if the citation doesn't clearly fit into the other categories.
    4. Consider the following when classifying:
       - 'background': The citation provides background information or additional context about a relevant problem , concept , approach , or topic
       - 'method': The citation refers to the use of a specific method , tool , approach , or dataset from the reference paper.
       - 'result': The citation compares or contrasts the results or findings of the manuscript with those in the reference paper.
       - 'unknown': Use when the intent is unclear or doesn't fit the other categories.

    Provide your response in this format:
    Intent: [intent], Confidence: [score]

    Example:
    Text: "These #REFR in which the optimal strategy maximizes the expected reward under the most adversarial distribution over the uncertainty set."
    Response: Intent: background, Confidence: 0.85
    
    Text: "These indicators were developed because evidences have been published that this data is -similar to bibliometric data -field-and time-dependent (see, e.g., #REFR ."
    Response: Intent: method, Confidence: 0.90
    
    Text: "In the case of uniformly bounded delays, the derived link between epoch and time sequence enables us to compare our rates in the strongly convex case (Theorem 3.1) with the ones obtained for PIAG #REFR 27, 28]"
    Response: Intent: result, Confidence: 0.80
    '''

    user_prompt = f'''
    Classify the intent of the following scientific citation:

    Text: {text}

    Please provide the classification in the following format:
    Intent: [intent], Confidence: [score]
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
    )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 解析模型输出
    intent, confidence = 'unknown', 0.0
    for line in response.split('\n'):
        if line.startswith('Intent:'):
            parts = line.split(',')
            if len(parts) == 2:
                intent = parts[0].replace('Intent:', '').strip().lower()
                confidence = float(parts[1].replace('Confidence:', '').strip())

    return intent, confidence


def load_sentiment_datasets(test_size=0.2, val_size=0.1, seed=42, filepath='../data/corpus.txt', is_split=True):
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
    elif filepath == '../data/citation_sentiment_corpus_expand.csv':
        df = pd.read_csv(filepath)
        label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
        df['Sentiment'] = df['Sentiment'].map(label_map)
        sentences = df['Text'].tolist()
        labels = df['Sentiment'].tolist()

    if is_split:
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            sentences,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=seed)

        # df_aug = pd.read_csv('../data/train_data_aug3.csv')
        # train_texts = df_aug['Citation_Text'].tolist()
        # train_labels = df_aug['Sentiment'].tolist() # 替换增强后的整个数据集
        # train_texts, train_labels = shuffle(train_texts, train_labels, random_state=seed) # 打乱新的训练集

        val_ratio = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts,
            train_val_labels,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=seed)

        # Print label distribution
        print("Train set label distribution:", Counter(train_labels))
        print("Validation set label distribution:", Counter(val_labels))
        print("Test set label distribution:", Counter(test_labels))

        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    else:
        return sentences, labels


def process_dataset(sentences, labels, tokenizer, model, device):
    results = []
    for sentence, sentiment in tqdm(zip(sentences, labels), desc="Processing citations", total=len(sentences)):
        intent, confidence = classify_intent(sentence, tokenizer, model, device)
        results.append({
            'text': sentence,
            'sentiment': sentiment,
            'intent': intent,
            'confidence': confidence
        })
    return results


def save_results(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    seed = 42
    seed_everything(seed)

    file_path = '../data/corpus.txt'
    output_dir = '../output/corpus_with_intent.csv'
    model_name = 'Meta-Llama-3.1-8B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2', device_map=device)

    sentences, labels = load_sentiment_datasets(filepath=file_path, is_split=False)
    results = process_dataset(sentences, labels, tokenizer, model, device)
    save_results(results, output_dir)


if __name__ == '__main__':
    main()