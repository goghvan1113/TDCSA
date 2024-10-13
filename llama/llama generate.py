from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import os

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

def generate_samples(texts, target_class, num_samples):
    generated_texts = []
    for _ in tqdm(range(num_samples)):
        prompt = f"Generate a scientific citation text with a {target_class} sentiment, and the text should be a complete sentence."
        messages = [
            {'role': 'system', 'content': 'You are an expert in scientific citation sentiment analysis who can generate a scientific citation text with given sentiment and example. Please answer with generated text only.'},
            {'role': 'user', 'content': prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_input = tokenizer([text], return_tensors='pt').to(device)
        attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
        generated_ids = model.generate(
            model_input.input_ids,
            max_new_tokens=50,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_input.input_ids, generated_ids)
        ]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_texts.append(generated_text)
    return generated_texts

def balance_dataset(df):
    positive_texts = df[df['Sentiment'] == 1]['Citation_Text'].tolist()
    negative_texts = df[df['Sentiment'] == 2]['Citation_Text'].tolist()
    neutral_texts = df[df['Sentiment'] == 0]['Citation_Text'].tolist()

    num_positive = len(positive_texts)
    num_negative = len(negative_texts)
    num_neutral = len(neutral_texts)

    max_samples = max(num_positive, num_negative, num_neutral) // 2

    # Generate new positive and negative samples
    new_positive_texts = generate_samples(positive_texts, 'positive', max_samples - num_positive)
    new_negative_texts = generate_samples(negative_texts, 'negative', max_samples - num_negative)

    # Reduce the number of neutral samples
    reduced_neutral_texts = random.sample(neutral_texts, max_samples)

    # Combine all samples
    balanced_texts = positive_texts + new_positive_texts + negative_texts + new_negative_texts + reduced_neutral_texts
    balanced_labels = [1] * (num_positive + len(new_positive_texts)) + [2] * (num_negative + len(new_negative_texts)) + [0] * max_samples

    balanced_df = pd.DataFrame({'Citation_Text': balanced_texts, 'Sentiment': balanced_labels})
    return balanced_df

def main():
    seed_everything(seed)
    df = pd.read_csv('../data/citation_sentiment_corpus_new.csv')
    balanced_df = balance_dataset(df)
    balanced_df.to_csv('./data/citation_sentiment_corpus_balanced.csv', index=False)
    print("Balanced dataset saved to './data/citation_sentiment_corpus_balanced.csv'")

if __name__ == '__main__':
    main()