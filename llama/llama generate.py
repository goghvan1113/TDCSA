import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 启用Cudnn


def generate_samples(texts, sentiment, num_samples_per_text, tokenizer, model, device):
    generated_texts = []
    for original_text in tqdm(texts):
        system_prompt = f'''
            You are an AI assistant specialized in generating scientific citation sentiment data. Your task is to generate new examples similar to the given ones, maintaining the same sentiment and style. The generated examples should be diverse and realistic, mimicking the characteristics of real scientific citations.

            When generating new examples:
            1. Maintain the same sentiment (positive, negative, or neutral) as the original.
            2. Use similar language, tone, and structure to the original citation.
            3. Vary the content, focusing on different aspects of scientific work (e.g., methodology, results, implications).
            4. Ensure the generated text is coherent and resembles a real scientific citation.
            5. Do not copy the original text verbatim; create new, unique examples.

            Your output should be a list of generated examples, each on a new line starting with a double hyphen (--).
        '''
        user_prompt = f'''
            Generate new examples of scientific citation sentiment data based on the following input:

            Sentiment: {sentiment}
            Text: {original_text}
            Generate {num_samples_per_text} similar examples.

            Please provide {num_samples_per_text} new, unique examples that maintain the same sentiment and style as the original text.
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
            repetition_penalty=1.2,
        )
        generated_ids = generated_ids[0, len(model_input.input_ids[0]):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract individual samples from the generated text
        samples = [sample.strip() for sample in generated_text.split('--') if sample.strip()]
        generated_texts.extend(samples[:num_samples_per_text])  # Ensure we only take the requested number of samples
        print(f"Generated text: {generated_text}")

    return generated_texts

def balance_dataset(csv_path, model, tokenizer, device):
    df = pd.read_csv(csv_path)
    sentiment_counts = df['Sentiment'].value_counts()

    positive_texts = df[df['Sentiment'] == 'p']['Citation_Text'].tolist()
    negative_texts = df[df['Sentiment'] == 'n']['Citation_Text'].tolist()
    neutral_texts = df[df['Sentiment'] == 'o']['Citation_Text'].tolist()

    neutral_sample_count = sentiment_counts['o'] // 2
    num_positive = neutral_sample_count // sentiment_counts['p'] - 1
    num_negative = neutral_sample_count // sentiment_counts['n'] - 3 # 负类要生成少一点
    reduced_neutral_texts = random.sample(neutral_texts, neutral_sample_count)

    new_positive_texts = generate_samples(positive_texts, 'positive', num_positive, tokenizer, model, device)
    new_negative_texts = generate_samples(negative_texts, 'negative', num_negative, tokenizer, model, device)

    balanced_texts = positive_texts + new_positive_texts + negative_texts + new_negative_texts + reduced_neutral_texts
    balanced_labels = [1] * len(positive_texts) + [1] * len(new_positive_texts) + [2] * len(negative_texts) + [2] * len(new_negative_texts) + [0] * len(reduced_neutral_texts)
    sources = ['original'] * len(positive_texts) + ['new'] * len(new_positive_texts) + ['original'] * len(negative_texts) + ['new'] * len(new_negative_texts) + ['original'] * len(reduced_neutral_texts)

    balanced_df = pd.DataFrame(
        {'Citation_Text': balanced_texts, 'Sentiment': balanced_labels, 'Source': sources})

    return balanced_df

def main():
    seed = 42
    seed_everything(seed)

    csv_path = f'../data/citation_sentiment_corpus.csv'
    model_name = 'Meta-Llama-3.1-8B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2', device_map=device)
    balanced_df = balance_dataset(csv_path, model, tokenizer, device)
    balanced_df.to_csv('../data/citation_sentiment_corpus_balanced.csv', index=False)

if __name__ == '__main__':
    main()