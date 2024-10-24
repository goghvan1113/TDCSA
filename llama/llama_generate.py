import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

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
            6. Do not include any additional notes, explanations, or comments in your response.
            7. If the number of samples requested is 1, generate exactly one example and do not Do not include any additional notes, optional sentences or explanations in your response.

            Your output should be a list of generated examples, each on a new line starting with a double hyphen (--).
        '''
        user_prompt = f'''
            Generate new examples of scientific citation sentiment data based on the following input in english:

            Sentiment: {sentiment}
            Text: {original_text}
            Generate {num_samples_per_text} similar examples.

            Please strictly provide {num_samples_per_text} new, unique examples. If {num_samples_per_text} is 1, provide only one example without using "or" to connect multiple sentences. Do not include any additional notes or explanations in your response.        
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
        print(generated_text)
        # Extract individual samples from the generated text
        samples = [sample.strip() for sample in generated_text.split('--') if sample.strip()]
        generated_texts.extend(samples[:num_samples_per_text])  # Ensure we only take the requested number of samples

    return generated_texts

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

def balance_dataset(csv_path, model, tokenizer, device):
    sentences, labels, _, _, _, _ = load_sentiment_datasets(filepath=csv_path, test_size=0.2, is_split=True)
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    labels = [label_map[label] for label in labels]
    df = pd.DataFrame({'Citation_Text': sentences, 'Sentiment': labels})

    sentiment_counts = df['Sentiment'].value_counts()

    positive_texts = df[df['Sentiment'] == 'positive']['Citation_Text'].tolist()
    negative_texts = df[df['Sentiment'] == 'negative']['Citation_Text'].tolist()
    neutral_texts = df[df['Sentiment'] == 'neutral']['Citation_Text'].tolist()

    # 下面是采样一半中性样本，以及和中性样本匹配的样本
    # num_neutral = sentiment_counts['neutral'] // 2
    # num_positive = neutral_sample_count // sentiment_counts['positive'] - 1
    # num_negative = neutral_sample_count // sentiment_counts['negative'] - 4 # 负类要生成少一点
    # neutral_texts = random.sample(neutral_texts, num_neutral)#
    
    # 下面是中性样本不变，正负各多生成一倍样本
    num_positive, num_negative = 2, 3

    new_positive_texts = generate_samples(positive_texts, 'positive', num_positive, tokenizer, model, device)
    new_negative_texts = generate_samples(negative_texts, 'negative', num_negative, tokenizer, model, device)

    balanced_texts = positive_texts + new_positive_texts + negative_texts + new_negative_texts + neutral_texts
    balanced_labels = [1] * len(positive_texts) + [1] * len(new_positive_texts) + [2] * len(negative_texts) + [2] * len(new_negative_texts) + [0] * len(neutral_texts)
    sources = ['original'] * len(positive_texts) + ['new'] * len(new_positive_texts) + ['original'] * len(negative_texts) + ['new'] * len(new_negative_texts) + ['original'] * len(neutral_texts)

    balanced_df = pd.DataFrame(
        {'Citation_Text': balanced_texts, 'Sentiment': balanced_labels, 'Source': sources})

    return balanced_df

def main():
    seed = 42 # 做数据增强的时候这个种子要和微调文件的种子相同
    seed_everything(seed)

    csv_path = f'../data/corpus.txt'
    model_name = 'Meta-Llama-3.1-8B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2', device_map=device)
    balanced_df = balance_dataset(csv_path, model, tokenizer, device)
    balanced_df.to_csv('../data/train_data_aug.csv', index=False)

if __name__ == '__main__':
    main()