from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import json
import os

from bert.plot_results import plot_confusion_matrix


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 启用Cudnn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

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
    elif filepath == '../data/citation_sentiment_corpus_expand.csv' or filepath == '../data/citation_sentiment_corpus_expand_athar.csv':
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

def citation_classify(file_path, model, tokenizer, device):
    texts, real_labels = load_sentiment_datasets(filepath=file_path, is_split=False)
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    labels = [label_map[label] for label in real_labels]

    print("Total texts: ", len(texts))
    print(f"Positive labels: {real_labels.count(1)}, Negative labels: {real_labels.count(2)},Neutral labels: {real_labels.count(0)}")

    pred_labels = []
    exception_responses = []

    # prompt source: https://github.com/aielte-research/LlamBERT/blob/main/LLM/model_inputs/IMDB/promt_eng_0-shot_prompts.json
    for text in tqdm(texts):
        system_prompt = '''
        You are an expert in scientific citation sentiment analysis, specifically trained to evaluate the rhetorical intent and attitude expressed in academic citations. Your task is to classify citations as 'positive', 'neutral', or 'negative' based on careful linguistic and contextual analysis.

        CLASSIFICATION CRITERIA:

        1. Positive Citations:
           - Express strong agreement or endorsement
           - Highlight significant contributions or advances
           - Use explicit praise or positive evaluation
           - Build upon the cited work as foundation
           - Keywords: "significant", "innovative", "crucial", "successfully", "effective", "important contribution"

        2. Neutral Citations:
           - Factual references without evaluation
           - Background information or context
           - Methodology descriptions
           - Related work mentions
           - Balanced discussions with both pros and cons
           - Keywords: "reported", "investigated", "studied", "examined", "analyzed", "proposed"

        3. Negative Citations:
           - Express disagreement or criticism
           - Point out limitations or gaps
           - Question validity or reliability
           - Contrast with better approaches
           - Keywords: "limited", "fails to", "overlooks", "inadequate", "questionable", "inconsistent"

        CONTEXTUAL CONSIDERATIONS:
        - Consider hedging language and academic politeness
        - Evaluate both explicit and implicit criticism
        - Account for field-specific citation practices
        - Check for sarcasm or subtle criticism
        - Note temporal context (e.g., "at that time")

        EXAMPLES:

        Positive:
        1. "Smith et al.'s groundbreaking work (2020) established a robust framework that has significantly advanced our understanding of neural networks."
           Reasoning: Uses strong positive terms ("groundbreaking", "significantly advanced")
           Label: positive

        2. "The innovative approach proposed by Jones (2019) effectively resolved the long-standing challenges in quantum computing."
           Reasoning: Emphasizes innovation and successful problem-solving
           Label: positive

        Neutral:
        1. "The experiment used the methodology described in Brown (2018), employing a sample size of 500 participants."
           Reasoning: Purely descriptive, no evaluation
           Label: neutral

        2. "Previous studies have investigated this phenomenon using various approaches (Wang 2021; Lee 2022)."
           Reasoning: Simple acknowledgment of existing work
           Label: neutral

        Negative:
        1. "While Zhang's model (2021) attempts to address the issue, it fails to account for crucial environmental variables."
           Reasoning: Points out significant limitation
           Label: negative

        2. "The conclusions drawn by Miller (2019) are based on questionable assumptions and insufficient data."
           Reasoning: Direct criticism of methodology and conclusions
           Label: negative

        DECISION RULES:
        1. If the citation contains explicit positive evaluation → positive
        2. If the citation mainly reports facts or methods → neutral
        3. If the citation points out clear limitations or problems → negative
        4. If mixed but criticism outweighs praise → negative
        5. If mixed but praise outweighs criticism → positive
        6. If truly balanced or unclear → neutral

        Remember: Academic writing often uses subtle language. Look for:
        - Hedging words ("might", "could", "perhaps")
        - Contrast markers ("however", "although", "while")
        - Emphasis markers ("notably", "particularly", "especially")
        - Criticism softeners ("somewhat", "relatively", "rather")
        '''

        user_prompt = f'''
        Please analyze the following scientific citation and classify its sentiment as 'positive', 'neutral', or 'negative'. Consider both explicit and implicit sentiment markers, hedging language, and academic context:

        Citation text:
        {text}

        Provide only the sentiment label as response.
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
            max_new_tokens=16,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_input.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.lower()  # avoid Negative != negative
        print(response)

        try:
            if response == 'negative':
                pred_labels.append(2)
            elif response == 'positive':
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        except:
            print(f"Error in processing {text}, response: {response}")
            pred_labels.append(0)

    return real_labels, pred_labels


def label_unsupervised():
    df = pd.read_csv('../data/citing_paper_contexts_unlabeled.csv')
    texts = df['text'].tolist()

    pred_labels = []
    exception_responses = []

    # prompt source:
    for text in tqdm(texts):
        prompt = text

        messages = [{
            'role':
                'system',
            'content':
                'You are an expert in scientific citation sentiment analysis who can judge the attitude of a citation text. Please answer with \'positive\' , \'neutral\' or \'negative\' only!\n'
        }]
        prompt = f'Decide if the following scientific citation text expresses a positive, neutral, or negative sentiment towards the research or findings: \n {prompt} \nIf the citation text is positive, indicating approval, agreement, or support for the research, please answer \'positive\'. If the citation text is neutral, meaning it does not express a clear opinion or sentiment, please answer \'neutral\'. If the citation text is negative, indicating criticism, disagreement, or rejection of the research, please answer \'negative\'. Make your decision based on the overall tone and content of the citation. If the sentiment is unclear, default to \'neutral\'.'

        messages.append({'role': 'user', 'content': prompt})

        text = tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)

        model_input = tokenizer([text], return_tensors='pt').to(device)
        attention_mask = torch.ones(model_input.input_ids.shape,
                                    dtype=torch.long,
                                    device=device)
        generated_ids = model.generate(
            model_input.input_ids,
            max_new_tokens=16,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_input.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)[0]
        response = response.lower()  # avoid Negative != negative
        print(response)
        if response not in ['positive', 'negative', 'neutral']:
            exception_responses.append(response)
            response = 'neutral'
        if response == 'negative':
            pred_labels.append(2)
        elif response == 'positive':
            pred_labels.append(1)
        else:
            pred_labels.append(0)

        print(f'{response} \n')

    df['sentiment'] = pred_labels
    df.to_csv('./data/citing_paper_contexts_llm.csv', index=False)

    print("exception_responses: ", exception_responses)


if __name__ == '__main__':
    seed = 42
    seed_everything(seed)

    file_path = '../data/citation_sentiment_corpus_expand.csv'
    model_name = 'Meta-Llama-3.2-3B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2',
                                                 device_map=device)

    real_labels, pred_labels = citation_classify(file_path, model, tokenizer, device)
    report = classification_report(real_labels, pred_labels, target_names=['neutral', 'positive', 'negative'], digits=4)
    print(report)

    plot_confusion_matrix(real_labels, pred_labels, ['neutral', 'positive', 'negative'], title='Confusion Matrix')
