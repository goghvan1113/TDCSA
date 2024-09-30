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
                                             torch_dtype='auto',
                                             device_map=device)

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

def test():
    df = pd.read_csv('./data/citation_sentiment_corpus_new.csv')
    texts = df['Citation_Text'].tolist()
    real_labels = df['Sentiment'].tolist()

    print("Total texts: ", len(texts))
    print(f"Positive labels: {real_labels.count(1)}, Negative labels: {real_labels.count(2)},Neutral labels: {real_labels.count(0)}")

    pred_labels = []
    exception_responses = []

    # prompt source: https://github.com/aielte-research/LlamBERT/blob/main/LLM/model_inputs/IMDB/promt_eng_0-shot_prompts.json
    for text in tqdm(texts):
        system_prompt = \
        f'''
            You are an expert in scientific citation sentiment analysis who can judge the attitude of a citation text. Please answer with \'positive\' , \'neutral\' or \'negative\' only!
            
            Here are some examples to guide your judgment:
            1. Citation: "This study provides significant insights into the field and supports the growing body of evidence on climate change." 
               Sentiment: positive
               
            2. Citation: "While the methodology is sound, the results are largely inconclusive and require further investigation."
               Sentiment: neutral
               
            3. Citation: "The analysis lacks rigor and fails to account for critical variables, making the conclusions unreliable."
               Sentiment: negative
               
           If the citation text is positive, indicating approval, agreement, or support for the research, please answer \'positive\'. If the citation text is neutral, meaning it does not express a clear opinion or sentiment, please answer \'neutral\'. If the citation text is negative, indicating criticism, disagreement, or rejection of the research, please answer \'negative\'. Make your decision based on the overall tone and content of the citation. If the sentiment is unclear, default to \'neutral\'.
        '''
        user_prompt = f'Decide if the following scientific citation text expresses a positive, neutral, or negative sentiment towards the research or findings: \n {text} \n'

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

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

    acc = accuracy_score(real_labels, pred_labels)
    f1 = f1_score(real_labels, pred_labels, average='macro')
    precision = precision_score(real_labels, pred_labels, average='macro')
    recall = recall_score(real_labels, pred_labels, average='macro')

    file_path = './output/llm_results.csv'
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['model_name', 'seed', 'accuracy', 'f1', 'precision', 'recall'])
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)

    df = df._append(
        {
            'model_name': model_name,
            'seed': seed,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'shots': 3
        },
        ignore_index=True)
    df.to_csv('./output/llm_results.csv', index=False)

    print("exception_responses: ", exception_responses)


def label_unsupervised():
    df = pd.read_csv('./data/citing_paper_contexts_unlabeled.csv')
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
    seed_everything(seed)
    test()
    # label_unsupervised()