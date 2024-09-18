from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import json


seed = 42
model_name = 'Qwen2-7B-Instruct'
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
    print("Positive labels: , Negative labels: ,Neutral labels: ", real_labels.count(1), real_labels.count(2),
          real_labels.count(0))

    pred_labels = []
    exception_responses = []

    # prompt source: https://github.com/aielte-research/LlamBERT/blob/main/LLM/model_inputs/IMDB/promt_eng_0-shot_prompts.json
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
            max_new_tokens=512,
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
    precision = precision_score(real_labels, pred_labels)
    recall = recall_score(real_labels, pred_labels)
    mcc = matthews_corrcoef(real_labels, pred_labels)

    df = pd.read_csv('./output/llm_results.csv')

    df = df._append(
        {
            'model_name': 'qwen2-7b',
            'seed': seed,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc
        },
        ignore_index=True)
    df.to_csv('./output/llm_results.csv', index=False)

    print("exception_responses: ", exception_responses)


def label_unsupervised():
    df = pd.read_csv('./data/citation_paper_contexts_unlabeled.csv')
    texts = df['text'].tolist()

    pred_labels = []
    exception_responses = []

    # prompt source:
    for text in tqdm(texts):
        print(f'Enter a prompt to generate a response:')
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
            max_new_tokens=512,
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

    json.dump(pred_labels,
              open('output/unsupervised_labels.json', 'w'), indent=2)
    print("exception_responses: ", exception_responses)


if __name__ == '__main__':
    seed_everything(seed)
    test()
    # label_unsupervised()