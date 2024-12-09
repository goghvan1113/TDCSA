from collections import Counter

from openai import OpenAI, APITimeoutError
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
import time

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

deepseek_api = "sk-47cf9e5ebda644b4b8dd48e5a9c1268d"
deepseek_url = "https://api.deepseek.com/v1"
deepseek_model = "deepseek-chat"
deepbricks_api = "sk-ybgZNYqegwDjhGRDZKIHOYoYQLWTCSLh52Qbv0uF81J0U3n0"
deepbricks_url = "https://api.deepbricks.ai/v1/"
deepbricks_model = "gpt-3.5-turbo"

shot6_prompt = '''
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
    1. "In fact, researchers in sentiment analysis have realized benefits by decomposing the problem into S/O and polarity classification  ."
       Reasoning: Direct mention of positive outcomes ("benefits")
       Label: positive

    2. "Another successful distributed network model, called the beat frequency model, uses beats between multiple oscillators to produce a much wider range of durations than the intrinsic periods of individual oscillators  ."
       Reasoning: Explicit praise with "successful" and highlights advantages
       Label: positive

    Neutral:
    1. "ang and Lee   applied two different classifiers to perform sentiment annotation in two sequential steps: the first classifier separated subjective   texts from objective   ones and then they used the second classifier to classify the subjective texts into positive and negative"
       Reasoning: Pure methodological description without evaluation
       Label: neutral

    2. "Several teams had approaches that relied   on an IBM model of statistical machine translation  , with different improvements brought by different teams, consisting of new submodels, improvements in the HMM model, model combination for optimal alignment, etc. Se-veral teams used symmetrization metrics, as introduced in    , most of the times applied on the alignments produced for the two directions sourcetarget and targetsource, but also as a way to combine different word alignment systems."
       Reasoning: Factual reporting of different approaches without judgment
       Label: neutral

    Negative:
    1. "Although, there are various manual/automatic evaluation methods for these systems, e.g., BLEU  , these methods are basically incapable of dealing with an MTsystem and a w/p-MT-system at the same time, as they have different output forms."
       Reasoning: Directly criticizes methods' limitations using "incapable"
       Label: negative

    2. "Some other recent work has focused on the problem of implicit citation extraction (Kaplan et al., 2009; Qazvinian and Radev, 2010). Kaplan et al. (2009) explore co-reference chains for citation extraction using a combination of co-reference resolution techniques (Soon et al., 2001; Ng and Cardie, 2002). However, the corpus that they use consists of only 94 citations to 4 papers and is likely to be too small to be representative."
       Reasoning: Questions data representativeness, criticizes sample size
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

def citation_classify(text, model, tokenizer, device, with_api=True):
    shot0_prompt = '''
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
        {'role': 'system', 'content': shot0_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    if with_api:
        client = OpenAI(
            base_url=deepbricks_url,
            api_key=deepbricks_api
        )
        response = client.chat.completions.create(
            model=deepbricks_model,
            messages=messages,
            max_tokens=16,
            temperature=0.1,
            top_p=0.95,
            stream=False,
        )
        response = response.choices[0].message.content
    else:
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

    response = response.lower().strip()
    print(response)

    if response == 'negative':
        pred_label = 2
    elif response == 'positive':
        pred_label = 1
    else:
        pred_label = 0

    return pred_label

def process_dataset(file_path, model, tokenizer, device, checkpoint_dir=None, with_api=True):
    if checkpoint_dir is None:
        checkpoint_dir = "../test/checkpoints_deepseek"

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "classification_checkpoint.json")

    # Load checkpoint if exists
    start_idx = 0
    pred_labels = []
    real_labels = []

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            start_idx = checkpoint_data['last_processed_index'] + 1
            pred_labels = checkpoint_data['pred_labels']
            real_labels = checkpoint_data['real_labels']
            print(f"Resuming from index {start_idx}")
    else:
        # Load dataset
        texts, real_labels = load_sentiment_datasets(filepath=file_path, is_split=False)
        # Map labels
        label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
        labels = [label_map[label] for label in real_labels]

        print("Total texts: ", len(texts))
        print(f"Positive labels: {real_labels.count(1)}, Negative labels: {real_labels.count(2)}, Neutral labels: {real_labels.count(0)}")

    max_retries = 3
    texts, _ = load_sentiment_datasets(filepath=file_path, is_split=False)

    try:
        for idx in tqdm(range(start_idx, len(texts)),
                        desc="Classifying texts",
                        initial=start_idx,
                        total=len(texts)):
            text = texts[idx]
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Call citation_classify for the single text
                    pred_label = citation_classify(text, model, tokenizer, device, with_api=with_api)
                    pred_labels.append(pred_label)
                    break  # Break retry loop on success

                except APITimeoutError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"\nTimeout error on item {idx}. Retrying ({retry_count}/{max_retries})...")
                        time.sleep(5)
                    else:
                        print(f"\nMax retries exceeded for item {idx}. Saving checkpoint and exiting.")
                        raise e

            # Save checkpoint every 10 items
            if idx % 10 == 0:
                checkpoint_data = {
                    'last_processed_index': idx,
                    'pred_labels': pred_labels,
                    'real_labels': real_labels,
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # Save checkpoint on any error
        print(f"\nError occurred at index {idx}: {str(e)}")
        checkpoint_data = {
            'last_processed_index': idx - 1,
            'pred_labels': pred_labels,
            'real_labels': real_labels,
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        raise e

    # Clean up checkpoint file if processing completed successfully
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return real_labels, pred_labels

def main():
    seed = 42
    seed_everything(seed)

    file_path = '../data/citation_sentiment_corpus_expand.csv'
    model_name = 'Meta-Llama-3.2-3B-Instruct'
    model_dir = f'../pretrain_models/{model_name}'
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # model = AutoModelForCausalLM.from_pretrained(model_dir,
    #                                              torch_dtype=torch.bfloat16,
    #                                              attn_implementation='flash_attention_2',
    #                                              device_map='auto')
    model = None

    real_labels, pred_labels = process_dataset(file_path, model, tokenizer, device, with_api=True)

    report = classification_report(real_labels, pred_labels, target_names=['neutral', 'positive', 'negative'], digits=4)
    print(report)

    plot_confusion_matrix(real_labels, pred_labels, ['neutral', 'positive', 'negative'], title='Confusion Matrix')

if __name__ == '__main__':
    main()
