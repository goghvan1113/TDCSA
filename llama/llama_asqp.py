import argparse
from collections import Counter, defaultdict

import torch
from openai import OpenAI, max_retries
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random
import numpy as np
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from openai import APITimeoutError
import time

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


deepseek_api = "sk-47cf9e5ebda644b4b8dd48e5a9c1268d"
deepseek_url = "https://api.deepseek.com/v1"
deepseek_model = "deepseek-chat"
deepbricks_api = "sk-ybgZNYqegwDjhGRDZKIHOYoYQLWTCSLh52Qbv0uF81J0U3n0"
deepbricks_url = "https://api.deepbricks.ai/v1/"
deepbricks_model = "LLama-3.1-70b"


def extract_sentiment_quadruples(text, sentiment, tokenizer, model, device, with_api=False):
    system_prompt = '''
        You are an AI assistant specialized in analyzing scientific citations and extracting aspect sentiment quadruples according to the following elements definition:

        -An "aspect term" refers to a specific contribution, method, finding, or component of the cited work that appears explicitly as a substring in the citation text
        -An "opinion term" refers to the evaluation or assessment expressed towards a particular aspect of the cited work, appearing explicitly as a substring in the citation text
        -The "aspect category" indicates the broad category that the aspect term belongs to, chosen from a predefined set:
            * METHODOLOGY (methods, algorithms, techniques, frameworks, or approaches)
            * PERFORMANCE (results, efficiency, accuracy, effectiveness, or any measurable outcomes)
            * INNOVATION (novelty, contributions, or advances in the field)
            * APPLICABILITY (practical value, usefulness, or real-world applications)
            * LIMITATION (drawbacks, constraints, weaknesses, or problems)
        -The "sentiment polarity" indicates the author's attitude towards the cited aspect: "positive" or "negative"

        Extraction Guidelines:
        1. Only extract up to 2 sentiment quadruples per citation.
        2. Language Patterns:
            Aspects: noun phrases, technical terms, research components
            Opinions: adjectives, verb phrases, adverb-adjective pairs
        3. Ensure the aspect and opinion are explicitly mentioned in the text.
        4. Aspect category must be one of the predefined categories, verify aspect-category alignment based on definitions.
        5. Sentiment polarity must be either "positive" or "negative", consider academic context for polarity judgment.
        6. Ignore purely objective statements without evaluative content.

        Examples:
        Text: Substantial improvements have been made to parse western language such as English, and many powerful models have been proposed.
        Sentiment Quadruples:
        [("improvements", "substantial", "PERFORMANCE", "positive", 0.95), ("models", "powerful", "METHODOLOGY", "positive", 0.92)]

        Text: However, one of the major limitations of these advances is the structured syntactic knowledge, which is important to global reordering, has not been well exploited.
        Sentiment Quadruples:
        [("structured syntactic knowledge", "has not been well exploited", "LIMITATION", "negative", 0.90), ("syntactic knowledge", "important", "METHODOLOGY", "positive", 0.88)]

        Special Cases to Consider:
        1. Implicit Opinions:
        When opinions are implied through comparison:
        "Their method performs better than baseline approaches."
        Sentiment Quadruples:
        [("method", "performs better", "PERFORMANCE", "positive", 0.93)]

        2. Compound Aspects:
        When multiple aspects are mentioned together:
        "The model architecture and training strategy are innovative."
        Sentiment Quadruples:
        [("model architecture", "innovative", "INNOVATION", "positive", 0.91), ("training strategy", "innovative", "INNOVATION", "positive", 0.91)]

        3. Context-Dependent Polarity:
        Consider academic writing context:
        "This simple approach..." could be positive (elegance) or negative (oversimplified) depending on context.
        '''

    user_prompt = f'''
        Analyze the sentiment elements in this scientific citation text. Provide exactly up to 2 quadruples containing (aspect term, opinion term, aspect category, sentiment polarity, confidence score) for the following citation:

        Text: {text}
        Overall Sentiment: {sentiment}

        Provide your response in the format of a Python list of tuples:
        [("aspect term", "opinion term", "aspect category", "sentiment polarity", confidence score), ("aspect term", "opinion term", "aspect category", "sentiment polarity", confidence score)]

        Only include the list of quadruples in your response, with no additional text.
        '''
    messages = [
        {'role': 'system', 'content': system_prompt},
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
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            stream=False,
        )
        response = response.choices[0].message.content
        response = response.strip('`python\n')
        response = response.strip('`\n')

    else:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_input = tokenizer([input_text], return_tensors='pt').to(device)

        attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
        generated_ids = model.generate(
            model_input.input_ids,
            max_new_tokens=128,
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_input.input_ids, generated_ids)]  # remove the input text
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(response)
    try:
        # Convert string to Python list using eval()
        quadruples = eval(response)
        filtered_quadruples = []
        for quad in quadruples:
            if quad[-1] >= 0.5:  # 基础置信度过滤
                filtered_quadruples.append(quad)
        return filtered_quadruples
    except:
        print("Error in extracting sentiment quadruples")
        print(response)
        return []


def need_verification(quad, text_features):
    """确定四元组是否需要验证"""
    def has_sentiment_conflict(aspect, polarity, all_quads):
        """检查是否存在情感冲突"""
        for q in all_quads:
            if q[0] == aspect and q[3] != polarity:
                return True
        return False

    def is_critical_category(category):
        """判断是否是关键类别"""
        critical_categories = {'METHODOLOGY', 'PERFORMANCE'}
        return category in critical_categories

    def has_low_confidence(confidence):
        """检查置信度"""
        return confidence < 0.8

    # 验证触发条件
    triggers = {
        'sentiment_conflict': has_sentiment_conflict(quad[0], quad[3], text_features),
        'critical_category': is_critical_category(quad[2]),
        'low_confidence': has_low_confidence(quad[4])
    }

    return any(triggers.values()), triggers

# 算是chain of thoughts模式
def verify_quadruples_quality(text, quadruples, tokenizer, model, device, with_api=False):

    verified_results = {'validations': []}
    for quad in quadruples:
        needs_verification, triggers = need_verification(quad, quadruples)

        if not needs_verification:
            # 对于不需要验证的四元组，直接添加到结果中
            verified_results['validations'].append({
                'quadruple': quad[:4],  # 不包含置信度
                'relationships': {
                    'aspect_opinion_validity': 1.0,
                    'aspect_category_validity': 1.0,
                    'opinion_sentiment_validity': 1.0
                },
                'is_valid': True,
                'confidence': quad[4],
                'issues': [],
                'correction': None
            })
            continue

        # 根据触发条件调整验证深度
        verification_depth = 'deep' if triggers['sentiment_conflict'] else 'standard'

        # 设置较低的温度以确保稳定性
        verification_settings = {
            'temperature': 0.3,
            'top_p': 0.6,
            'max_tokens': 2048 if verification_depth == 'deep' else 1024
        }

    system_prompt = f'''You are a scientific citation quadruple verifier. Your task is to carefully validate whether opinion terms appropriately describe aspects, aspects fit their categories, and opinions align with sentiment polarities in academic writing.

    For each quadruple, analyze the three key relationships:
    1. Aspect-Opinion: Verify the opinion meaningfully describes or evaluates the aspect 
    2. Aspect-Category: Check if the aspect is appropriately categorized
    3. Opinion-Sentiment: Confirm the opinion's sentiment aligns with the assigned polarity

    Provide validation results in the exact required format.'''

    user_prompt = f'''
    Citation: {text}
    Quadruples to verify: {quadruples}

    For each quadruple, follow this analysis process:

    Step 1 - Check Aspect-Opinion Relationship:
    - Does the opinion clearly describe or evaluate the aspect?
    - Is the relationship supported by the citation context?

    Step 2 - Verify Aspect-Category Match:
    - Is the aspect appropriately categorized? 
    - Does the category reflect the aspect's role in the citation?

    Step 3 - Validate Opinion-Sentiment:
    - Does the opinion's meaning match the assigned sentiment?
    - Consider both explicit and implicit sentiment in academic context

    Then provide this exact JSON response:
    {{
        "validations": [
            {
                "quadruple": [aspect, opinion, category, polarity],
                "is_valid": true/false,
                "confidence": float,
                "issues": [] or ["relationship_type: issue"],
                "correction": {
                    "aspect": null or "new_aspect",
                    "opinion": null or "new_opinion", 
                    "category": null or "new_category",
                    "polarity": null or "new_polarity",
                    "reason": "explanation"
                }
            }
        ]
    }}
    
    The JSON object：json
    '''

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    if with_api: # 使用api
        client = OpenAI(
            base_url=deepseek_url,
            api_key=deepseek_api
        )
        response = client.chat.completions.create(
            model=deepseek_model,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.6,
            stream=False,
        )
        response = response.choices[0].message.content

    else:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_input = tokenizer([input_text], return_tensors='pt').to(device)

        attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
        generated_ids = model.generate(
            model_input.input_ids,
            max_new_tokens=2048,  # Increased for detailed feedback
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,  # Lower temperature for more focused response
            top_p=0.6,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_input.input_ids, generated_ids)]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # **JSON Response**作为字符串的分割，前半部分是reasoning，后半部分是json response
    print(response)

    validation_results = response.strip('`json\n')  # 移除开头的```json
    validation_results = validation_results.strip('`\n')  # 移除结尾的``
    print(validation_results)

    try:
        validation_results = json.loads(validation_results) #是{}格式
        return validation_results
    except json.JSONDecodeError as e: # 解析llm返回的代码错误
        print(f"JSON parsing error: {e}")
        print("Response:", response)
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Response:", response)
        return None

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


def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_dataset(file_path, tokenizer, model, device, checkpoint_dir=None, with_api=True):

    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "processing_checkpoint.json")

    # Load checkpoint if exists
    start_idx = 0
    results = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            start_idx = checkpoint_data['last_processed_index'] + 1
            results = checkpoint_data['results']
            print(f"Resuming from index {start_idx}")

    sentences, labels = load_sentiment_datasets(filepath=file_path, is_split=False)
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    labels = [label_map[label] for label in labels]
    df = pd.DataFrame({'Citation_Text': sentences, 'Sentiment': labels})
    df = df[df['Sentiment'].isin(['positive', 'negative'])]
    # df = df.head(5) # 测试用

    max_retries = 3
    try:
        for idx in tqdm(range(start_idx, len(df)),
                        desc="Extracting sentiment quadruples",
                        initial=start_idx,
                        total=len(df)):
            item = df.iloc[idx]
            retry_count = 0

            while retry_count < max_retries:
                try:
                    quadruples = extract_sentiment_quadruples(item['Citation_Text'],
                                                              item['Sentiment'],
                                                              tokenizer,
                                                              model,
                                                              device,
                                                              with_api=with_api)
                    results.append({
                        'text': item['Citation_Text'],
                        'overall_sentiment': item['Sentiment'],
                        'sentiment_quadruples': quadruples
                    })
                    break  # Break the retry loop if successful

                except APITimeoutError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"\nTimeout error on item {idx}. Retrying ({retry_count}/{max_retries})...")
                        time.sleep(5)  # Wait 5 seconds before retrying
                    else:
                        print(f"\nMax retries exceeded for item {idx}. Saving checkpoint and exiting.")
                        raise e

            # Save checkpoint every 10 items
            if idx % 10 == 0:
                checkpoint_data = {
                    'last_processed_index': idx,
                    'results': results
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # Save checkpoint on any error
        print(f"\nError occurred at index {idx}: {str(e)}")
        checkpoint_data = {
            'last_processed_index': idx - 1,  # Save the last successfully processed index
            'results': results
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        raise e

    return results


def process_dataset_with_verification(file_path, extractor_model, verifier_model, tokenizer, device,
                                      initial_output_dir=None, final_output_dir=None, checkpoint_dir=None, with_api=True):
    """
    Process the dataset with both extraction and verification steps, including error handling and checkpointing

    Args:
        checkpoint_dir: Directory to save checkpoint files during processing
    """
    # First get the initial extractions
    # initial_results = process_dataset(file_path, tokenizer, extractor_model, device)
    # save_results(initial_results, initial_output_dir)

    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "processing_checkpoint.json")

    # Load checkpoint if exists
    start_idx = 0
    results = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            start_idx = checkpoint_data['last_processed_index'] + 1
            results = checkpoint_data['results']
            print(f"Resuming from index {start_idx}")

    # Load initial results
    initial_results = load_results(initial_output_dir)

    # Process remaining items
    max_retries = 3  # Maximum number of retry attempts for each item

    try:
        for idx in tqdm(range(start_idx, len(initial_results)),
                        desc="Verifying extractions",
                        initial=start_idx,
                        total=len(initial_results)):
            item = initial_results[idx]
            retry_count = 0

            while retry_count < max_retries:
                try:
                    verification, reasoning = verify_quadruples_quality(
                        item['text'],
                        item['sentiment_quadruples'],
                        tokenizer,
                        verifier_model,
                        device,
                        with_api=with_api
                    )
                    # Process verification results
                    verified_result = {
                        'text': item['text'],
                        'overall_sentiment': item['overall_sentiment'],
                        'original_quadruples': item['sentiment_quadruples'],
                        'verification_results': verification,
                        'final_quadruples': []
                    }

                    # Only keep quadruples that pass verification
                    if verification and 'validations' in verification:
                        for val in verification['validations']:
                            if val['is_valid']:
                                verified_result['final_quadruples'].append(val['quadruple'])
                            elif val['correction']:
                                verified_result['final_quadruples'].append(val['correction'])

                    results.append(verified_result)
                    break  # Break the retry loop if successful

                except APITimeoutError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"\nTimeout error on item {idx}. Retrying ({retry_count}/{max_retries})...")
                        time.sleep(5)  # Wait 5 seconds before retrying
                    else:
                        print(f"\nMax retries exceeded for item {idx}. Saving checkpoint and exiting.")
                        raise e

            # Save checkpoint every 10 items
            if idx % 10 == 0:
                checkpoint_data = {
                    'last_processed_index': idx,
                    'results': results
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # Save checkpoint on any error
        print(f"\nError occurred at index {idx}: {str(e)}")
        checkpoint_data = {
            'last_processed_index': idx - 1,  # Save the last successfully processed index
            'results': results
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        raise e

    # Save final results
    if final_output_dir:
        save_results(results, final_output_dir)

    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(results)

    # Clean up checkpoint file if processing completed successfully
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return results, quality_metrics


def calculate_quality_metrics(verified_results):
    """
    Calculate overall quality metrics for the verified extractions
    """
    metrics = {
        'total_original_quadruples': 0,
        'total_valid_quadruples': 0,
        'total_corrected_quadruples': 0,
        'invalid_quadruples': 0,
        'category_confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'common_issues': defaultdict(int),
        'average_confidence': 0.0,
    }

    for result in verified_results:
        try:
            if 'verification_results' and 'validations' in result['verification_results']:
                validations = result['verification_results']['validations']
                metrics['total_original_quadruples'] += len(result['original_quadruples'])
                metrics['total_valid_quadruples'] += sum(1 for v in validations if v['is_valid'])
                metrics['total_corrected_quadruples'] += sum(
                    1 for v in validations if not v['is_valid'] and v['correction'])
                metrics['invalid_quadruples'] += sum(1 for v in validations if not v['is_valid'])

                # Track common issues
                for validation in validations:
                    for issue in validation['issues']:
                        metrics['common_issues'][issue] += 1
                    metrics['average_confidence'] += validation['confidence']
        except TypeError as e:
            print(f"TypeError: {e}")
            print("Skipping this result due to error.")

    # Calculate averages
    total_validations = sum(
        len(r['verification_results']['validations']) for r in verified_results if r['verification_results'])
    if total_validations > 0:
        metrics['average_confidence'] /= total_validations

    return metrics


def main():
    seed = 42
    seed_everything(seed)

    file_path = '../data/citation_sentiment_corpus_expand.csv'
    initial_output_dir = '../output/sentiment_asqp_results_corpus_expand_llama405b.json'
    final_output_dir = '../output/sentiment_asqp_results_corpus_expand_verified_deepseek.json'
    extractor_model_name = 'Meta-Llama-3-8B-Instruct'
    verifier_model_name = 'Meta-Llama-3.1-8B-Instruct'
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(f'D:/llm/{extractor_model_name}')
    # extractor_model = AutoModelForCausalLM.from_pretrained(
    #     f'../pretrain_models/{extractor_model_name}',
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation='flash_attention_2',
    #     device_map=device
    # )
    extractor_model = None #不加载到显存里面
    # verifier_model = AutoModelForCausalLM.from_pretrained(
    #     f'D:/llm/{verifier_model_name}',
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation='flash_attention_2',
    #     device_map=device
    # )
    verifier_model = None

    # initial_results = process_dataset(file_path, tokenizer, extractor_model, device, with_api=True)
    # save_results(initial_results, initial_output_dir)

    results, quality_metrics = process_dataset_with_verification(
        file_path,
        extractor_model,
        verifier_model,
        tokenizer,
        device,
        initial_output_dir,
        final_output_dir,
        with_api=True
    )
    print("Quality metrics:", quality_metrics)


if __name__ == '__main__':
    main()