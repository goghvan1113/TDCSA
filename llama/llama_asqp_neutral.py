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

nvidia_model = "meta/llama-3.1-405b-instruct"
nvidia_url = "https://integrate.api.nvidia.com/v1"
api_key_1 = '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'  ## 1663653541
api_key_2 = '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'  # gaof23@mails
api_call_count = 0
max_api_calls = 950
deepseek_api = "sk-47cf9e5ebda644b4b8dd48e5a9c1268d"
deepseek_url = "https://api.deepseek.com/v1"
deepseek_model = "deepseek-chat"
deepbricks_api = "sk-ybgZNYqegwDjhGRDZKIHOYoYQLWTCSLh52Qbv0uF81J0U3n0"
deepbricks_url = "https://api.deepbricks.ai/v1/"
deepbricks_model = "gpt-3.5-turbo"


def extract_sentiment_quadruples(text, sentiment, tokenizer, model, device, with_api=False):

    nvidia_api = api_key_2
    system_prompt = '''
            You are an AI assistant specialized in analyzing neutral scientific citations and extracting aspect sentiment quadruples. Your primary focus is on objective, descriptive statements in academic writing, while remaining sensitive to subtle evaluative elements. Follow these guidelines:

            Element Definitions:
            -An "aspect term" refers to:
                * Methods, tools, or systems being used
                * Datasets or corpora
                * Experimental settings
                * Technical components or parameters
                * Metrics or measurements

            -An "opinion term" in neutral scientific writing typically appears as:
                * Technical descriptions (e.g., "is defined as", "consists of", "is specified by")
                * Process descriptions (e.g., "was trained on", "was evaluated using")
                * Quantitative statements (e.g., "contains X components", "uses Y parameters")
                * Comparative descriptions (e.g., "differs from", "is similar to")

            -The "aspect category" remains:
                * METHODOLOGY (methods, algorithms, techniques, approaches)
                * PERFORMANCE (results, metrics, measurements)
                * INNOVATION (novel aspects, contributions)
                * APPLICABILITY (use cases, applications)
                * LIMITATION (constraints, requirements)

            -The "sentiment polarity" can be:
                * "neutral" - For purely descriptive statements about methods, data, or processes
                * "positive" - When implicit positive evaluation exists (e.g., "standard dataset", "widely used method")
                * "negative" - When implicit negative evaluation exists (e.g., "requires substantial resources")

            Special Attention Points:
            1. Default to Neutral for:
               - Pure technical descriptions
               - Experimental setup details
               - Data processing steps
               - System components description

            2. Look for Hidden Sentiment in:
               - Terms implying authority ("standard", "widely-used", "established")
               - Scale indicators ("large-scale", "extensive", "comprehensive")
               - Efficiency markers ("efficient", "fast", "lightweight")
               - Resource requirements ("requires", "needs", "demands")

            3. Context Considerations:
               - Academic writing often uses understated language
               - Technical superiority might be implied through comparison
               - Resource requirements might imply limitations
               - Implementation details might suggest complexity
            
            Examples:
            Text: "To tackle this problem, we defined 2The best results of Collins and Roark are achieved when the parser utilizes the information about the final punctuation and the look-ahead."
            Quadruples: [("parser", "utilizes information about final punctuation and look-ahead", "METHODOLOGY", "neutral", 0.9), ("results", "best when utilizing information", "PERFORMANCE", "positive", 0.85)]
    
            Text: "In the second pass, 5-gram and 6-gram zero-cutoff stupid-backoff language models estimated using 4.7 billion words of English newswire text are used to generate lattices for phrasal segmentation model rescoring."
            Quadruples: [("language models", "5-gram and 6-gram zero-cutoff stupid-backoff", "METHODOLOGY", "neutral", 0.95)]
    
            Text: "In this paper, we give an overview of NLPWin, a multi-application natural language analysis and generation system under development at Microsoft Research, incorporating analysis systems for 7 languages."
            Quadruples: [("NLPWin", "multi-application natural language analysis and generation system", "METHODOLOGY", "neutral", 0.95), ("analysis systems", "incorporating 7 languages", "APPLICABILITY", "positive", 0.85)]
        '''

    user_prompt = f'''
        Analyze this scientific citation that has been labeled as having neutral overall sentiment. Extract up to 2 sentiment quadruples, paying special attention to both purely descriptive elements and any subtle evaluative content that might be present:

        Text: {text}
        Overall Sentiment: {sentiment}

        Provide your response strictly as a Python list of tuples:
        [("aspect term", "opinion term", "aspect category", "sentiment polarity", confidence score), ("aspect term", "opinion term", "aspect category", "sentiment polarity", confidence score)]

        Only include the list of quadruples in your response, with no additional text.
        '''
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    if with_api:
        client = OpenAI(
            base_url=nvidia_url,
            api_key=nvidia_api
        )
        response = client.chat.completions.create(
            model=nvidia_model,
            messages=messages,
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            stream=False,
        )
        response = response.choices[0].message.content
        print(response)
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
        return quadruples
    except:
        print("Error in extracting sentiment quadruples")
        print(response)
        return []


# 算是chain of thoughts模式
def verify_quadruples_quality(text, quadruples, tokenizer, model, device, with_api=False):

    global api_call_count
    if api_call_count < max_api_calls:
        nvidia_api = api_key_1
    else:
        nvidia_api = api_key_2
    api_call_count += 1

    system_prompt = '''You are a critical quadruple verification system for scientific citations. Your role is to analyze quadruples through step-by-step reasoning and provide structured validation results.

    Verification Process:
    For each quadruple, follow these steps:
    1. First state the quadruple being analyzed
    2. Conduct evidence analysis
    3. Perform relationship reasoning
    4. Make validity assessment
    5. Provide structured validation result

    Then combine all validations into a final JSON response.
    '''

    user_prompt = f'''Citation: {text}
    Quadruples: {quadruples}

    For each quadruple, provide your analysis in this format:

    **Quadruple Analysis [number]:**
    Input: [aspect, opinion, category, polarity, confidence]

    **Evidence Analysis:**
    - Identify exact text matches
    - Analyze surrounding context

    **Relationship Reasoning:**
    - Aspect-Opinion connection
    - Category appropriateness
    - Sentiment justification

    **Validity Assessment:**
    - Determine validity
    - List any issues
    - Propose corrections if needed

    **Validation Result:**
    {{
        "quadruple": [aspect, opinion, category, polarity],
        "is_valid": boolean,
        "confidence": float,
        "issues": [] or ["Type: description"],
        "correction": null or [new_aspect, new_opinion, new_category, new_polarity]
    }}

    After analyzing all quadruples, provide:

    **Final Combined Response:**
    {{
        "validations": [
            // array of all validation results
        ],
        "overall_quality": float
    }}

    Requirements:
    1. Maintain exact text matches for aspects and opinions
    2. Use only predefined categories
    3. Show clear reasoning in each step
    4. Ensure JSON format in validation results
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
            top_p=0.95,
            stream=False,
        )
        response = response.choices[0].message.content
        # **JSON Response**作为字符串的分割，前半部分是reasoning，后半部分是json response
        reasoning = response.split('**Final Combined Response:**')[0]
        validation_results = response.split('**Final Combined Response:**')[1]
        # validation_results = validation_results.strip('`json\n')  # 移除开头的```json
        # validation_results = validation_results.strip('`\n')  # 移除结尾的``
        # print(response)

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
            top_p=0.95,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_input.input_ids, generated_ids)]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(response)
        # **JSON Response**作为字符串的分割，前半部分是reasoning，后半部分是json response
        reasoning = response.split('**Final Combined Response:**')[0]
        validation_results = response.split('**Final Combined Response:**')[1]
        # validation_results = validation_results.strip('`json\n')  # 移除开头的```json
        # validation_results = validation_results.strip('`\n')  # 移除结尾的``
        print(validation_results)

    try:
        validation_results = json.loads(validation_results) #是{}格式
        return validation_results, reasoning
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


def process_dataset(file_path, tokenizer, model, device, checkpoint_dir=None):

    if checkpoint_dir is None:
        checkpoint_dir = "../test/checkpoints_deepseek"

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
    df = df[df['Sentiment'].isin(['neutral'])]
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
                                                              with_api=False)
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
                                      initial_output_dir=None, final_output_dir=None, checkpoint_dir=None):
    """
    Process the dataset with both extraction and verification steps, including error handling and checkpointing

    Args:
        checkpoint_dir: Directory to save checkpoint files during processing
    """
    # First get the initial extractions
    # initial_results = process_dataset(file_path, tokenizer, extractor_model, device)
    # save_results(initial_results, initial_output_dir)

    if checkpoint_dir is None:
        checkpoint_dir = "../test/checkpoints_deepseek"

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
                        with_api=False
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
    initial_output_dir = '../output/sentiment_asqp_results_corpus_expand_llama8b_neutral.json'
    final_output_dir = '../output/sentiment_asqp_results_corpus_expand_verified_llama.json'
    extractor_model_name = 'Meta-Llama-3-8B-Instruct'
    verifier_model_name = 'Meta-Llama-3.1-8B-Instruct'
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(f'../pretrain_models/{extractor_model_name}')
    extractor_model = AutoModelForCausalLM.from_pretrained(
        f'../pretrain_models/{extractor_model_name}',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map=device
    )
    # extractor_model = None #不加载到显存里面
    # verifier_model = AutoModelForCausalLM.from_pretrained(
    #     f'../pretrain_models/{verifier_model_name}',
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation='flash_attention_2',
    #     device_map=device
    # )
    # verifier_model = None

    initial_results = process_dataset(file_path, tokenizer, extractor_model, device)
    save_results(initial_results, initial_output_dir)

    # results, quality_metrics = process_dataset_with_verification(
    #     file_path,
    #     extractor_model,
    #     verifier_model,
    #     tokenizer,
    #     device,
    #     initial_output_dir,
    #     final_output_dir,
    # )
    # print("Quality metrics:", quality_metrics)

    # 'average_confidence': 0.8289524599226354,
    # 'total_original_quadruples': 3618,
    # 'total_valid_quadruples': 2032,
    # 'total_corrected_quadruples': 1586,
    # 'invalid_quadruples': 1586



if __name__ == '__main__':
    main()