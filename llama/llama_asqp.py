import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from openai import OpenAI, APITimeoutError
import google.generativeai as genai
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM


class SentimentQuadrupleExtractor:
    def __init__(
            self,
            tokenizer,
            model=None,
            device='cuda:0',
            use_api=True,
            api_config=None
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.use_api = use_api

        # Default API configuration
        self.api_config = {
            'deepbricks_api': "sk-ybgZNYqegwDjhGRDZKIHOYoYQLWTCSLh52Qbv0uF81J0U3n0",
            'deepbricks_url': "https://api.deepbricks.ai/v1/",
            'deepbricks_model': "LLama-3.1-70b",
            'deepseek_api': "sk-47cf9e5ebda644b4b8dd48e5a9c1268d",
            'deepseek_url': "https://api.deepseek.com/v1",
            'deepseek_model': "deepseek-chat",
            'modelscope_api': "a0c23353-5040-49da-8b56-1f6d44c1f41c",
            'modelscope_url': "https://api-inference.modelscope.cn/v1/",
            'gemini_api': "AIzaSyAw36pxwPN8A6H3wJKFHSyG0hOGdYHmSto",
        }
        if api_config:
            self.api_config.update(api_config)

    def _need_verification(self, quadruples: List[Tuple], overall_sentiment: str) -> Tuple[bool, Dict[str, bool]]:
        """
        Determine if quadruples need verification based on specified conditions.

        Args:
            quadruples: List of sentiment quadruples
            overall_sentiment: Overall sentiment of the text

        Returns:
            Tuple of (needs_verification, triggers)
        """
        triggers = {
            'mixed_sentiment': False,
            'low_confidence': False,
            'sentiment_mismatch': False
        }

        # Check for mixed sentiments
        if len(quadruples) == 2:
            sentiments = [quad[3] for quad in quadruples]
            triggers['mixed_sentiment'] = len(set(sentiments)) > 1

        # Check for low confidence
        triggers['low_confidence'] = any(quad[4] < 0.8 for quad in quadruples)

        # Check for sentiment mismatch with overall sentiment
        if len(quadruples) >= 1:
            quad_sentiments = set(quad[3] for quad in quadruples)
            if len(quad_sentiments) == 1:  # All quadruples have same sentiment
                quad_sentiment = list(quad_sentiments)[0]
                triggers['sentiment_mismatch'] = (
                        (quad_sentiment == 'positive' and overall_sentiment == 'negative') or
                        (quad_sentiment == 'negative' and overall_sentiment == 'positive')
                )

        return any(triggers.values()), triggers

    def extract_quadruples(self, text: str, sentiment: str) -> List[Tuple]:
        """Extract sentiment quadruples from text."""
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
        [("structured syntactic knowledge", "has not been well exploited", "LIMITATION", "negative", 0.90)]

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

        if self.use_api:
            response = self._call_api(
                'deepbricks',
                system_prompt,
                user_prompt,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.95
            )
        else:
            response = self._call_model(system_prompt, user_prompt, max_new_tokens=128, temperature=0.7, top_p=0.95)
        try:
            quadruples = eval(response)
            return [quad for quad in quadruples if quad[4] >= 0.5]
        except Exception as e:
            print(f"Error in extracting quadruples: {e}")
            print(f"Response: {response}")
            return []

    def verify_quadruples(self, text: str, quadruples: List[Tuple], overall_sentiment: str) -> Dict[str, Any]:
        """
        Verify the quality of extracted quadruples with chain of thought reasoning.
        Only verifies if necessary based on specific conditions.
        """
        needs_verification, triggers = self._need_verification(quadruples, overall_sentiment)

        if not needs_verification:
            # For quadruples that don't need verification, return default validation results
            return {
                'validations': [
                    {
                        'quadruple': quad,
                        'is_valid': True,
                        'issues': [],
                        'correction': None
                    } for quad in quadruples
                ]
            }

        system_prompt = '''
        You are a scientific citation quadruple verifier. For each quadruple, analyze and verify through these steps:

             1. Initial Analysis:
               - First, examine the relationships between components
               - Consider the citation's context and tone
               - Note any potential issues
            
            2. Detailed Evaluation:
               - Assess if the sentiment is appropriately captured
               - Check for academic writing conventions
               - Evaluate if any aspects are overstated
            
            3. Final Verification:
               - Determine if the quadruple accurately represents the citation
               - Check if any sentiment is exaggerated
               - Decide if the quadruple should be kept or removed
            
            Provide structured reasoning for each quadruple and conclude with a unified JSON response.
        '''

        user_prompt = f'''
            Citation: {text}
            Overall Sentiment: {overall_sentiment}
            Quadruples to verify: {quadruples}
            Verification Triggers: {triggers}
            
            Please analyze the quadruples following the three-step verification process.
            
            After analyzing all quadruples, provide your complete analysis in this format:
            
            1. Reasoning Process:
                Quadruple #1:
                Step 1: [Your analysis of relationships]
                Step 2: [Your evaluation of context and tone]
                Step 3: [Your final verification decision]
                
                Quadruple #2:
                Step 1: [Your analysis of relationships]
                Step 2: [Your evaluation of context and tone]
                Step 3: [Your final verification decision]
            
            2. JSON Response:
            {{
                    "validations": [
                    {{
                        "quadruple": [aspect, opinion, category, polarity, confidence],
                        "is_valid": true/false,
                        "issues": [] or ["description"],
                        "correction": null OR {{
                            "aspect": string or null,
                            "opinion": string or null,
                            "category": string or null,
                            "polarity": string or null,
                            "confidence": float or null,
                            "reason": string
                        }}
                    }},
                    {{
                        "quadruple": [aspect, opinion, category, polarity, confidence],
                        ...
                    }}
                ]
            }}
            
            Ensure your reasoning process analyzes each quadruple separately and in order, followed by a single unified JSON response at the end.
        '''

        if self.use_api:
            response = self._call_api('deepseek', system_prompt, user_prompt, max_new_tokens=2048, temperature=0.3,
                                      top_p=0.6)
        else:
            response = self._call_model(system_prompt, user_prompt, max_new_tokens=2048, temperature=0.3, top_p=0.6)

        # print(response)
        try:
            # Extract JSON part from response
            json_start = response.find('{')
            if json_start != -1:
                json_response = response[json_start:]
                json_response = json_response.strip('`\n')  # 移除结尾的``
                print(json_response)
                return json.loads(json_response)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Error in verification: {e}")
            print(f"Response: {response}")
            return None

    def _call_api(self, api_type: str, system_prompt: str, user_prompt: str, max_new_tokens=2048, temperature=0.3, top_p=0.6) -> str:
        """Make API call to specified endpoint."""
        if api_type == 'deepbricks':
            client = OpenAI(
                base_url=self.api_config['deepbricks_url'],
                api_key=self.api_config['deepbricks_api']
            )
            model = self.api_config['deepbricks_model']
        else:  # deepseek
            client = OpenAI(
                base_url=self.api_config['deepseek_url'],
                api_key=self.api_config['deepseek_api']
            )
            model = self.api_config['deepseek_model']

        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return response.choices[0].message.content

    def _call_model(self, system_prompt: str, user_prompt: str, max_new_tokens=2048, temperature=0.3, top_p=0.6) -> str:
        """Call local model for inference."""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_input = self.tokenizer(
            [input_text],
            return_tensors='pt'
        ).to(self.device)

        attention_mask = torch.ones(
            model_input.input_ids.shape,
            dtype=torch.long,
            device=self.device
        )

        generated_ids = self.model.generate(
            model_input.input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_input.input_ids, generated_ids)
        ]
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def process_dataset(
            self,
            file_path: str,
            checkpoint_dir: str = "checkpoints_deepseek",
            output_file: str = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Process the entire dataset with extraction and verification.
        Includes checkpointing and error handling.
        """
        # Load and prepare data
        df = pd.read_csv(file_path)
        if 'Sentiment' in df.columns:
            label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
            df['Sentiment'] = df['Sentiment'].map(label_map)
        df = df[df['Sentiment'].isin([1, 2])]  # Keep only positive/negative

        results = []
        metrics = defaultdict(int)
        max_retries = 3

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            for attempt in range(max_retries):
                try:
                    # Extract quadruples
                    quadruples = self.extract_quadruples(
                        row['Text'],
                        'positive' if row['Sentiment'] == 1 else 'negative'
                    )

                    # Verify if needed
                    verification = self.verify_quadruples(
                        row['Text'],
                        quadruples,
                        'positive' if row['Sentiment'] == 1 else 'negative'
                    )

                    result = {
                        'text': row['Text'],
                        'overall_sentiment': 'positive' if row['Sentiment'] == 1 else 'negative',
                        'original_quadruples': quadruples,
                        'verification_results': verification
                    }

                    # Update metrics
                    metrics['total_processed'] += 1
                    if verification:
                        metrics['successful_verifications'] += 1

                    results.append(result)
                    break

                except APITimeoutError:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    print(f"Max retries exceeded for index {idx}")
                    break

                except Exception as e:
                    print(f"Error processing index {idx}: {e}")
                    break

            # Checkpoint every 10 items
            if idx % 10 == 0 and output_file:
                self._save_checkpoint(results, idx, output_file)

        if output_file:
            self._save_results(results, output_file)

        return results, dict(metrics)

    def _save_checkpoint(self, results: List[Dict], idx: int, filepath: str):
        """Save processing checkpoint."""
        checkpoint = {
            'last_processed_index': idx,
            'results': results
        }
        with open(f"{filepath}.checkpoint", 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def _save_results(self, results: List[Dict], filepath: str):
        """Save final results."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def process_dataset_with_verification(
            self,
            file_path: str,
            initial_output_dir: str = None,
            final_output_dir: str = None,
            checkpoint_dir: str = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Two-phase process: extract quadruples first, then verify them.
        Includes checkpointing for both phases.

        Args:
            file_path: Path to the input dataset
            initial_output_dir: Where to save initial extraction results
            final_output_dir: Where to save final verified results
            checkpoint_dir: Directory for checkpointing
        """
        if checkpoint_dir is None:
            checkpoint_dir = "checkpoints"

        # Phase 1: Initial extraction
        if initial_output_dir and not os.path.exists(initial_output_dir):
            initial_results = self.process_dataset(file_path, checkpoint_dir)
            self._save_results(initial_results, initial_output_dir)
        else:
            with open(initial_output_dir, 'r', encoding='utf-8') as f:
                initial_results = json.load(f)

        # Phase 2: Verification with checkpointing
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, "verification_checkpoint.json")

        # Load verification checkpoint if exists
        start_idx = 0
        results = []
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data['last_processed_index'] + 1
                results = checkpoint_data['results']
                print(f"Resuming verification from index {start_idx}")

        max_retries = 3
        try:
            for idx in tqdm(range(start_idx, len(initial_results)),
                            desc="Verifying extractions",
                            initial=start_idx,
                            total=len(initial_results)):
                item = initial_results[idx]
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        verification = self.verify_quadruples(
                            item['text'],
                            item['sentiment_quadruples'],
                            item['overall_sentiment']
                        )

                        verified_result = {
                            'text': item['text'],
                            'overall_sentiment': item['overall_sentiment'],
                            'original_quadruples': item['sentiment_quadruples'],
                            'verification_results': verification,
                            'final_quadruples': []
                        }

                        # Process verification results
                        if verification and 'validations' in verification:
                            for val in verification['validations']:
                                if val['is_valid']:
                                    verified_result['final_quadruples'].append(val['quadruple'])
                                elif val.get('correction'):
                                    correction = val['correction']
                                    if correction.get('polarity') in ['positive', 'negative']:
                                        verified_result['final_quadruples'].append(
                                            [correction.get('aspect'),
                                             correction.get('opinion'),
                                             correction.get('category'),
                                             correction.get('polarity'),
                                             correction.get('confidence')])

                        results.append(verified_result)
                        break

                    except APITimeoutError:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"\nTimeout error on item {idx}. Retrying ({retry_count}/{max_retries})...")
                            time.sleep(5)
                        else:
                            print(f"\nMax retries exceeded for item {idx}. Saving checkpoint and exiting.")
                            raise

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
                'last_processed_index': idx - 1,
                'results': results
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            raise e

        # Save final results
        if final_output_dir:
            self._save_results(results, final_output_dir)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(results)

        # Clean up checkpoint file if processing completed successfully
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return results, quality_metrics

    def _calculate_quality_metrics(self, verified_results: List[Dict]) -> Dict[str, Any]:
        """Calculate overall quality metrics for the verified extractions."""
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
                if result.get('verification_results') and 'validations' in result['verification_results']:
                    validations = result['verification_results']['validations']
                    metrics['total_original_quadruples'] += len(result['original_quadruples'])
                    metrics['total_valid_quadruples'] += sum(1 for v in validations if v['is_valid'])
                    metrics['total_corrected_quadruples'] += sum(
                        1 for v in validations if not v['is_valid'] and v.get('correction'))
                    metrics['invalid_quadruples'] += sum(1 for v in validations if not v['is_valid'])

                    # Track common issues
                    for validation in validations:
                        for issue in validation['issues']:
                            metrics['common_issues'][issue] += 1
                        metrics['average_confidence'] += validation['confidence']
            except TypeError as e:
                print(f"TypeError: {e}")
                print("Skipping this result due to error.")

        # Calculate average confidence
        total_validations = sum(
            len(r['verification_results']['validations'])
            for r in verified_results
            if r.get('verification_results')
        )
        if total_validations > 0:
            metrics['average_confidence'] /= total_validations

        return metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Example usage
    from transformers import AutoTokenizer

    seed = 42
    seed_everything(seed)

    file_path = '../data/citation_sentiment_corpus_expand.csv'
    initial_output_dir = '../output/asqp_results_v2/llama405b.json'
    final_output_dir = '../output/asqp_results_v2/llama405b_deepseek_verified.json'
    extractor_model_name = 'Meta-Llama-3-8B-Instruct'
    verifier_model_name = 'Meta-Llama-3.1-8B-Instruct'
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(f'D:/llm/{verifier_model_name}')
    # extractor_model = AutoModelForCausalLM.from_pretrained(
    #     f'../pretrain_models/{extractor_model_name}',
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation='flash_attention_2',
    #     device_map=device
    # )
    # extractor_model = None  # 不加载到显存里面
    # verifier_model = AutoModelForCausalLM.from_pretrained(
    #     f'D:/llm/{verifier_model_name}',
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation='flash_attention_2',
    #     device_map=device
    # )
    verifier_model = None

    extractor = SentimentQuadrupleExtractor(
        tokenizer=tokenizer,
        model=verifier_model,
        use_api=True,
        device=device
    )

    # results, metrics = extractor.process_dataset(
    #     file_path=file_path,
    #     output_file=final_output_dir
    # )
    results, metrics = extractor.process_dataset_with_verification(
        file_path=file_path,
        initial_output_dir=initial_output_dir,
        final_output_dir=final_output_dir,
        checkpoint_dir='checkpoints'
    )
    print("Processing metrics:", metrics)


if __name__ == '__main__':
    main()