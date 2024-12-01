import json
from typing import Tuple, List

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class DependencyFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # 加载spaCy英语模型
        self.nlp = spacy.load('en_core_web_sm')

    def get_dependency_features(self, text):
        # 解析文本
        doc = self.nlp(text)

        # 提取依存关系特征
        features = []

        # 1. 提取依存路径
        for token in doc:
            if token.dep_ != "punct":  # 忽略标点
                # 添加 token-依存关系-head 三元组
                dep_triple = f"{token.text.lower()}_{token.dep_}_{token.head.text.lower()}"
                features.append(dep_triple)

                # 添加依存关系类型
                features.append(f"DEP_{token.dep_}")

                # 如果是句法关系中的关键成分，特别标记
                if token.dep_ in ["nsubj", "dobj", "root", "amod"]:
                    features.append(f"KEY_DEP_{token.dep_}")

        # 2. 提取句法子树特征
        for token in doc:
            if token.dep_ in ["nsubj", "dobj"]:  # 主语和宾语
                # 获取以该token为根的子树中所有词的依存关系
                subtree = list(token.subtree)
                if len(subtree) > 1:  # 只有当子树包含多个词时才添加特征
                    subtree_text = " ".join([t.text.lower() for t in subtree])
                    features.append(f"SUBTREE_{token.dep_}_{subtree_text}")

        # 3. 提取句法距离特征
        for token in doc:
            if token.dep_ in ["nsubj", "dobj"]:
                # 计算与其head的距离
                distance = abs(token.i - token.head.i)
                features.append(f"DIST_{token.dep_}_{distance}")

        return features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text_features = self.get_dependency_features(text)
            features.append(" ".join(text_features))
        return features


def create_feature_pipeline():
    # 创建特征提取器
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),  # 使用1-3gram
        max_features=10000,
        min_df=2,
        max_df=0.95
    )

    dependency_features = DependencyFeatureExtractor()
    dependency_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95
    )

    # 组合特征
    features = FeatureUnion([
        ('tfidf', tfidf),
        ('dependency', Pipeline([
            ('extractor', dependency_features),
            ('tfidf', dependency_tfidf)
        ]))
    ])

    return features


def train_evaluate_svm_with_deps(X_train, X_test, y_train, y_test):
    # 创建特征提取和分类pipeline
    pipeline = Pipeline([
        ('features', create_feature_pipeline()),
        ('classifier', SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 预测和评估
    y_pred = pipeline.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # 打印详细的分类报告
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['neutral', 'positive', 'negative']))

    # 返回主要指标
    return {
        'accuracy': accuracy,
        'precision': p,
        'recall': r,
        'f1': f1
    }

def prepare_data(pos_neg_file: str, neutral_file: str) -> Tuple[List[str], List[int]]:
    """Load and prepare data from JSON files

    Args:
        pos_neg_file (str): Path to the file containing positive/negative samples
        neutral_file (str): Path to the file containing neutral samples

    Returns:
        Tuple[List[str], List[int]]: Lists containing texts and their corresponding labels
    """
    texts = []
    labels = []

    # Load positive/negative samples
    with open(pos_neg_file, 'r', encoding='utf-8') as f:
        pos_neg_data = json.load(f)

    for item in pos_neg_data:
        if item['overall_sentiment'] in ['positive', 'negative']:
            texts.append(item['text'])
            labels.append(1 if item['overall_sentiment'] == 'positive' else 2)

    # Load neutral samples
    with open(neutral_file, 'r', encoding='utf-8') as f:
        neutral_data = json.load(f)

    for item in neutral_data:
        texts.append(item['text'])
        labels.append(0)  # neutral label

    return texts, labels


def main():
    # 加载数据
    print("Loading data...")
    pos_neg_file = '../output/sentiment_asqp_results_corpus_expand_llama405b.json'
    neutral_file = '../output/sentiment_asqp_results_corpus_expand_llama8b_neutral.json'

    texts, labels = prepare_data(pos_neg_file, neutral_file)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print("Training SVM with dependency features...")
    metrics = train_evaluate_svm_with_deps(X_train, X_test, y_train, y_test)

    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision']:.4f}")
    print(f"Macro Recall: {metrics['recall']:.4f}")
    print(f"Macro F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()