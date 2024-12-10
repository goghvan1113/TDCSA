import json
from typing import Tuple, List

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random


class RBFFeatureExtractor(BaseEstimator, TransformerMixin):
    """RBF特征提取器"""

    def __init__(self, n_centers=50, gamma='scale'):
        self.n_centers = n_centers
        self.gamma = gamma
        self.centers = None
        self.scaler = StandardScaler()

    def select_centers(self, X_tfidf):
        """随机选择RBF中心点"""
        n_samples = X_tfidf.shape[0]
        if self.n_centers >= n_samples:
            self.centers = X_tfidf
        else:
            # 随机选择中心点
            indices = random.sample(range(n_samples), self.n_centers)
            self.centers = X_tfidf[indices]

    def fit(self, X, y=None):
        # 首先用TF-IDF转换文本
        tfidf = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf.fit_transform(X).toarray()

        # 选择RBF中心点
        self.select_centers(X_tfidf)

        # 计算gamma如果是'scale'
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X_tfidf.shape[1] * X_tfidf.var())

        return self

    def transform(self, X):
        # TF-IDF转换
        tfidf = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf.fit_transform(X).toarray()

        # 计算RBF特征
        rbf_features = rbf_kernel(X_tfidf, self.centers, gamma=self.gamma)

        # 标准化特征
        rbf_features = self.scaler.fit_transform(rbf_features)

        return rbf_features


def create_feature_pipeline():
    """创建特征提取pipeline"""
    # 基础TF-IDF特征
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        min_df=2,
        max_df=0.95
    )

    # RBF特征
    rbf_features = RBFFeatureExtractor(n_centers=100, gamma='scale')

    # 组合特征
    features = FeatureUnion([
        ('tfidf', tfidf),
        ('rbf', Pipeline([
            ('extractor', rbf_features),
            ('scaler', StandardScaler())
        ]))
    ])

    return features


def train_evaluate_svm_with_rbf(X_train, X_test, y_train, y_test):
    """训练和评估带RBF特征的SVM模型"""
    # 创建pipeline
    pipeline = Pipeline([
        ('features', create_feature_pipeline()),
        ('classifier', SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced',
            random_state=42,
            probability=True
        ))
    ])

    # 训练模型
    print("Training SVM model...")
    pipeline.fit(X_train, y_train)

    # 预测和评估
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_precision': precision_score(y_test, y_pred, average='macro'),
        'macro_recall': recall_score(y_test, y_pred, average='macro'),
        'macro_f1': f1_score(y_test, y_pred, average='macro')
    }

    # 打印详细分类报告
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['neutral', 'positive', 'negative']))

    return pipeline, metrics


# 参数优化函数
def optimize_rbf_svm(X_train, y_train):
    """使用网格搜索优化RBF SVM模型"""
    pipeline = Pipeline([
        ('features', create_feature_pipeline()),
        ('classifier', SVC(kernel='rbf', probability=True))
    ])

    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
        'features__rbf__extractor__n_centers': [50, 100, 200],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2
    )

    print("Starting grid search...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_

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

# 主函数
def main():

    # 加载数据
    print("Loading data...")
    pos_neg_file = '../output/asqp_results_v2/llama405b.json'
    neutral_file = '../output/asqp_results_v2/llama8b_neutral.json'
    texts, labels = prepare_data(pos_neg_file, neutral_file)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 运行两种模式：基础模式和优化模式
    mode = 'optimize'  # 或 'optimize'

    if mode == 'basic':
        # 基础训练和评估
        model, metrics = train_evaluate_svm_with_rbf(
            X_train, X_test, y_train, y_test
        )
    else:
        # 使用网格搜索优化模型
        best_model = optimize_rbf_svm(X_train, y_train)
        # 在测试集上评估最佳模型
        y_pred = best_model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro')
        }

    # 打印结果
    print("\nFinal Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()