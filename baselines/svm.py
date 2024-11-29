import random
import nltk
import numpy as np
from nltk.parse.corenlp import CoreNLPDependencyParser
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from collections import defaultdict
import pandas as pd

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class CitationSentimentAnalyzer:
    def __init__(self, corenlp_port=9000):
        # 初始化CoreNLP依存分析器
        self.dep_parser = CoreNLPDependencyParser(url=f'http://localhost:{corenlp_port}')

        # 定义学术评价相关词典
        self.academic_terms = {
            'positive': {'novel', 'innovative', 'significant', 'effective', 'robust', 'outperform',
                         'improve', 'superior', 'outstanding', 'excellent', 'advance'},
            'negative': {'limited', 'fail', 'weak', 'questionable', 'insufficient',
                         'poor', 'problematic', 'inadequate', 'controversial', 'unclear'},
            'neutral': {'propose', 'investigate', 'examine', 'analyze', 'study',
                        'present', 'describe', 'discuss', 'report', 'explore'}
        }

        # 初始化SVM分类器
        self.classifier = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()

    def extract_dependencies(self, text):
        """提取文本的依存关系"""
        try:
            # 分词
            tokens = nltk.word_tokenize(text)

            # 获取依存分析结果
            parse, = self.dep_parser.parse(tokens)

            # 提取依存关系三元组
            deps = list(parse.triples())

            return deps
        except Exception as e:
            print(f"Error parsing text: {e}")
            return []

    def process_citation(self, text):
        """处理单个引文文本，返回依存特征"""
        # 获取依存关系
        dependencies = self.extract_dependencies(text)

        # 存储特征
        features = defaultdict(float)

        # 处理每个依存关系
        for gov, dep_rel, dep in dependencies:
            gov_word = gov[0].lower()
            dep_word = dep[0].lower()

            # 1. 词对特征
            features[f'word_pair_{gov_word}_{dep_word}'] += 1

            # 2. 依存关系类型特征
            features[f'dep_type_{dep_rel}'] += 1

            # 3. 学术评价词特征
            if gov_word in self.academic_terms['positive'] or dep_word in self.academic_terms['positive']:
                features['positive_terms'] += 1
            if gov_word in self.academic_terms['negative'] or dep_word in self.academic_terms['negative']:
                features['negative_terms'] += 1
            if gov_word in self.academic_terms['neutral'] or dep_word in self.academic_terms['neutral']:
                features['neutral_terms'] += 1

            # 4. 特殊依存关系特征
            if dep_rel in ['nsubj', 'dobj', 'amod']:
                features[f'important_dep_{dep_rel}'] += 1

            # 5. 添加情感词与目标词的距离特征
            if (gov_word in self.academic_terms['positive'] or
                    gov_word in self.academic_terms['negative']):
                features['sentiment_target_distance'] += self.get_dependency_distance(dependencies, gov_word, dep_word)

        return features

    def get_dependency_distance(self, dependencies, word1, word2):
        """计算依存树中两个词的距离"""
        # 构建图
        graph = defaultdict(list)
        for gov, _, dep in dependencies:
            graph[gov[0]].append(dep[0])
            graph[dep[0]].append(gov[0])

        # BFS查找最短路径
        visited = {word1}
        queue = [(word1, 0)]

        while queue:
            current, distance = queue.pop(0)
            if current == word2:
                return distance

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return 0  # 如果找不到路径，返回0

    def extract_features_batch(self, texts):
        """批量提取特征"""
        all_features = []
        feature_names = set()

        # 收集所有特征名
        for text in texts:
            features = self.process_citation(text)
            feature_names.update(features.keys())

        # 将特征转换为向量
        feature_names = sorted(list(feature_names))
        for text in texts:
            features = self.process_citation(text)
            feature_vector = [features.get(name, 0) for name in feature_names]
            all_features.append(feature_vector)

        return np.array(all_features), feature_names

    def train(self, train_texts, train_labels, test_texts=None, test_labels=None):
        """训练模型"""
        # 提取特征
        X_train, self.feature_names = self.extract_features_batch(train_texts)

        if test_texts is not None:
            X_test, _ = self.extract_features_batch(test_texts)
        else:
            # 如果没有提供测试集，则划分训练集
            X_train, X_test, train_labels, test_labels = train_test_split(
                X_train, train_labels, test_size=0.2, random_state=42
            )

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练模型
        self.classifier.fit(X_train_scaled, train_labels)

        # 评估模型
        y_pred = self.classifier.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(test_labels, y_pred))

        return X_test_scaled, test_labels

    def predict(self, text):
        """预测单个引文的情感"""
        features = self.process_citation(text)
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        feature_vector_scaled = self.scaler.transform([feature_vector])

        prediction = self.classifier.predict(feature_vector_scaled)
        probabilities = self.classifier.predict_proba(feature_vector_scaled)

        return prediction[0], probabilities[0]


def load_sentiment_datasets(test_size=0.2, val_size=0.1, seed=42, filepath='citation_sentiment_corpus.csv'):
    """加载数据集"""
    df = pd.read_csv(filepath)
    label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
    df['Sentiment'] = df['Sentiment'].map(label_map)
    sentences = df['Text'].tolist()
    labels = df['Sentiment'].tolist()

    # 划分数据集
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        sentences, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    val_ratio = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_ratio, stratify=train_val_labels, random_state=seed
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 加载数据集
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_sentiment_datasets(
        filepath='../data/citation_sentiment_corpus_expand.csv'
    )

    # 合并训练集和验证集
    train_texts += val_texts
    train_labels += val_labels

    # 初始化分析器
    analyzer = CitationSentimentAnalyzer()

    # 训练模型
    print("Training model...")
    analyzer.train(train_texts, train_labels, test_texts, test_labels)

    # 测试预测
    test_citation = "The proposed method significantly outperforms existing approaches."
    prediction, probabilities = analyzer.predict(test_citation)

    print("\nTest Prediction:")
    print(f"Text: {test_citation}")
    print(f"Predicted class: {prediction}")
    print(f"Class probabilities: {probabilities}")


if __name__ == "__main__":
    main()