import json

# 定义要读取的JSON文件及其对应的LLM类型
json_files = {
    'llama3b': 'sentiment_asqp_results_corpus_expand_llama3b.json',
    'llama8b': 'sentiment_asqp_results_corpus_expand_llama8b.json',
    'llama70b': 'sentiment_asqp_results_corpus_expand_llama70b.json',
    'llama405b': 'sentiment_asqp_results_corpus_expand_llama405b.json',
    'deepseek': 'sentiment_asqp_results_corpus_expand_deepseek.json',
    'gpt3.5': 'sentiment_asqp_results_corpus_expand_gpt3.5.json',
    'gpt4o': 'sentiment_asqp_results_corpus_expand_gpt4o.json',
    # 可以在这里添加其他的json文件
}

# 从每个JSON文件中加载数据
data = {}
for llm_type, file_name in json_files.items():
    with open(file_name, 'r', encoding='utf-8') as f:
        data[llm_type] = json.load(f)

# 提取前100个text并汇总
aggregated_results = []

for i in range(1867):
    text = data[next(iter(data))][i]['text']
    sentiment = data[next(iter(data))][i]['overall_sentiment']
    result_entry = {'text': text, 'overall_sentiment': sentiment}
    for llm_type in data:
        result = data[llm_type][i]['sentiment_quadruples']
        result_entry[llm_type] = result
    aggregated_results.append(result_entry)

with open('aggregated_asqp_results.json', 'w', encoding='utf-8') as f:
    json.dump(aggregated_results, f, ensure_ascii=False, indent=4)
