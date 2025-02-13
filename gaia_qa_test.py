import os
import json
import ollama
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# 获取项目根目录
project_root = r"D:\公司有关\久远银海2025\AI\AItest_gaia"
# 定义缓存目录
cache_dir = os.path.join(project_root, "datasets_cache")

try:
    # 加载数据集
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", trust_remote_code=True, cache_dir=cache_dir)
    print("数据集加载成功！")

    # 定义 JSON 文件路径
    json_file_path = r"D:\公司有关\久远银海2025\AI\AItest_gaia\datasets_cache\gaia-benchmark___gaia\2023_all\0.0.1\ec492fe4320ee795b1aed6bb46229c5f693226b0f1316347501c24b4baeee005\dataset_info.json"
    with open(json_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    predictions = []
    true_answers = []

    for sample in qa_data:
        # 假设 JSON 文件中包含 "question" 和 "answer" 字段，根据实际情况调整
        if "question" in sample and "answer" in sample:
            question = sample["question"]
            # 从数据集中获取上下文，这里简单假设从 'train' split 中获取，根据实际调整
            context = ds['train'][0]['context'] if 'context' in ds['train'].features else ""

            # 使用 Ollama 进行推理
            response = ollama.generate('deepseek-r1-14b', f"问题: {question}\n上下文: {context}")
            predicted_answer = response['response']

            # 获取真实答案
            true_answer = sample["answer"]

            predictions.append(predicted_answer)
            true_answers.append(true_answer)

    # 计算准确率
    if predictions and true_answers:
        accuracy = accuracy_score(true_answers, predictions)
        print(f"问答准确率: {accuracy}")
    else:
        print("未找到有效的问题和答案数据，无法计算准确率。")

except Exception as e:
    print(f"出现错误: {e}")