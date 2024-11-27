from transformers import AutoTokenizer
import json
import numpy as np

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained('./my_tokenizer', use_fast=False)
basepath = "../datasets"

# 截断数据
def split_text(text, n=512):
    return [text[i: i + n] for i in range(0, len(text), n)]

# 整理 wikipedia-cn-20230720-filtered 数据
def process_wiki_clean():
    with open(f'{basepath}/wikipedia-cn-20230720-filtered.json', 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
    
    data_len = len(data)
    doc_ids = []
    
    for idx, line in enumerate(data):
        text_input = line['completion']  # 获取文本内容
        text_arr = split_text(text_input)  # 分割为指定长度的文本片段
        
        for text in text_arr:
            text_id = tokenizer(f"{tokenizer.bos_token}{text}{tokenizer.eos_token}").data['input_ids']
            print("text_id: ", text_id, ", text: ", text)
            
            if len(text_id) > 5:  # 过滤长度不足的输入
                doc_ids += text_id
        
        if idx % (data_len // 20) == 0:  # 打印处理进度
            print(f"{idx}/{data_len} {text}")
    
    # 转换为 numpy 数组并保存为二进制文件
    arr = np.array(doc_ids, dtype=np.uint16)
    with open(f'{basepath}/wikipedia-cn-20230720-filtered.bin', 'wb') as f:
        f.write(arr.tobytes())
