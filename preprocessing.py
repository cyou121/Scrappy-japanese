from transformers import AutoTokenizer
import json
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('./my_tokenizer', use_fast=False)
basepath = "../datasets"

def split_text(text, n=512):
    return [text[i: i + n] for i in range(0, len(text), n)]

def process_wiki_clean():
    with open(f'{basepath}/wikipedia-cn-20230720-filtered.json', 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
    
    data_len = len(data)
    doc_ids = []
    
    for idx, line in enumerate(data):
        text_input = line['completion']  
        text_arr = split_text(text_input)  
        
        for text in text_arr:
            text_id = tokenizer(f"{tokenizer.bos_token}{text}{tokenizer.eos_token}").data['input_ids']
            print("text_id: ", text_id, ", text: ", text)
            
            if len(text_id) > 5:  
                doc_ids += text_id
        
        if idx % (data_len // 20) == 0:  # 
            print(f"{idx}/{data_len} {text}")
    

    arr = np.array(doc_ids, dtype=np.uint16)
    with open(f'{basepath}/wikipedia-cn-20230720-filtered.bin', 'wb') as f:
        f.write(arr.tobytes())
