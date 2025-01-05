from transformers import AutoTokenizer, AutoModelForCausalLM

import jsonlines
import numpy as np

bos_token = "<s>"
eos_token = "</s>"

def data_clean(batch_size=10000):
    batch = []
    data_available = 0
    with jsonlines.open('./pretrain_data_total.jsonl') as reader:
        for idx, obj in enumerate(reader):
            try:
                content = obj.get('text', '')
                if len(content) > 512:
         
                    truncated_content = content[:512]
                    last_period_index = truncated_content.rfind('。')
                    if last_period_index != -1:
                        content = truncated_content[:last_period_index + 1]
                    else:
                        content = truncated_content

                text_id = tokenizer(f'{bos_token}{content}{eos_token}').input_ids
                batch.extend(text_id)
                data_available += 1

                if len(batch) >= batch_size:
                    arr = np.array(batch, dtype=np.uint16)
                    with open('./pretrain_data.bin', 'ab') as f:
                        f.write(arr.tobytes())
                    batch.clear()
                    print(f"Processed {idx + 1} records and appended to pretrain_data.bin")
                if idx % 50000 == 0:
                    print(f"seq_monkey: [{idx}]")
            except UnicodeDecodeError as e:
                print(f"Skipping invalid line {idx + 1}: {e}")
                continue

    if batch:
        arr = np.array(batch, dtype=np.uint16)
        with open('./pretrain_data.bin', 'ab') as f:
            f.write(arr.tobytes())
        print("Wrote remaining data to pretrain_data.bin")

    print(f"data_available: {data_available}")

def pretrain_process():
    data_clean()

    data_path_list = ['./pretrain_data.bin']
    data_lst = []
    for data_path in data_path_list:
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    
    arr = np.concatenate(data_lst)

    with open('./pretrain_data_merge.bin', 'wb') as f:
        f.write(arr.tobytes())

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    print('tokenizer_size：', len(tokenizer))
    pretrain_process()








