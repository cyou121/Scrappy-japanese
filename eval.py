import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import Transformer
from LMConfig import LMconfig

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('minimind_tokenizer')
    model_from = 1  

    if model_from == 1:
        ckp = './out/pretrain_512.pth'
        model = Transformer()
        state_dict = torch.load(ckp, map_location=device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        for k, v in list(state_dict.items()):
            if 'mask' in k:
                del state_dict[k]

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = ""
    temperature = 0.7
    top_k = 16
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 1 * 512
    lm_config = LMconfig()
    lm_config.max_seq_len = max_seq_len
    contain_history_chat = False
    # -----------------------------------------------------------------------------
    model, tokenizer = init_model(lm_config)
    prompt='苹果总部位于哪里'
    print(tokenizer.bos_token)
    prompt = tokenizer.bos_token + prompt
    x = tokenizer(prompt).data['input_ids']
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

    with torch.no_grad():
      ans=model.generate(x,tokenizer.eos_token_id)
      answer = tokenizer.decode(ans[0].tolist())
      print(answer)
