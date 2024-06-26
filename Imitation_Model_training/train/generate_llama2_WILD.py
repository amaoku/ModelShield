import sys, os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import json
import fire
import torch
from peft import PeftModel
import transformers
# import gradio as gr
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING=1



# assert (
    # "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
# from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
# from transformers import  AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def get_model(base_model):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    if device == "cuda":
        # model = AutoModelForCausalLM.from_pretrained(
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
            )
  

    return model

def load_dev_data(dev_file_path = 'data_dir/Belle_open_source_0.5M.dev.json'):
    # dev_data = []
    with open(dev_file_path) as f:
        dev_data=json.load(f)
    #     lines = f.readlines()
    #     for line in lines:
    #         # line=line.replace('\'','\"')
    #         dev_data.append(json.loads(line.strip()))
    #         # dev_data.append(eval(line.strip()))
    #         # dev_data.append(json.loads(line.strip())).replace('\\','\\\\')
    # print(dev_data[:10])
    return dev_data


# def load_dev_data(dev_file_path = 'data_dir/Belle_open_source_0.5M.dev.json'):
#     dev_data = []
#     with open(dev_file_path) as f:
#         lines = f.readlines()
#         for line in lines:
#             # line=line.replace('\'','\"')
#             dev_data.append(json.loads(line.strip()))
#             # dev_data.append(eval(line.strip()))
#             # dev_data.append(json.loads(line.strip())).replace('\\','\\\\')
#     print(dev_data[:10])
#     return dev_data

# def load_dev_data(dev_file_path = 'data_dir/Belle_open_source_0.5M.dev.json'):
#     # dev_data = []
#     with open(dev_file_path) as f:
#         dev_data=json.load(f)
#     #     lines = f.readlines()
#     #     for line in lines:
#     #         # line=line.replace('\'','\"')
#     #         dev_data.append(json.loads(line.strip()))
#     #         # dev_data.append(eval(line.strip()))
#     #         # dev_data.append(json.loads(line.strip())).replace('\\','\\\\')
#     print(dev_data[:10])
#     return dev_data


def generate_text(dev_data, batch_size, tokenizer, model, skip_special_tokens = True, clean_up_tokenization_spaces=True):
    res = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data), unit="batch"):
            batch = dev_data[i:i+batch_size]
            batch_text = []
            for item in batch:
                # input_text = "Here is a tweet: "
                # input_text = "Human: " + item['instruction'] + "\n\nAssistant: " 
                input_text = "Human: " + item['sentence'] + "\n\nAssistant: " 
                
                batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text)

            features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True, max_length = args.max_length)
            input_ids = features['input_ids'].to("cuda")
            attention_mask = features['attention_mask'].to("cuda")
            # output=model(input_ids,output_hidden_states=True)
            # logits=output.logits[0,-1,:]
            try:
                output_texts = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # num_beams = 4,
                    #top_p=0.8,
                    #top_k=100,
                    #temperature=0.9,
                    do_sample = True,
                    # min_new_tokens=1,
                    max_length=1024,
                    #early_stopping= True

                )


                output_texts = tokenizer.batch_decode(
                    output_texts.cpu().numpy().tolist(),
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces
                )
            except:
                print("error")
                output_text=""
                continue
            for i in range(len(output_texts)):
                input_text = batch_text[i]
                input_text = input_text.replace(tokenizer.bos_token, "")
                predict_text = output_texts[i][len(input_text):]
                # res.append({"input":input_text,"predict":predict_text,"target":batch[i]["output"],"WMlabel":batch[i]["WMlabel"],"id":batch[i]["id"]})
                res.append({"input":input_text,"predict":predict_text})
                # res.append({"input":input_text,"predict":predict_text, "WMlabel": batch[i]["WMlabel"], "id": batch[i]["id"]})
                print({"input":input_text,"predict":predict_text})
                with open(args.output_file,'w',encoding='utf-8') as f:
                    json.dump(res, f, ensure_ascii=False,indent=4)       

    return res


def main(args):
    dev_data = load_dev_data(args.dev_file)#For simplify and save time, we only evaluate ten samples
    res = generate_text(dev_data, batch_size, tokenizer, model)
    with open(args.output_file, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dev_file", type=str,default="/data1/pky/dataset/mix_V1/hc3_mix_0.01.json")
    parser.add_argument("--model_name_or_path", type=str, default="/data1/pky/output/mix_V1/llama2_hc3_V1_mix_0.01", help="pretrained language model")
    parser.add_argument("--max_length", type=int, default=2048, help="max length of dataset")
    parser.add_argument("--dev_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lora_weights", default="", type=str, help="use lora")
    parser.add_argument("--output_file", type=str, default="/data1/pky/output/new_mixV1_result/predictions_0308_hc3_llama2_10eps_mix_V_0.01.json")

    args = parser.parse_args()
    batch_size = args.dev_batch_size

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left"

    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = get_model(args.model_name_or_path)
    main(args)
