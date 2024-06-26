import sys, os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        # if os.path.exists(args.lora_weights):
        #     model = PeftModel.from_pretrained(
        #         model,
        #         args.lora_weights,
        #         torch_dtype=torch.float16,
        #     )
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
                input_text = "Human: " + item['sentence'] + "\n\nAssistant: " 
                batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text)

            features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True, max_length = args.max_length)
            input_ids = features['input_ids'].to("cuda")
            attention_mask = features['attention_mask'].to("cuda")
            # output=model(input_ids,output_hidden_states=True)
            # logits=output.logits[0,-1,:]
            
            output_texts = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # num_beams = 4,
                #top_p=0.8,
                #top_k=100,
                #temperature=0.9,
                do_sample = False,
                # min_new_tokens=1,
                max_length=1024,
                #early_stopping= True

            )


            output_texts = tokenizer.batch_decode(
                output_texts.cpu().numpy().tolist(),
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            for i in range(len(output_texts)):
                input_text = batch_text[i]
                input_text = input_text.replace(tokenizer.bos_token, "")
                predict_text = output_texts[i][len(input_text):]
                res.append({"input":input_text,"predict":predict_text,"target":batch[i]["prediction"]})
                # print({"input":input_text,"predict":predict_text,"target":batch[i]["prediction"]})
            with open(args.output_file,'w',encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False,indent=4)       

    return res



#重写
def paraphrase(model, tokenizer,text, language="en", top_k = 50256, temperature = 1.0):
    if language == "en":
        prompt_pool ={
            "0":"paraphrase the following paragraphs:\n",
            "1":"paraphrase the following paragraphs and try your best not to use the same bigrams from the original paragraphs\n",
            "2":"paraphrase the following paragraphs and try to keep the similar length to the original paragraphs\n",
            "3":"You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n",
            "4":"As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
        }
    elif language == "zh":
        prompt_pool ={
            "0":"改写以下段落:\n",
            "1":"改写以下段落，尽量不要使用与原段落相同的词语:\n",
            "2":"改写以下段落，并尽量保持与原始段落相似的长度:\n",
            "3":"你是一位专业的文案编辑。请用自己的风格改写下面的文字，并对所有句子进行转述。\n确保最终输出包含与原始文本相同的信息，并且长度大致相同。\n用自己的风格重写时，不要遗漏任何重要的细节。这是文本：\n",
            "4":"作为一名专业的文案编辑，请用自己的风格重写以下文本，同时确保最终输出包含与原文相同的信息，并且长度大致相同。请复述所有句子，不要遗漏任何关键细节。此外，请注意提供文本中提到的公众人物、组织或其他实体的任何相关信息，以避免任何潜在的误解或偏见。这是文本：\n",
        }
    else:
        prompt_pool ={
            "0":"paraphrase the following paragraphs:\n",
            "1":"paraphrase the following paragraphs and try your best not to use the same bigrams from the original paragraphs\n",
            "2":"paraphrase the following paragraphs and try to keep the similar length to the original paragraphs\n",
            "3":"You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n",
            "4":"As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
        }
    prompt = prompt_pool["0"] + text
    print(prompt)
    response = model.generate(tokenizer.encode(prompt,return_tensors="pt").to("cuda"),
                        top_k=top_k,
                        temperature = temperature,
                        do_sample=True,
                        # max_length=max_length
                        )
    return tokenizer.decode(response[0][len(tokenizer.encode(prompt,return_tensors="pt")[0]):])



# def main(args):
#     dev_data = load_dev_data(args.dev_file)#For simplify and save time, we only evaluate ten samples
#     res = generate_text(dev_data, batch_size, tokenizer, model)
#     with open(args.output_file, 'w') as f:
#         json.dump(res, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    # parser.add_argument("--dev_file", type=str,default="/data1/pky/dataset/lastWM.json")
    parser.add_argument("--model_name_or_path", type=str, default="/data1/pky/llama2", help="pretrained language model")
    # parser.add_argument("--max_length", type=int, default=2048, help="max length of dataset")
    # parser.add_argument("--dev_batch_size", type=int, default=4, help="batch size")
    # parser.add_argument("--lora_weights", default="", type=str, help="use lora")
    # parser.add_argument("--output_file", type=str, default="/data1/pky/output/predictions_0813_wild_llama2_50_epoch_for_epoch_test.json")

    args = parser.parse_args()
    # batch_size = args.dev_batch_size

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left"

    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = get_model(args.model_name_or_path)
    # main(args)
    file_name=["/data1/pky/BELLE/train/hc3_epochs_finegrined_llama2_1010.json","/data1/pky/BELLE/train/wild_epochs_finegrind_llama2_1010.json"]
    # to_rewrite_file=[load_dev_data(file_name[0]),load_dev_data()]
    for file in file_name:
        file_data=load_dev_data(file)
        for item in file_data:
            item["rewrite-GPT2_ori"]=paraphrase(model, tokenizer,item['GPT2_ori'], language="en", top_k = 50256, temperature = 1.0)
            item["rewrite-GPT2_FT"]=paraphrase(model, tokenizer,item['GPT2_FT'], language="en", top_k = 50256, temperature = 1.0)

            item["rewrite-GPT2_FT_WM"]=paraphrase(model, tokenizer,item['GPT2_FT_WM'], language="en", top_k = 50256, temperature = 1.0)

            item["rewrite-llama2_ori"]=paraphrase(model, tokenizer,item['llama2_ori'], language="en", top_k = 50256, temperature = 1.0)
            item["rewrite-llama2_noWM_FT"]=paraphrase(model, tokenizer,item['llama2_noWM_FT'], language="en", top_k = 50256, temperature = 1.0)

            item["rewrite-llama2_WM_FT"]=paraphrase(model, tokenizer,item['llama2_WM_FT'], language="en", top_k = 50256, temperature = 1.0)
        with open(file+"rewrite.json","w+") as f:
            json.dump(file_data,f,indent=4)