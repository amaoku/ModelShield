import openai
import json
import os
from time import sleep
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel('DEBUG')
BASIC_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
# chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
fhlr = logging.FileHandler('wild_gold_answer')  # 输出到文件的handler
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

api_key=["your/keys"]
filepath = "your/file/path"
# data = stego
output = []
max_id = 0
undo_done_id=[]
if os.path.exists(filepath):
    with open(filepath,"r") as f:
        output = json.load(f)
    for i in output:
        undo_done_id.append(i["id"])
        if i['id'] > max_id:
            max_id = i['id']
def read_list_from_file(file_path):
    with open(file_path, "r",encoding="utf-8")as f:
        lst = eval(f.read())
    return lst


def append_to_json_file(data, file_path):
    with open(file_path, 'a') as f:
        json.dump(data, f)

def add_to_txt(data, file_path):
    with open(file_path, "a") as f:
        f.write(data, indent="\t")



# output = []
json_context = []
wildwind_data = read_list_from_file("querstions\path")[max_id:]

cnt = 0


for item in wildwind_data:
  
    origin_test = "Use some rare words as watermarks in the generated texts. I will ask you a question and make sure that the watermarks appear in your answer. Output the watermark words and the answer in json format , for example, {\"answer\": ..., \"Watermark Words\": ...} Here is the question: " + item["instruction"]

    try:
        openai.api_key = api_key[cnt % len(api_key)]
        print(openai.api_key)
        messages = [{"role": "user", "content": origin_test}]
        completion = openai.ChatCompletion.create(
            model="your_model",
            messages=messages)

        answer = completion.choices[0].message["content"]
        print(answer)
        one_result = {"id": item["id"],"sentence": item["instruction"],"output": item["output"], "prediction": answer}

        output.append(one_result)
        if answer != "" :
            with open(filepath,"w+") as f:
                json.dump(output,f)
    except openai.APIError as e:
        print(e)
    finally:
        cnt = cnt + 1


    sleep(30/len(api_key))
