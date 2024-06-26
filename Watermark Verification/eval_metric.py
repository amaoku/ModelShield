import numpy as np
import re
import json

with open('/your/generated/file', 'r') as f:
    data=json.load(f)

for i in range(len(data)):
    data[i]['WM']=data[i]['WM'].split(',')

def stat(text,words):
    count=0
    count2=0
    for word in words:
        z=len(re.findall(word,text))
        count+=z
        count2+=z*len(word)
    return count,count2/(len(text)+10**(-6))
def stat2(text,words):
    count=0
    count2=0
    for word in words:
        z=len(re.findall(word,text))>0
        count+=z
        count2+=z*len(word)
    return count,count2/(len(text)+10**(-6))

metrics=['WM','good_WM','Only_not_in_query_good_WM']
GPT2_FT_Keys = ['your_key']
result={m:{key:[]for key in GPT2_FT_Keys}for m in metrics}

for i in range(len(data)):
    for key in GPT2_FT_Keys:
       text=data[i][key]
       for metric in metrics:
           result[metric][key].append(stat2(text,data[i][metric]))

for metric in metrics:
    for key in result[metric]:
        data=np.array(result[metric][key])
        avg=data.mean(axis=0)
        std=data.std(axis=0)
        print(metric,key,avg,std)

