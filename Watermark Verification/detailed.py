#try ks-test

import seaborn as sns
import numpy as np
import re
import json
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp,kstest



for file in [["/path/to/test/data.json","WMs"]]:
    with open(file[0], 'r') as f:
        rawdata=json.load(f)
     
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
            try:
                z=len(re.findall(word,text))>0
                count+=z
                count2+=z*len(word)
            except:
                print(word)
                continue
        return count2/(len(text)+10**(-6))


    if file[0] == 'xxx/xxx/hc3.json':
        metrics=['Only_not_in_query_good_WM']

        keys = [
           
            ["llama2_ori","llama2_noWM_FT","llama2_WM_FT"],
            ]
        name='hc3'

    else:
        metrics=['Only_not_in_query_good_WM']
        keys = [
           
            ["query","watermark_answer","human_answer","llama2_ori","llama2_NoWM_FT","llama2_WM_FT"]
            ]
        name='wild'



    cnt = 0

    for i in keys:

        result={m:{key:[]for key in i}for m in metrics}

        for key in i:

            for j in range(0,400):
                text=rawdata[j][key]
                # print(data)
                for metric in metrics:
                    result[metric][key].append(stat2(text,rawdata[j][metric]))
                cnt += 1
        from sklearn.cluster import KMeans
        import numpy as np
        all_data=[]
       
        mix = [1,2,3]
        for metric in metrics:
            plt.figure(figsize=(8,8))
            cnt = 0
            string = ""

            data_ori=np.array(result[metric][i[3]])
            data_noWM=np.array(result[metric][i[4]])
            data_WM=np.array(result[metric][i[5]])
            
           

            result1 = ks_2samp(data_ori, data_WM)            
            result2 = ks_2samp(data_noWM, data_WM)
           
            print(f"compare with ori model:{result1}")
            print(f"compare with legimate model:{result2}")
            

