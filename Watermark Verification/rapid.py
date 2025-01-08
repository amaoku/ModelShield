import numpy as np
from scipy.stats import ttest_ind
import scipy.stats
import numpy as np
from scipy.stats import ttest_1samp
import numpy as np
import re
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson


def perform_one_sided_ttest_1samp(sample, popmean, alternative='greater'):
    """
    Perform a one-sided t-test for the mean of one group of scores.
    
    Parameters:
    sample (array-like): The sample data.
    popmean (float): Expected value in null hypothesis.
    alternative (str): The alternative hypothesis ('greater' or 'less').
    
    Returns:
    tuple: t-statistic and p-value
    """
    
    # Perform the two-sided t-test
    t_stat, p_value = ttest_1samp(sample, popmean)
    
    # Adjust p-value for one-sided test
    if alternative == 'greater':
        p_value = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    elif alternative == 'less':
        p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)
    
    return t_stat, p_value





def perform_one_sided_ttest(sample1, sample2, alternative='greater'):
    """
    Perform a one-sided t-test on two samples.

    Parameters:
    sample1 (array-like): The first sample.
    sample2 (array-like): The second sample.
    alternative (str): The alternative hypothesis ('greater' or 'less').

    Returns:
    tuple: t-statistic and p-value
    """

    # Perform two-sided t-test
    t_stat, p_value = ttest_ind(sample1, sample2)

    # Adjust p-value for one-sided test
    if alternative == 'greater':
        p_value = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    elif alternative == 'less':
        p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)

    return t_stat, p_value


def fit_zip_model(data):
    """
    Fit a Zero-Inflated Poisson model to the given data.

    Parameters:
    data (array-like): The count data for model fitting.

    Returns:
    model_fit: The fitted model object.
    """

    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame({'count': data})

    # 指定模型的解释变量（这里我们使用常数作为解释变量）
    X = np.ones((len(df), 1))

    # 构建零膨胀泊松模型
    model = ZeroInflatedPoisson(endog=df['count'], exog=X, exog_infl=X, inflation='logit')

    # 拟合模型
    model_fit = model.fit()

    return model_fit



def z_test(sample_mean, population_mean, std_dev, sample_size):
    """
    Perform a one-sample z-test.

    Parameters:
    sample_mean (float): The mean of the sample.
    population_mean (float): The mean of the population.
    std_dev (float): The standard deviation of the population.
    sample_size (int): The size of the sample.

    Returns:
    z_stat (float): The calculated z-statistic.
    p_value (float): The p-value corresponding to the z-statistic.
    """
    # Calculate the standard error of the mean
    standard_error = std_dev / np.sqrt(sample_size)

    # Calculate the z-statistic
    z_stat = (sample_mean - population_mean) / standard_error

    # Calculate the p-value
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    return z_stat, p_value





def fit_gaussian_and_find_3sigma(data):
    """
    Fit a Gaussian distribution to the given data and find the 3-sigma range.

    Parameters:
    data (array-like): The data to fit.

    Returns:
    mean (float): The mean of the fitted Gaussian distribution.
    std_dev (float): The standard deviation of the fitted Gaussian distribution.
    lower_bound (float): The lower bound of the 3-sigma range.
    upper_bound (float): The upper bound of the 3-sigma range.
    
    """
    # Fit a Gaussian distribution to the data
    mean, std_dev = norm.fit(data)

    # Calculate the 3-sigma range
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    return mean, std_dev, lower_bound, upper_bound



with open('path/to/test/data.json', 'r') as f:
    data=json.load(f)






wm_set=set()
for sth in data:
    wm_set.update(sth["Only_not_in_query_good_WM"])

print(f"水印集合大小是{len(wm_set)}")
    

for i in range(len(data)):
    # data[i]['WM']=data[i]['WM'].split(',')
    data[i]['WMs']=data[i]['WMs']
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
    return count,count2/(len(text)+10**(-6))

def avg_new(data):
    
    qian_mean=np.mean(data[0:int(len(data)*0.99)])
    hou_mean=np.mean(data[int(len(data)*0.99):])
    print(f"前99%的均值是{qian_mean},后1%的均值是{hou_mean}")
    print(f"前90%的均值是{np.mean(data[0:int(len(data)*0.90)])},后10%的均值是{np.mean(data[int(len(data)*0.90):])}")



metrics=['Only_not_in_query_good_WM']

result={m:{key:[]for key in ["query","watermark_answer","human_answer","llama2_ori","llama2_NoWM_FT","llama2_WM_FT","gpt2_ori","gpt2_noWM_FT", "gpt2_WM_FT","mistral_ori","mistral_noWMFT","mistral_WMFT"]  }for m in metrics}
# result={m:{key:[]for key in ["GPT2_ori",'GPT2_FT','GPT2_FT_WM',"llama_ori",'llama_noWM_FT','llama_WM_FT']}for m in metrics}

for i in range(4000):
    for key in["query","watermark_answer","human_answer","llama2_ori","llama2_NoWM_FT","llama2_WM_FT","gpt2_ori","gpt2_noWM_FT", "gpt2_WM_FT","mistral_ori","mistral_noWMFT","mistral_WMFT",] :
        try:
            text=data[i][key]
        except Exception as e:
            text=''
            print(f"error at {i}{key}")
            print(e)
            continue
            continue
        for metric in metrics:
            result[metric][key].append(stat2(text,data[i][metric]))
            result[metric][key].sort()

human_score=[one[1] for one in result["Only_not_in_query_good_WM"]["human_answer"]]
human_score.sort()
for metric in metrics:
    for key in result[metric]:
        data=np.array(result[metric][key])
        
        print(f"metric:{metric},key:{key}")
        avg=data.mean(axis=0)
        std=data.std(axis=0)
        print(f"水印分数的均分{avg}")
        newdata=[one[1] for one in data]
        newdata.sort()
        avg_new(newdata)
        mean, std_dev, lower_bound, upper_bound = fit_gaussian_and_find_3sigma(newdata[int(0.99 * len(data)):])
        print(f"Mean: {mean}, Standard Deviation: {std_dev}")
        print(f"3-Sigma Range: {lower_bound} to {upper_bound}")
        print("人类前1%点正态分布的均值")
        mean, std_dev, lower_bound, upper_bound = fit_gaussian_and_find_3sigma(human_score[int(0.99 * len(newdata)):])
        print(f"Mean: {mean}, Standard Deviation: {std_dev}")
        print(f"3-Sigma Range: {lower_bound} to {upper_bound}")
        
        popmean =  0.11 # 零假设中的总体均值

# 执行单边单样本t检验，检验样本均值是否大于总体均值
        t_stat, p_value = perform_one_sided_ttest_1samp(newdata[int(0.99 * len(data)):], popmean, alternative='greater')

        print("t-statistic:", t_stat)
        print("p-value:", p_value)
        
        # 执行单边t检验，检验样本1的均值是否大于样本2
        # t_stat, p_value = perform_one_sided_ttest(newdata[int(0.99 * len(newdata)):], human_score[int(0.99 * len(newdata)):], alternative='greater')

        # print("t-statistic:", t_stat)
        # print("p-value:", p_value)

        # 判断显著性
        if p_value < 0.05:
            print("样本1的均值在统计学上显著大于样本2。")
        else:
            print("样本1的均值在统计学上不显著大于样本2。")
        print("------------------------")








