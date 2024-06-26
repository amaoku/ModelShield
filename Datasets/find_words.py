import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet
import numpy
nltk.download('wordnet')

def read_list_from_file(file_path):

    with open(file_path, 'r') as f:
        lst=eval(f.read())
        return lst

def get_output_list(input_list):
    output_list=[]
    for i in input_list:
        output_list.append(i['instruction'])
    return output_list

def count_frequency_of_words_sorted(input_list):
    input_list_split=[item.split() for item in input_list]
    word_dict={}
    for i in input_list_split:
        for j in i:
            if j in word_dict:
                word_dict[j]+=1
            else:
                word_dict[j]=1



    return sorted(word_dict.items(),key=lambda x:x[1],reverse=True)




def dic2json(dic,file_path):
    with open(file_path,'w') as f:
        json.dump(dic,f)


def find_synonym(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

# def add_synonym_to_dic(dic):
#
#     for i in list(dic):
#         i['synonym']=find_synonym(i[0])
#     return list(dic)
def TDIDF(input_list):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(input_list)
    words = vectorizer.get_feature_names()
    weight = X.toarray()

    for i in range(len(input_list)):
        tfidf_scores=[(words[j],X[i,j]) for j in range(X.shape[1])]
        tfidf_scores=sorted(tfidf_scores,key=lambda x:x[1],reverse=True)
        print("Sentence",i)
        # for item in tfidf_scores:
        #     print("%s:%f"%(item[0],item[1]))
    return weight

if __name__ == '__main__':
    input_list=read_list_from_file('/data3/WaterMarking/HC3_ChatGPT_deduplication.json')
    output_list=get_output_list(input_list)
    sorted_frequency_dic=count_frequency_of_words_sorted(output_list)

    weight=TDIDF(output_list)
    # add_syn_dic=add_synonym_to_dic(sorted_frequency_dic)
    empty_list=[]
    for i in sorted_frequency_dic:
        empty_list.append({'word':i[0],'frequency':i[1],'synonym':find_synonym(i[0])})

    dic2json(empty_list,'/data3/WaterMarking/HC3_ChatGPT_deduplication_sorted_frequency_instruction_human.json')
