
import os
import json

def read_list_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_list_to_json(file_path,data):
    with open(file_path, "w") as f:
        json.dump(data,f,indent=4)

def get_sorted_word_frequency_list():
    word_frequency_list=[]
    read_all_frequency_words=read_list_from_json("/data3/WaterMarking/wild_dataset/wild_human_sorted_frequency.json")
    for item in read_all_frequency_words:
        if item["frequency"]<5:
            word_frequency_list.append(item["word"])
    return word_frequency_list

def find_frequent_good_wm_sets(candidate_set_frequency_WM_Words):
    dict={}
    for item in candidate_set_frequency_WM_Words:
        if item not in dict:
            dict[item]=1
        else:
            dict[item]+=1
    return dict

def sort_dict_by_value(dict):
    return sorted(dict.items(), key=lambda x: x[1], reverse=True)





if __name__ == "__main__":
    print("start")
    candidate_set_frequency_WM_Words=[]
    WM_FT_100=[]
    WM_FT_100_data=read_list_from_json("/data3/WaterMarking/predictions_0531_WM_GPT2_100epoch.json")
    for i in range(4234):
        WM_FT_100.append(WM_FT_100_data[i]["predict"])

    ChatGPT_FT_100=[]
    ChatGPT_FT_100_data=read_list_from_json("/data3/WaterMarking/predictions_0531_NO_WM_trained_GPT2_ChatgptHC3-100epoch.json")
    for i in range(4234):
        ChatGPT_FT_100.append(ChatGPT_FT_100_data[i]["predict"])
    # human_FT=[]
    # human_FT_data=read_list_from_json("/data3/WaterMarking/predictions_0520_NO_WM_trained_GPT2_HUMAN.json")
    # for i in range(4234):
    #     human_FT.append(human_FT_data[i]["predict"])
    #
    # GPT2_ori=[]
    # GPT2_ori_data=read_list_from_json("/data3/WaterMarking/predictions_0519_origin_GPT2.json")
    # for i in range(4234):
    #     GPT2_ori.append(GPT2_ori_data[i]["predict"])
    human_count_i=0

    gpt_result=read_list_from_json("/data3/WaterMarking/0523_all_data_4234.json")
    word_frequency_list=get_sorted_word_frequency_list()
    for item in gpt_result:


        # item["human_FT"]=human_FT[human_count_i]
        # item["GPT2_ori"]=GPT2_ori[human_count_i]
        item["GPT2_FT_WM"]=WM_FT_100[human_count_i]
        item["GPT2_FT"]=ChatGPT_FT_100[human_count_i]
        human_count_i+=1
        good_WM=[]
        wmword=item["WM"].split(",")
        for word in wmword:
            if (word not in item["query"])&(word in word_frequency_list):
                good_WM.append(word)
                candidate_set_frequency_WM_Words.append(word)
        item["good_WM"]=good_WM

    #     count_dic_in_WM_FT={}
    #     count_dic_in_FT={}
    #     count_dict_in_human_FT={}
    #     count_dict_in_GPT2_ori={}
    #     average_count_in_WM_FT={}
    #     average_count_in_FT={}
    #     average_count_in_human_FT={}
    #     average_count_in_GPT2_ori={}
    #
    #
    #     for water_mark in item['WM'].split(","):
    #         count_dic_in_WM_FT[water_mark]=item["GPT2_FT_WM"].count(water_mark)
    #         # count_in_WM_FT=len(item["GPT2_FT_WM"].split(water_mark))-1
    #         average_count_in_WM_FT[water_mark]=count_dic_in_WM_FT[water_mark]/len(item["GPT2_FT_WM"])
    #
    #         count_dic_in_FT[water_mark]=item["GPT2_FT"].count(water_mark)
    #         average_count_in_FT[water_mark]=count_dic_in_FT[water_mark]/len(item["GPT2_FT"])
    #
    #         count_dict_in_human_FT[water_mark]=item["human_FT"].count(water_mark)
    #         average_count_in_human_FT[water_mark]=count_dict_in_human_FT[water_mark]/len(item["human_FT"])
    #
    #         count_dict_in_GPT2_ori[water_mark]=item["GPT2_ori"].count(water_mark)
    #         if len(item["GPT2_ori"])!=0:
    #             average_count_in_GPT2_ori[water_mark]=count_dict_in_GPT2_ori[water_mark]/len(item["GPT2_ori"])
    #         else:
    #             average_count_in_GPT2_ori[water_mark]=0
    #
    #
    #
    #     item["count_dic_in_WM_FT"]=count_dic_in_WM_FT
    #     item["count_dic_in_FT"] = count_dic_in_FT
    #     item["count_dict_in_human_FT"] = count_dict_in_human_FT
    #     item["count_dict_in_GPT2_ori"] = count_dict_in_GPT2_ori
    #     item["average_count_in_WM_FT"]=average_count_in_WM_FT
    #
    #     item["average_count_in_FT"]=average_count_in_FT
    #
    #     item["average_count_in_human_FT"]=average_count_in_human_FT
    #     item["average_count_in_GPT2_ori"]=average_count_in_GPT2_ori
    #
    # count_dic_of_frequency_of_good_WM=find_frequent_good_wm_sets(candidate_set_frequency_WM_Words)
    # new_dict=sort_dict_by_value(count_dic_of_frequency_of_good_WM)
    # write_list_to_json("count_dic_of_frequency_of_good_WM.json", new_dict)
    #
    #

    write_list_to_json("/data3/WaterMarking/0601_all_data_4234_100epoch_WMFT_FT.json",gpt_result)


    # count_result=[]
    # for item in gpt_result:
    #     wm_in_wm_ft_generation=[]
    #     wm_in_ft_generation=[]
    #     wm_in_HC3_chatgpt=[]
    #     wmword=item["WM"].split(",")
    #     for word in wmword:
    #         if word in item["GPT2_FT_WM"]:
    #             count_result.append(1)
    #             wm_in_wm_ft_generation.append(word)
    #         if word in item["GPT2_FT"]:
    #             wm_in_ft_generation.append(word)
    #         if word in item["HC3_Chatgpt_origin"]:
    #             wm_in_HC3_chatgpt.append(word)
    #
    #
    #     item["WM_in_WM-FT"]=wm_in_wm_ft_generation
    #     item["WM_in_FT"]=wm_in_ft_generation
    #     item["WM_in_HC3"]=wm_in_HC3_chatgpt
    #
    # write_list_to_json("/data3/WaterMarking/0518_all_data_4234_without_oriGPT_with_WM_in_Generation.json",gpt_result)


