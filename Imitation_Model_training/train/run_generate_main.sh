echo "start llama2 wild"
python generate_llama2_noWM.py --dev_file='/data1/pky/dataset/instinwild_en_4000.json' --model_name_or_path='/data1/pky/output/epoch_llama2/trained_model_llama2-wild-WM-epoch_5' --output_file="/data1/pky/output/main_repeat/predictions_0623_llama2_wild_main_repeat_4.json"
python generate_llama2_noWM.py --dev_file='/data1/pky/dataset/instinwild_en_4000.json' --model_name_or_path='/data1/pky/output/epoch_llama2/trained_model_llama2-wild-WM-epoch_5' --output_file="/data1/pky/output/main_repeat/predictions_0623_llama2_wild_main_repeat_5.json"
python generate_llama2_noWM.py --dev_file='/data1/pky/dataset/instinwild_en_4000.json' --model_name_or_path='/data1/pky/output/epoch_llama2/trained_model_llama2-wild-WM-epoch_5' --output_file="/data1/pky/output/main_repeat/predictions_0623_llama2_wild_main_repeat_6.json"
echo "end llama2 wild"
# python generate_mistral_noWM.py --dev_file='/data1/pky/dataset/instinwild_en_4000.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_wild' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_wild_main_repeat_4_new.json"
# python generate_mistral_noWM.py --dev_file='/data1/pky/dataset/instinwild_en_4000.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_wild' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_wild_main_repeat_5_new.json"
# python generate_mistral_noWM.py --dev_file='/data1/pky/dataset/instinwild_en_4000.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_wild' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_wild_main_repeat_6_new.json"
# python generate_mistral_noWM.py --dev_file='/data1/pky/dataset/instinwild_en_4000.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_wild' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_wild_main_repeat_3.json"

# python /data1/pky/BELLE/train/generate_gpt2_epochsy.py --dev_file='/data1/pky/dataset/lastWM.json' --model_name_or_path='/data1/pky/output/epoch_gpt2/gpt2-WM-wild-epoch-9' --output_file='/data1/pky/output/main_repeat/predictions_0622_gpt2_wild_main_repeat_4.json'
# python /data1/pky/BELLE/train/generate_gpt2_epochsy.py --dev_file='/data1/pky/dataset/lastWM.json' --model_name_or_path='/data1/pky/output/epoch_gpt2/gpt2-WM-wild-epoch-9' --output_file='/data1/pky/output/main_repeat/predictions_0622_gpt2_wild_main_repeat_5.json'
# python /data1/pky/BELLE/train/generate_gpt2_epochsy.py --dev_file='/data1/pky/dataset/lastWM.json' --model_name_or_path='/data1/pky/output/epoch_gpt2/gpt2-WM-wild-epoch-9' --output_file='/data1/pky/output/main_repeat/predictions_0622_gpt2_wild_main_repeat_6.json'

echo "start llama2 hc3"
python /data1/pky/BELLE/train/generate_llama2_WILD.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/epoch_llama2/trained_model_llama2-hc3-WM-epoch_5' --output_file="/data1/pky/output/main_repeat/predictions_0623_llama2_wild_man_repeat_4.json"
python /data1/pky/BELLE/train/generate_llama2_WILD.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/epoch_llama2/trained_model_llama2-hc3-WM-epoch_5' --output_file="/data1/pky/output/main_repeat/predictions_0622_llama2_wild_main_repeat_5.json"
python /data1/pky/BELLE/train/generate_llama2_WILD.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/epoch_llama2/trained_model_llama2-hc3-WM-epoch_5' --output_file="/data1/pky/output/main_repeat/predictions_0622_llama2_wild_main_repeat_6.json"
echo "end llama2 hc3"
# python /data1/pky/BELLE/train/generate_mistral_noWM_copy.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_hc3' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_hc3_main_repeat_4_new.json"
# python /data1/pky/BELLE/train/generate_mistral_noWM_copy.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_hc3' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_hc3_main_repeat_5_new.json"
# python /data1/pky/BELLE/train/generate_mistral_noWM_copy.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_hc3' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_hc3_main_repeat_6_new.json"
# python /data1/pky/BELLE/train/generate_mistral_noWM_copy.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/wm_mistrals/wm_mistral_hc3' --output_file="/data1/pky/output/main_repeat/predictions_0604_mistral_hc3_main_repeat_3.json"

# python /data1/pky/BELLE/train/generate_gpt2_epochsy.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/epoch_gpt2/gpt2-WM-hc3-epoch-9' --output_file='/data1/pky/output/main_repeat/predictions_0622_gpt2_hc3_main_repeat_4.json'
# python /data1/pky/BELLE/train/generate_gpt2_epochsy.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/epoch_gpt2/gpt2-WM-hc3-epoch-9' --output_file='/data1/pky/output/main_repeat/predictions_0622_gpt2_hc3_main_repeat_5.json'
# python /data1/pky/BELLE/train/generate_gpt2_epochsy.py --dev_file='/data1/pky/dataset/Train_for_finetune_1-5158_filter_4234_survival.json' --model_name_or_path='/data1/pky/output/epoch_gpt2/gpt2-WM-hc3-epoch-9' --output_file='/data1/pky/output/main_repeat/predictions_062_gpt2_hc3_main_repeat_6.json'



