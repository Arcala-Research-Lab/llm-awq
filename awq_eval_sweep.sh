#Run Perplexity Evaluation on Wanda + Awq models


echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --load_awq awq_wanda_ft/w4_q8/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q8/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --load_awq awq_wanda_ft/w4_q16/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q16/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --load_awq awq_wanda_ft/w4_q32/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q32/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --load_awq awq_wanda_ft/w4_q64/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q64/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w4_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --load_awq awq_wanda_ft/w4_q128/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w4_q128/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --load_awq awq_wanda_ft/w3_q8/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q8/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --load_awq awq_wanda_ft/w3_q16/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q16/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --load_awq awq_wanda_ft/w3_q32/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q32/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --load_awq awq_wanda_ft/w3_q64/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q64/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w3_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --load_awq awq_wanda_ft/w3_q128/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w3_q128/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q8/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --load_awq awq_wanda_ft/w2_q8/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q8/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q16/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --load_awq awq_wanda_ft/w2_q16/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q16/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q32/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --load_awq awq_wanda_ft/w2_q32/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q32/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q64/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --load_awq awq_wanda_ft/w2_q64/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q64/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_0 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_0.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_1 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_1.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_2 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_2.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_3 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_3.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_5 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_5.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_6 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_6.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_7 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_7.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_0_9 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_0_9.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_2_4 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_2_4.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 2048...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 2048 >> awq_wanda_ft/w2_q128/eval_report.log
echo -e " Evaluating llama_7b_4_8 with seqlen 4096...\n" >> awq_wanda_ft/w2_q128/eval_report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --load_awq awq_wanda_ft/w2_q128/llama_7b_4_8.pt --q_backend "fake" --tasks "wikitext" --eval_seqlen 4096 >> awq_wanda_ft/w2_q128/eval_report.log
