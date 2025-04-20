# Run all LoRA + AWQ trials (LoRA then AWQ)
# IMPORTANT: Need to obtain lora_merged_models before running this from wanda repo's lora_ft/merge_lora.py


echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_0.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_1.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_2.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_3.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_4.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_5.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_6.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_7.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_8.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_0_9.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_2_4.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w4_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w4_q8/llama_7b_4_8.pt >> awq_wanda_ft/w4_q8/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_0.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_1.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_2.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_3.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_4.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_5.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_6.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_7.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_8.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w4_q16/report.log
# python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_0_9.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_2_4.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w4_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w4_q16/llama_7b_4_8.pt >> awq_wanda_ft/w4_q16/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_0.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_1.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_2.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_3.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_4.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_5.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_6.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_7.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_8.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_0_9.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_2_4.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w4_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w4_q32/llama_7b_4_8.pt >> awq_wanda_ft/w4_q32/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_0.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_1.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_2.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_3.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_4.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_5.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_6.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_7.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_8.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_0_9.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_2_4.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w4_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w4_q64/llama_7b_4_8.pt >> awq_wanda_ft/w4_q64/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_0.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_1.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_2.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_3.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_4.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_5.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_6.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_7.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_8.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_0_9.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_2_4.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w4_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w4_q128/llama_7b_4_8.pt >> awq_wanda_ft/w4_q128/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_0.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_1.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_2.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_3.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_4.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_5.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_6.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_7.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_8.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_0_9.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_2_4.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w3_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w3_q8/llama_7b_4_8.pt >> awq_wanda_ft/w3_q8/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_0.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_1.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_2.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_3.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_4.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_5.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_6.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_7.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_8.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_0_9.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_2_4.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w3_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w3_q16/llama_7b_4_8.pt >> awq_wanda_ft/w3_q16/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_0.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_1.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_2.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_3.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_4.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_5.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_6.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_7.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_8.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_0_9.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_2_4.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w3_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w3_q32/llama_7b_4_8.pt >> awq_wanda_ft/w3_q32/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_0.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_1.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_2.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_3.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_4.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_5.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_6.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_7.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_8.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_0_9.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_2_4.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w3_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w3_q64/llama_7b_4_8.pt >> awq_wanda_ft/w3_q64/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_0.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_1.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_2.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_3.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_4.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_5.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_6.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_7.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_8.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_0_9.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_2_4.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w3_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w3_q128/llama_7b_4_8.pt >> awq_wanda_ft/w3_q128/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_0.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_1.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_2.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_3.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_4.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_5.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_6.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_7.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_8.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_0_9.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_2_4.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w2_q8/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 8 --run_awq --dump_awq awq_wanda_ft/w2_q8/llama_7b_4_8.pt >> awq_wanda_ft/w2_q8/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_0.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_1.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_2.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_3.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_4.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_5.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_6.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_7.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_8.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_0_9.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_2_4.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w2_q16/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 16 --run_awq --dump_awq awq_wanda_ft/w2_q16/llama_7b_4_8.pt >> awq_wanda_ft/w2_q16/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_0.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_1.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_2.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_3.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_4.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_5.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_6.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_7.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_8.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_0_9.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_2_4.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w2_q32/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 32 --run_awq --dump_awq awq_wanda_ft/w2_q32/llama_7b_4_8.pt >> awq_wanda_ft/w2_q32/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_0.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_1.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_2.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_3.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_4.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_5.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_6.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_7.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_8.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_0_9.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_2_4.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w2_q64/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 64 --run_awq --dump_awq awq_wanda_ft/w2_q64/llama_7b_4_8.pt >> awq_wanda_ft/w2_q64/report.log
echo -e " Running llama_7b_0_0...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_0 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_0.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_1...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_1 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_1.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_2...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_2 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_2.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_3...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_3 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_3.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_4...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_4 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_4.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_5...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_5 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_5.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_6...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_6 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_6.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_7...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_7 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_7.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_8...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_8 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_8.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_0_9...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_0_9 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_0_9.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_2_4...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_2_4 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_2_4.pt >> awq_wanda_ft/w2_q128/report.log
echo -e " Running llama_7b_4_8...\n" >> awq_wanda_ft/w2_q128/report.log
python -m awq.entry --model_path ../wanda/lora_ft/lora_merged_models/wanda/llama_7b_4_8 --cache_dir llm_weights --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_wanda_ft/w2_q128/llama_7b_4_8.pt >> awq_wanda_ft/w2_q128/report.log
