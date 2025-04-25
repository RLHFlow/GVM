initial_model=$1
act_params="embed_tokens" # embed_tokens, lm_head, etc.
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
model_prefix=$5
data_start=0
data_end=200000000
stage_1_samples_per_prompt=$3
stage_2_samples=$((stage_1_samples_per_prompt*(data_end-data_start)))
stage_2_samples_per_prompt=$4
train_size=200000000
alpha=$9
beta=${10}
system_prompt="qwen25-math-cot" # "qwen25-math-cot", "hendrydong-longcot"
filter_threshold=$6
filter_insufficient=$7

i=$2
iteration_num=$i
model_name_or_path=$initial_model
data_path=ScaleML-RLHF/numina_math_${i}
suffix=$8 # "numina_math_${i}_n${stage_1_samples_per_prompt}"

mkdir -p data/${model_prefix}/${suffix}/data_${i}

for i in $(seq 0 $((my_world_size - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_1_collect_data.py --local_index $i --world_size $my_world_size \
        --model_name_or_path $model_name_or_path --iter $iteration_num --data_path $data_path \
        --model_prefix=$model_prefix --end=$data_end --suffix=$suffix --stage_1_samples=$stage_1_samples_per_prompt \
        --system_prompt=$system_prompt &
done

wait

for i in $(seq 0 $((my_world_size - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python stage_2_calc_acceptRates_grads.py --local_index $i --iter $iteration_num \
        --model_name_or_path=$model_name_or_path --act_params=$act_params --model_prefix=$model_prefix \
        --end=$data_end --suffix=$suffix --num_collect_files=$my_world_size --stage_1_samples=$stage_1_samples_per_prompt \
        --system_prompt=$system_prompt &
done

wait

python stage_2_calc_sample_size.py --num_collect_files=$my_world_size --suffix=$suffix --iter=$iteration_num \
    --model_prefix=$model_prefix --stage_2_samples=$stage_2_samples --alpha=$alpha --beta=$beta --stage_1_samples=$stage_2_samples_per_prompt \
    --filter_threshold=$filter_threshold --filter_insufficient=$filter_insufficient
