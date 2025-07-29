set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS

data=numina_math_em
project_name=em-raft
algorithm=grpo
model=Llama-3.2-1B-Instruct
policy_loss=plusplus # vanilla, plusplus (importance sample + clipping)
stage_1_samples_per_prompt=8
stage_2_samples_per_prompt=8
filter_threshold=1.0
filter_insufficient=0.0
alpha=0.001
beta=2.0
rollout_n=1
data_shuffle=True
clip_higher_ratio=0.3

for i in {1..10}; do
    if [ $i -eq 1 ]; then
        model_name_or_path=meta-llama/Llama-3.2-1B-Instruct
    else
        model_name_or_path=checkpoints/em-raft/${model}-${algorithm}-${policy_loss}-${data}-n${stage_1_samples_per_prompt}-${stage_2_samples_per_prompt}-clipH${clip_higher_ratio}-iter$((i-1))/global_step_9/actor/huggingface 
    fi

    if [ $i -ne 0 ]; then
        cd em/
        em_model_name_or_path=$model_name_or_path
        if [ $i -ne 1 ]; then
            em_model_name_or_path="../$em_model_name_or_path"
        fi
        bash run_em.sh $em_model_name_or_path $i $stage_1_samples_per_prompt $stage_2_samples_per_prompt $model $filter_threshold $filter_insufficient "numina_math_${i}_n${stage_1_samples_per_prompt}_${stage_2_samples_per_prompt}_filter${filter_threshold}_insufficient${filter_insufficient}" $alpha $beta
        wait
        cd ..
    fi
    
    iter=$i
    experiment_name=${model}-${algorithm}-${policy_loss}-${data}-n${stage_1_samples_per_prompt}-${stage_2_samples_per_prompt}-clipH${clip_higher_ratio}-iter${iter}
    GPUS=(0 1 2 3 4 5 6 7)
    my_world_size=${#GPUS[@]}
    total_epochs=1

    if [ $i -eq 0 ]; then
        sample_sizes_data="em/data/Llama-3.2-3B-Instruct/numina_math_1_n8_8_filter1.0_insufficient0.0/data_1/sample_sizes_sample8_a0.001_b2.0.json"
    else
        sample_sizes_data="em/data/${model}/numina_math_${iter}_n${stage_1_samples_per_prompt}_${stage_2_samples_per_prompt}_filter${filter_threshold}_insufficient${filter_insufficient}/data_${iter}/sample_sizes_sample${stage_2_samples_per_prompt}_a${alpha}_b${beta}.json"
    fi

    math_train_path=./data/numina_math_${iter}/train.parquet
    math_test_path=./data/math500/test.parquet 

    train_files="['$math_train_path']"
    test_files="['$math_test_path']"

    mkdir -p logs/${project_name}

    start_model=$model_name_or_path
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$algorithm \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=3072 \
        data.filter_overlong_prompts=True \
        data.shuffle=$data_shuffle \
        actor_rollout_ref.model.path="$start_model" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.policy_loss=$policy_loss \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.rollout.name=vllm \
        +actor_rollout_ref.rollout.use_em=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=$rollout_n \
        +actor_rollout_ref.rollout.sample_sizes_data="$sample_sizes_data" \
        actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${project_name} \
        trainer.experiment_name=${experiment_name} \
        trainer.n_gpus_per_node=$my_world_size \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.save_freq=5 \
        trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
        trainer.test_freq=5 \
        trainer.total_epochs=$total_epochs 2>&1 | tee logs/${project_name}/${experiment_name}.log

    python scripts/legacy_model_merger.py merge --backend=fsdp --hf_model_path=meta-llama/$model --local_dir=checkpoints/${project_name}/${experiment_name}/global_step_9/actor --target_dir=checkpoints/${project_name}/${experiment_name}/global_step_9/actor/huggingface
done
