export WANDB_PROJECT="Pose-Estimation"
export WANDB_RUN_GROUP='SigLIPose'
export WANDB_WATCH=""

export TORCH_DISTRIBUTED_DEBUG="OFF"
export TORCHDYNAMO_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export OMP_NUM_THREADS=2

# export ACCELERATE_FP8_BACKEND="TE"
# export ACCELERATE_FP8_FORMAT="HYBRID"

export ACCELERATE_USE_FSDP=true
export FSDP_VERSION=2
export FSDP_RESHARD_AFTER_FORWARD=true
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
export FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP"
export FSDP_CPU_RAM_EFFICIENT_LOADING=false
export FSDP_ACTIVATION_CHECKPOINTING=false

# accelerate launch --config_file="/root/workspace/config/fsdp.yaml" \
deepspeed --include=localhost:0,1,2,3 --master_port=8532 \
    '/home/jp/workspace/SigLIPose/src/main.py' \
    --output_dir="/home/jp/output_dir/SigLIPose2/train" \
    --run_name="sft" \
    --per_device_train_batch_size=10 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=1 \
    --num_train_epochs=3 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to='wandb' \
    --learning_rate=2e-5 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --eval_strategy='steps' \
    --eval_steps=100 \
    --save_strategy='epoch' \
    --save_steps=500 \
    --logging_strategy='steps' \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --ddp_timeout=18000000 \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --dataloader_prefetch_factor=5 \
    --dataloader_num_workers=4 \
    --torch_dtype='bfloat16' \
    --remove_unused_columns=true
    # --optim='lomo' \
    # --deepspeed='/root/workspace/config/ZeRO_3_act_check.json'
