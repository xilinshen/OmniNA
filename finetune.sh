export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=8048 finetune.py \
    --model_name_or_path ./output/pretrain/checkpoint-10/ \
    --train_data_path data/finetune_train_example.csv \
    --val_data_path data/finetune_val_example.csv \
    --bf16 True \
    --bf16_full_eval True \
    --output_dir ./output/finetune/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 20 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 25 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --full_determinism \
    --tf32 False \
    --model_max_length 601 \
    --report_to tensorboard \
    --dataloader_num_workers 2 \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
