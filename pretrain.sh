export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=8086 pretrain.py \
    --model_name_or_path llama-220m/ \
    --train_data_path ./data/pretrain_train_example.csv \
    --val_data_path ./data/pretrain_val_example.csv \
    --bf16 True \
    --output_dir output/pretrain/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0001 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 32 \
    --full_determinism \
    --tf32 False \
    --model_max_length 601 \
    --report_to tensorboard \
    --dataloader_num_workers 2 \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
