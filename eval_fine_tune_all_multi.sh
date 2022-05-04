gpus=$1
export CUDA_VISIBLE_DEVICES=$gpus
for architecture in clip_vit_b_16_pretrained vit_base_patch16_224 modus_prime_tali_viat_pretrained modus_prime_tali_viat_scratch
  do
    for dataset in aircraft traffic_sign dtd omniglot mscoco cu_birds
      do
        for method in config/method/finetune_all.yaml config/method/finetune-with-instance-norm.yaml config/method/finetune-with-instance-norm-all.yaml

          do
            echo "python3 eval.py --architecture $architecture --method $method --dataset $dataset --gpus $gpus"
            PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
            --method_config $method \
            --data_config config/data/variable.yaml \
            --opts \
             base_source ilsvrc_2012_v2 \
             val_source $dataset \
             test_source $dataset \
             arch $architecture \
             val_episodes 10 \
             eval_mode test \
             val_batch_size 1 \
             loader_version pytorch \
             load_from_timm True ;\
          done
      done
  done
