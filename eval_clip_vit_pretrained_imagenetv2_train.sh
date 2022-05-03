
export architecture=clip_vit_b_16_pretrained
gpus=$1
for dataset in aircraft traffic_sign dtd omniglot mscoco cu_birds
  do
    for method in config/method/protonet.yaml config/method/finetune.yaml config/method/finetune-all.yaml config/method/finetune-with-instance-norm.yaml config/method/finetune-with-instance-norm-all.yaml config/method/simpleshot.yaml config/method/maml.yaml config/method/tim_adm.yaml

      do
        echo "python3 eval.py --architecture $architecture --method $method --dataset $dataset --gpus $gpus"
        PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
        --method_config $method \
        --data_config config/data/variable.yaml \
        --opts \
         gpus $gpus \
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
