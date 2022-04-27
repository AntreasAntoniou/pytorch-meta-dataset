PYTHONHASHSEED=0 python -m src.train --base_config config/base.yaml \
                        --method_config config/method/finetune.yaml \
                        --data_config config/data/variable.yaml \
                        --opts \
                        base_source ilsvrc_2012_v2 \
                        val_source aircraft \
                        arch clip_vit_b_16_pretrained \
                        debug False \
                        loader_version pytorch \
                        load_from_timm False \
                        num_workers 20 \

