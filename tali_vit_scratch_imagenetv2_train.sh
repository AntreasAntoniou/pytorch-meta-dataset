PYTHONHASHSEED=0 python -m src.train --base_config config/base.yaml \
                        --method_config config/method/finetune.yaml \
                        --data_config config/data/variable.yaml \
                        --opts \
                        base_source ilsvrc_2012_v2 \
                        val_source aircraft \
                        arch modus_prime_tali_viat_scratch \
                        debug False \
                        loader_version pytorch \
                        load_from_timm True \
                        num_workers 20 \

