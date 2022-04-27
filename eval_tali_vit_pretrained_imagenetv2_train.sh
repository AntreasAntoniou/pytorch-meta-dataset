PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source aircraft \
				 test_source aircraft \
				 arch modus_prime_tali_viat_pretrained \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

