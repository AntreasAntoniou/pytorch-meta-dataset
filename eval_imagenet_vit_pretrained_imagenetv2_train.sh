export architecture=vit_large_patch16_224

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source aircraft \
				 test_source aircraft \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source traffic_sign \
				 test_source traffic_sign \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source quickdraw \
				 test_source quickdraw \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source dtd \
				 test_source dtd \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source omniglot \
				 test_source omniglot \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source fungi \
				 test_source fungi \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source mscoco \
				 test_source mscoco \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\

PYTHONHASHSEED=0 python -m src.eval  --base_config config/base.yaml \
				--method_config config/method/finetune.yaml \
				--data_config config/data/variable.yaml \
				--opts \
				 base_source ilsvrc_2012_v2 \
				 val_source cu_birds \
				 test_source cu_birds \
				 arch $architecture \
				 val_episodes 10 \
				 eval_mode test \
				 val_batch_size 1 \
				 loader_version pytorch \
				 load_from_timm False ;\
