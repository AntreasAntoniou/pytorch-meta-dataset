#!/bin/bash
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=ilsvrc_2012_v2 \
  --ilsvrc_2012_data_root=$DATASRC/imagenet/train \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=omniglot \
  --omniglot_data_root=$DATASRC/omniglot \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=aircraft \
  --aircraft_data_root=$DATASRC/fgvc-aircraft-2013b \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=cu_birds \
  --cu_birds_data_root=$DATASRC/CUB_200_2011 \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=dtd \
  --dtd_data_root=$DATASRC/dtd \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=quickdraw \
  --quickdraw_data_root=$DATASRC/quickdraw \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=fungi \
  --fungi_data_root=$DATASRC/fungi \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=vgg_flower \
  --vgg_flower_data_root=$DATASRC/vgg_flower \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=traffic_sign \
  --traffic_sign_data_root=$DATASRC/GTSRB \
  --splits_root=$SPLITS \
  --records_root=$RECORDS


python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=mscoco \
  --mscoco_data_root=$DATASRC/mscoco \
  --splits_root=$SPLITS \
  --records_root=$RECORDS