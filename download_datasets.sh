#!/bin/bash
conda install unzip

# Download Omniglot, create a folder for it and extract it there
cd $DATASRC
mkdir -p omniglot
cd omniglot
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
unzip images_background.zip -d .
unzip images_evaluation.zip -d .


# Download fgvc-aircraft-2013b, create a folder for it and extract it there
cd $DATASRC
mkdir -p fgvc-aircraft-2013b
cd fgvc-aircraft-2013b
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xvf fgvc-aircraft-2013b.tar.gz


cd $DATASRC
mkdir -p CUB_200_2011
cd CUB_200_2011
wget https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/ # TODO report not existing, replace
mv index.html CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz


cd $DATASRC
mkdir -p dtd
cd dtd
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvf dtd-r1.0.1.tar.gz


mkdir -p $DATASRC/quickdraw
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $DATASRC/quickdraw


cd $DATASRC
mkdir -p fungi
cd fungi
wget https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz
wget https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz
tar -xvf fungi_train_val.tgz
tar -xvf train_val_annotations.tgz


cd $DATASRC
mkdir -p vgg_flower
cd vgg_flower
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
tar -xvf 102flowers.tgz


cd $DATASRC
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip -d $DATASRC/


mkdir -p $DATASRC/mscoco/train2017
cd $DATASRC/mscoco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
