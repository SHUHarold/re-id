#!/bin/sh
python train_img_model_xent.py -d market1501 -a resnet50 --max-epoch 60 --train-batch 32 --test-batch 32 --stepsize 20 --eval-step 20 --save-dir log/resnet50-xent-market1501 --gpu-devices 0,1 -j 0
