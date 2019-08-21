#!/bin/bash
python3.6 -m paddle.distributed.launch --selected_gpus=0,1,2,3,4,5,6,7 --log_dir $1 train.py   --use_data_parallel 1 --epoch 2
