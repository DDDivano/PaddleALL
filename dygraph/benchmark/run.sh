#!/bin/bash

python3.6 dygraph_mnist.py
nohup nvidia-smi -lms 1000 --query-gpu=memory.total,memory.used,memory.free,index,timestamp,name --format=csv -i 0 > memory.log 2>&1 &

