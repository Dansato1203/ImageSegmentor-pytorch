#! /bin/bash

cd /train
python3 train.py
python3 model_trace.py --weight-name $1
