#!/bin/bash
nohup python dataset/split_to_lmdb.py --input_path /home/nkcemeka/Documents/Datasets/slakh2100_flac_redux/train --output_path ./lmdb_midi --emb_model_path ./pretrained/AE_slakh.pt --slakh True > logextract.log 2>&1 &
echo $! > extract_pid.txt
