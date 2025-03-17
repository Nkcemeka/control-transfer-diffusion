#!/bin/bash
nohup python train_diffusion.py --name test_01 --db_path ./lmdb_midi  --config diffusion/configs/config --dataset_type waveform --gpu 0 --emb_model_path ./pretrained/AE_slakh.pt > logfile.log 2>&1 &
echo $! > train_pid.txt
