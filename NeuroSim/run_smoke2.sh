#!/bin/bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/ubuntu/.local/bin
export HF_ENDPOINT=https://hf-mirror.com
cd /home/ubuntu/iccad2026_bxkj/NeuroSim
/usr/bin/python3 llm_inference.py --model gpt2 --task smooth --adc_bits 7 --num_calib_batches 4 --num_eval_batches 10 --output_dir ./results/smoke_test2 --device cpu > /home/ubuntu/iccad2026_bxkj/NeuroSim/smoke_test2_out.log 2>&1
echo "EXIT=$?" >> /home/ubuntu/iccad2026_bxkj/NeuroSim/smoke_test2_out.log
