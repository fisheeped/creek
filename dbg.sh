#!/bin/bash
# -*- coding: utf-8 -*-

set -e

# 回到脚本路径
cd `dirname $0`

# 避免系统过载，否则警告
export OMP_NUM_THREADS=1
# 防止新建进程报错
export TERM=xterm


base_dir="/ceph2/yy/note/myllama"
include="0,1"

type_name=sft
job_name=${type_name}_`date +%m%d%H%M`
log_dir=$base_dir/output/${job_name}
code_dir=$base_dir/train



mkdir -p ${log_dir}
mkdir -p $base_dir/logs/tensorboard/$job_name
cp -r $code_dir ${log_dir}

echo code_dir:$code_dir

# my wandb
export WANDB_API_KEY=5728bf4cd5eb19136f52e2da3e30b7544c1bcab0
export WANDB_PROJECT=${type_name}
export WANDB_MODE=offline



Model_run() {
    cp $0 ${log_dir}
    cp $1 ${log_dir}
    echo "-----------job-line-----------" >> ${log_dir}/${type_name}.txt 2>&1
    port=`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`
    python -m debugpy --listen 42188 --wait-for-client -m deepspeed.launcher.runner \
        --include "localhost:$include" \
        --master_port $port $1 \
        --deepspeed $code_dir/configs/ds_config_zero2.json \
        --model_path $base_dir/output/tf_pretrain_04040505/model/checkpoint-1282 \
        --data_path $base_dir/data/kuakua \
        --cache True \
        --do_train True \
        --do_eval False \
        --bf16 True \
        --learning_rate 3e-5 \
        --per_device_train_batch_size 56 \
        --gradient_accumulation_steps 2 \
        --lr_scheduler_type cosine \
        --warmup_steps 10000 \
        --num_train_epochs 20 \
        --learning_rate 1e-4 \
        --use_fast_tokenizer True \
        --num_proc 64 \
        --save_strategy epoch \
        --output_dir ${log_dir}/model \
        --overwrite_output_dir \
        --save_safetensor True \
        --report_to tensorboard \
        --logging_strategy steps \
        --logging_steps 2 \
        --logging_dir $base_dir/logs/tensorboard/$job_name \
        --run_name ${job_name} 2>&1|tee -a ${log_dir}/${type_name}.txt
    # --num_train_epochs 3 \
            # --save_strategy epoch \
        # --evaluation_strategy  steps \

    sleep 2
}


Nvidia_log() {
    echo "-----------job-line-----------" >> ${log_dir}/nvidia_log.txt
    while true
    do
        date +"%y%m%d-%H%M%S" >> ${log_dir}/nvidia_log.txt
        nvidia-smi \
        --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory \
        --format=csv >> ${log_dir}/nvidia_log.txt
        free -h >> ${log_dir}/nvidia_log.txt
        sleep 5
    done
}



# 运行并获取pid
Model_run $code_dir/model_finetune.py & pid_run=$!
Nvidia_log & pid_log=$!

echo Model_run pid $pid_run
echo Model_log pid $pid_log

# 等待训练主程序完成，然后关掉日志进程
wait "$pid_run"
kill "$pid_log"


sleep 2
echo "success $0"