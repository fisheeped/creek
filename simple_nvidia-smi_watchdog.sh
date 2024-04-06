watch -n 2 -c \
    nvidia-smi --query-gpu=index,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory \
        --format=csv \
        -i 0,1,2,3,4,5,6,7