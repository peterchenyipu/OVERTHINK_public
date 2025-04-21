
allocate 2 gpus, interactive session
```bash
salloc -N1 -t0:30:00 --cpus-per-task 8 --ntasks-per-node=1 --gres=gpu:H100:2 --mem-per-gpu=80G
```

in one tmux window, run the server
```bash
source 
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --enable-reasoning --reasoning-parser deepseek_r1 \
    --tensor-parallel-size 2
# might need to set the --max-model-len, not sure
```

in another tmux window, run main.sh (modify the model to be deepseek-vllm)