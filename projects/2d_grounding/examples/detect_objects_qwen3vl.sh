#!/bin/bash

# 设置环境变量
cd /root/Codes/LlamaFactory

source ~/.bashrc

conda activate llm_4080

export HF_ENDPOINT=https://hf-mirror.com

# 定义基础路径 (方便后续修改)
BASE_IMG_PATH="/root/Codes/data/SKU110K_fixed/images"
BASE_OUT_PATH="/root/Codes/data/SKU110K_fixed/detections"

# 确保输出目录存在 (可选，防止报错)
mkdir -p "$BASE_OUT_PATH"

# 循环 1 到 10
for i in {1..10}
do
    echo "正在处理: test_$i.jpg ..."

    python scripts/detect_objects_qwen3vl.py \
        --image_path "${BASE_IMG_PATH}/test_${i}.jpg" \
        --output_path "${BASE_OUT_PATH}/test_${i}_detected.jpg"
done

echo "所有任务处理完成！"