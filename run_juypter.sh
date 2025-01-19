#!/bin/bash

# 笔记本路径列表
# notebooks=(
#     "./experiment_contrastive/train_clamer_contrastive.ipynb"
#     "./experiment_contrastive/train_ddc_contrastive.ipynb"
#     "./experiment_contrastive/train_GPT_contrastive.ipynb"
#     "./experiment_contrastive/train_charnn_contrastive.ipynb"
#     "./experiment_contrastive/train_cvae_contrastive.ipynb"
# )
notebooks=(
    "./test/1.ipynb"
    "./test/2.ipynb"
    "./test/3.ipynb"
    "./test/4.ipynb"
)

# 遍历每个笔记本
for notebook in "${notebooks[@]}"
do
    echo "Running notebook: $notebook"
    
    # 运行 Jupyter 笔记本，不保存输出文件
    jupyter nbconvert --to notebook --execute "$notebook" --stdout > /dev/null
    
    # 检查笔记本运行状态
    if [ $? -eq 0 ]; then
        echo "Notebook $notebook finished successfully."
    else
        echo "Notebook $notebook encountered an error."
    fi

    # # 执行 GPU 资源清理命令
    # echo "Cleaning up GPU processes..."
    # fuser -v /dev/nvidia* | awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

    # echo "Cleanup complete. Proceeding to next notebook..."
done

echo "All notebooks processed."