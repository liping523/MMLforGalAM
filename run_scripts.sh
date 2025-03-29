#!/bin/bash

# 定义Python脚本文件列表
scripts=("model01m1_1_fits_pre.py" "model02m1_2_real_fit_feature.py" "model03m2_1_images_to_fits.py" "model04m3_1_images_pre.py" "model05m2_m3_feature.py" "model06m4_mul_pre.py")

# 遍历脚本列表并依次运行
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python "$script"  
    echo "$script completed."
    echo "-------------------"
done

echo "All scripts have been executed."
