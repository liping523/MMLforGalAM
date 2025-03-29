import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
band = 'irg_Band_42'
# 加载数据
data = np.load(f"{band}.npz", allow_pickle=True)
img_names = data["images_name"]
fits_names = data["fits_name"]
logAge = data["logAge"]
MH = data["MH"]
images = data["images_feature"]
# 定义 R^2 系数的计算函数
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# 加载模型
start = time.time()
model = tf.keras.models.load_model(f'model_2/{band}/galaxy_image_to_spectrum_model.h5', custom_objects={'coeff_determination': coeff_determination})
simul_fits_features = model.predict(images)
end = time.time()
t1 = end - start
simul_fits_features[simul_fits_features < 0.0001] =0
print("t1:",t1)
print("simul_fits_features[0].max:",simul_fits_features[0].max())
print("simul_fits_features[0].min:",simul_fits_features[0].min())


model_A = load_model(f"model_3/{band}/galaxy_image_model.h5", custom_objects={'coeff_determination': coeff_determination})
model_A_modified = tf.keras.models.Model(inputs=model_A.input, outputs=model_A.layers[-2].output)
start = time.time()
images_feature = model_A_modified.predict(images)
end = time.time()
t2 = end - start
print("t2:",t2)

# 将 images_feature 的每一行归一化到 [0, 1]
images_feature_norm = (images_feature - images_feature.min(axis=1, keepdims=True)) / \
                           (images_feature.max(axis=1, keepdims=True) - images_feature.min(axis=1, keepdims=True))
                           
print("images_feature_norm[0].max:",images_feature_norm[0].max())
print("images_feature_norm[0].min:",images_feature_norm[0].min())


## 将所有信息保存为 .npz 文件（光谱特征不归一化）
##图像特征不归一化
#np.savez("feature_image_not_norm_linear.npz", 
#         images_name=img_names, 
#         fits_name=fits_names, 
#         logAge=logAge, 
#         MH=MH, 
#         fits_feature=simul_fits_features, 
#         images_feature=images_feature)

#print("数据已成功保存到 feature_image_not_norm_linear.npz 文件中。")

#图像特征归一化
np.savez(f"feature_{band}.npz", 
         images_name=img_names, 
         fits_name=fits_names, 
         logAge=logAge, 
         MH=MH, 
         fits_feature=simul_fits_features, 
         images_feature=images_feature_norm)

print(f"数据已成功保存到 feature_{band}.npz 文件中。")
'''
#保存特征为csv文件
# 创建原始数据的 DataFrame
test_df = pd.DataFrame({
    "images_name": img_names,
    "fits_name": fits_names,
    "logAge": logAge,
    "MH": MH
})

# 将 fits_features 添加到 DataFrame 中
simul_fits_features_df = pd.DataFrame(simul_fits_features, columns=[f"feature_{i+1}" for i in range(simul_fits_features.shape[1])])
test_df = pd.concat([test_df, simul_fits_features], axis=1)

# 保存原始数据到 CSV 文件
test_df.to_csv("simul_fits_features.csv", index=False)

# 创建归一化后的 DataFrame
normalized_df = pd.DataFrame({
    "images_name": img_names,
    "fits_name": fits_names,
    "logAge": logAge,
    "MH": MH
})

# 将归一化后的 fits_features 添加到 DataFrame 中
normalized_fits_features_df = pd.DataFrame(simul_fits_features_normalized, columns=[f"feature_{i+1}" for i in range(simul_fits_features.shape[1])])
normalized_df = pd.concat([normalized_df, normalized_fits_features_df], axis=1)

# 保存归一化后的数据到 CSV 文件
normalized_df.to_csv("simul_fits_features_normalized.csv", index=False)

print(f"模型预测耗时: {t1:.2f} 秒")
print("数据已成功保存到 'fits_features.csv' 和 'simul_fits_features_normalized.csv'。")
'''
