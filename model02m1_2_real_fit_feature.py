import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# 加载数据
data = np.load("data.npz", allow_pickle=True)
img_names = data["images_name"]
fits_names = data["fits_name"]
logAge = data["logAge"]
MH = data["MH"]
X_fits = data["fits_feature"]

# 定义 R^2 系数的计算函数
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# 加载模型
start = time.time()
model_B = load_model('model_1/fits_model_2para.h5', custom_objects={'coeff_determination': coeff_determination})
model_B_modified = tf.keras.models.Model(inputs=model_B.input, outputs=model_B.layers[-2].output)
fits_features = model_B_modified.predict(X_fits)
end = time.time()
t1 = end - start
print("fits_features[0].max:",fits_features[0].max())
print("fits_features[0].min:",fits_features[0].min())
# 将 fits_features 的每一行归一化到 [0, 1]
fits_features_normalized = (fits_features - fits_features.min(axis=1, keepdims=True)) / \
                           (fits_features.max(axis=1, keepdims=True) - fits_features.min(axis=1, keepdims=True))
                           
print("fits_features_normalized[0].max:",fits_features_normalized[0].max())
print("fits_features_normalized[0].min:",fits_features_normalized[0].min())
# 创建原始数据的 DataFrame
test_df = pd.DataFrame({
    "images_name": img_names,
    "fits_name": fits_names,
    "logAge": logAge,
    "MH": MH
})


## 将未归一化的 fits_features 添加到 DataFrame 中（光谱特征是需要经过归一化的，此处代码仅供参考）
#fits_features_df = pd.DataFrame(fits_features, columns=[f"feature_{i+1}" for i in range(fits_features.shape[1])])
#test_df = pd.concat([test_df, fits_features_df], axis=1)
## 保存原始数据到 CSV 文件
#test_df.to_csv("fits_features.csv", index=False)

# 创建归一化后的 DataFrame
normalized_df = pd.DataFrame({
    "images_name": img_names,
    "fits_name": fits_names,
    "logAge": logAge,
    "MH": MH
})
# 将归一化后的 fits_features 添加到 DataFrame 中
normalized_fits_features_df = pd.DataFrame(fits_features_normalized, columns=[f"feature_{i+1}" for i in range(fits_features.shape[1])])
normalized_df = pd.concat([normalized_df, normalized_fits_features_df], axis=1)

# 保存归一化后的数据到 CSV 文件
normalized_df.to_csv("fits_features_normalized_old.csv", index=False)

print(f"模型预测耗时: {t1:.2f} 秒")
print("数据已成功保存到'fits_features_normalized.csv'。")
