import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 读取 CSV 文件
file_path = 'pre_data_norm.csv'
df = pd.read_csv(file_path)

# 提取图像名和光谱特征
img_names = df['image_name'].values 
fits_names = df['fits_name'].values
logAge = df['logAge'].values 
MH = df['MH'].values
spectral_features = df.iloc[:, 4:].values  #4171维的光谱数据

# 初始化图像数组列表
images = []

# 遍历 CSV 文件的每一行，加载图像并处理
for index, row in df.iterrows():
    img_name = row['image_name']  
    img_path = f'images/{img_name}'  
    img = load_img(img_path, target_size=(128, 128))  
    img_array = img_to_array(img)  
    img_array /= 255.0  
    images.append(img_array) 

# 将图像列表转换为 NumPy 数组
images = np.array(images)

print("images.shape",images.shape)
print("images.max:",images[0].max())
print("images.min:",images[0].min())

# 将所有信息保存为 .npz 文件
np.savez("data_2000.npz", 
         images_name=img_names, 
         fits_name=fits_names, 
         logAge=logAge, 
         MH=MH, 
         fits_feature=spectral_features, 
         images_feature=images)

print("数据已成功保存到 data.npz 文件中。")

'''
#检查npz文件内容
import numpy as np
data = np.load("data_2000.npz", allow_pickle=True)

# 提取数据
img_names = data["images_name"]
fits_names = data["fits_name"]
logAge = data["logAge"]
MH = data["MH"]
spectral_features = data["fits_feature"]
image_features = data["images_feature"]

# 打印变量的形状、最大值和最小值
print("img_names:")
print(f"  Shape: {img_names.shape}")
print(f"  First 10 entries: {img_names[:10]}")
print()

print("fits_names:")
print(f"  Shape: {fits_names.shape}")
print(f"  First 10 entries: {fits_names[:10]}")
print()

print("logAge:")
print(f"  Shape: {logAge.shape}")
print(f"  Min: {np.min(logAge)}, Max: {np.max(logAge)}")
print(f"  First 10 entries: {logAge[:10]}")
print()

print("MH:")
print(f"  Shape: {MH.shape}")
print(f"  Min: {np.min(MH)}, Max: {np.max(MH)}")
print(f"  First 10 entries: {MH[:10]}")
print()

print("spectral_features:")
print(f"  Shape: {spectral_features.shape}")
print(f"  Min: {np.min(spectral_features)}, Max: {np.max(spectral_features)}")
print(f"  First 10 entries of the first spectrum: {spectral_features[0, :10]}")
print()

print("image_features:")
print(f"  Shape: {image_features.shape}")
print(f"  Min: {np.min(image_features)}, Max: {np.max(image_features)}")
print(f"  First 10 entries of the first image (flattened): {image_features[0, 54:64]}")
print()
'''
'''

import pandas as pd

# 读取CSV文件
file_path = '../../10w/LP/data_normalized.csv'
df = pd.read_csv(file_path)
spectral_features = df.iloc[:,4:4175].values
print("fit_feature.shape",spectral_features.shape)
print("fit_feature.max:",spectral_features[0].max())
print("fit_feature.min:",spectral_features[0].min())
# 获取行数和列数
num_rows, num_columns = df.shape
print(f"文件 {file_path} 的总行数: {num_rows}")
print(f"文件 {file_path} 的总列数: {num_columns}")
print(df.columns)
y = df.iloc[:,[4173,4174]].values
print("y.shape",y.shape)
print("y.max:",y[:,0].max())
print("y.min:",y[:,0].min())
print("y.max:",y[:,1].max())
print("y.min:",y[:,1].min())
# 检查前四列是否有空值
for col in df.columns[:2]:  # 只检查前四列
    null_count = df[col].isnull().sum()
    if null_count > 0:
        print(f"列 '{col}' 中为空的数量: {null_count}")
    else:
        print(f"列 '{col}' 中没有空值。")
'''
