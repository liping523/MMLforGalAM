import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.regularizers import l1, l2, L1L2
from sklearn.model_selection import train_test_split
import time
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Add, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib
from keras import regularizers 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras import layers, models,optimizers, regularizers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
matplotlib.use('Agg')

band = 'irg_Band_42'
folder_name = f'model_2/{band}'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 已创建。")
else:
    print(f"文件夹 '{folder_name}' 已存在。")

data = np.load(f"{band}.npz", allow_pickle=True)
df = pd.read_csv('fits_features_normalized_old.csv')

img_names = data["images_name"]
fits_names = data["fits_name"]
logAge = data["logAge"]
MH = data["MH"]
X = data["images_feature"]
Y = df.iloc[:, 4:].values
print("Y.shape:",Y.shape)
print("Y[0].max:",Y[0].max())
print("Y[0].min:",Y[0].min())
#scaler = MinMaxScaler()
## 对 Y 的每一列进行归一化
#Y = scaler.fit_transform(Y)

# 定义R^2系数的计算函数
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

indices = np.arange(len(X))
x_train, x_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(
    X, Y, indices, test_size=0.2, random_state=2)

# 第二次分割：从临时集中分割出验证集和测试集（各占临时集的 50%）
x_val, x_test, y_val, y_test, val_indices, test_indices = train_test_split(
    x_temp, y_temp, temp_indices, test_size=0.5, random_state=2)

np.random.seed(42)
tf.random.set_seed(42)
def build_model(input_shape=(128, 128, 3), output_dim=1024):
    model = models.Sequential()
    
    # 第一层卷积层, kernel_regularizer=regularizers.l2(0.001)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第二层卷积层, kernel_regularizer=regularizers.l2(0.001)
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第三层卷积层, kernel_regularizer=regularizers.l2(0.001)
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第四层卷积层, kernel_regularizer=regularizers.l2(0.001)
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))  
    model.add(layers.Dense(output_dim, activation='linear'))
    
    return model

model = build_model()
# 编译模型
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[coeff_determination])
model.summary()

# 训练模型，使用X_train和y_train作为训练数据，进行100个epochs的训练，使用1/9的数据作为验证集，verbose=2表示打印训练过程。
start = time.time()
history = model.fit(
    x_train, y_train,
    epochs=100,  
    validation_data=(x_val, y_val),
    verbose=2)
    
end = time.time()

t1 = end - start
print(t1)

# 保存模型
model.save(folder_name + '/galaxy_image_to_spectrum_model.h5')
with open(folder_name + '/modelsummary_images_to_fits.txt', 'w') as f:
    model.summary(print_fn=lambda x:f.write(x+'\n'))
    
# Convert history.history to DataFrame 将训练历史数据转换为DataFrame并保存为CSV文件
history_df = pd.DataFrame(history.history)
history_df.to_csv(folder_name + '/history_2para.csv', index=False)

# 绘制并保存训练过程中的损失图表
val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(folder_name + '/loss.png')
plt.close()

#
model = tf.keras.models.load_model(folder_name + '/galaxy_image_to_spectrum_model.h5', custom_objects={'coeff_determination': coeff_determination})

start = time.time()
y_pred = model.predict(x_test)
end = time.time()
t2 = end - start

# 将所有批次的预测结果合并为一个数组
#y_pred_ = np.concatenate(predictions, axis=0)
y_pred[y_pred < 0] = 0
#y_pred = np.zeros_like(y_pred_)

## 对每一行进行归一化
#for i in range(y_pred_.shape[0]):
#    min_val = np.min(y_pred_[i])
#    max_val = np.max(y_pred_[i])
#    # 避免除以零的情况
#    if max_val - min_val > 0:
#        y_pred[i] = (y_pred_[i] - min_val) / (max_val - min_val)
#    else:
#        y_pred[i] = y_pred_[i] 
           
images_name = img_names[test_indices]
fits_name = fits_names[test_indices]
logAge = logAge[test_indices]
MH = MH[test_indices]

print("y_pred.shape:", y_pred.shape)

print("Y.max:",y_test[0].max())
print("Y.min:",y_test[0].min())
print("y_pred_.max:",y_pred[0].max())
print("y_pred_.min:",y_pred[0].min())
print("y_pred.max:",y_pred[0].max())
print("y_pred.min:",y_pred[0].min())


csv_file = folder_name + "/pre_fits_1024.csv"

# 检查文件是否存在
file_exists = os.path.isfile(csv_file)
header = ["images_names", "fits_names", "logAge", "MH"] + [f"{col}" for col in range(1, 1025)]
# 写入CSV文件
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range(len(images_name)):
        row1 = [images_name[i], fits_name[i], logAge[i], MH[i]]
        row1 += y_pred[i].tolist()
        writer.writerow(row1)

print(f"Data saved to {csv_file}")
# 计算文件行数和列数
def calculate_rows_and_columns(file_path, headers):
    # 计算行数
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)  # 读取所有行
        row_count = len(rows)
    
    # 列数为标题行的列数
    column_count = len(headers) if headers else len(rows[0]) if rows else 0
    
    return row_count, column_count

# 获取行数和列数
rows, columns = calculate_rows_and_columns(csv_file, header)

print(f"CSV文件共有 {rows} 行和 {columns} 列。")
Y = y_test

mse_per_spectrum = np.mean((y_pred - Y) ** 2, axis=1)  # 每个光谱的MSE
mse = np.mean(mse_per_spectrum)  # 所有光谱的平均MSE
print(f"MSE: {mse}")

rmse_per_spectrum = np.sqrt(np.mean((y_pred - Y) ** 2, axis=1))  # 每个光谱的RMSE
rmse = np.mean(rmse_per_spectrum)  # 所有光谱的平均RMSE
print(f"RMSE: {rmse}")

mae_per_spectrum = np.mean(np.abs(y_pred - Y), axis=1)  # 每个光谱的MAE
mae = np.mean(mae_per_spectrum)  # 所有光谱的平均MAE
print(f"MAE: {mae}")

ss_res_per_spectrum = np.sum((Y - y_pred) ** 2, axis=1)  # 每个光谱的残差平方和
ss_tot_per_spectrum = np.sum((Y - np.mean(Y, axis=1, keepdims=True)) ** 2, axis=1)  # 每个光谱的总平方和
r2_per_spectrum = 1 - (ss_res_per_spectrum / ss_tot_per_spectrum)  # 每个光谱的R²
r2 = np.mean(r2_per_spectrum)  # 所有光谱的平均R²
print(f"R²: {r2}")

dot_product = np.sum(Y * y_pred, axis=1)
norm_test = np.linalg.norm(Y, axis=1)
norm_pred = np.linalg.norm(y_pred, axis=1)
cos_theta = np.clip(dot_product / (norm_test * norm_pred), -1, 1)
sam_per_spectrum = np.arccos(cos_theta)  # 每个光谱的SAM
sam = np.mean(sam_per_spectrum)  # 所有光谱的平均SAM
print(f"SAM: {sam} (单位：弧度)")

mean_test = np.mean(Y, axis=1, keepdims=True)
mean_pred = np.mean(y_pred, axis=1, keepdims=True)
cov_per_spectrum = np.sum((Y - mean_test) * (y_pred - mean_pred), axis=1)
std_test = np.linalg.norm(Y - mean_test, axis=1)
std_pred = np.linalg.norm(y_pred - mean_pred, axis=1)
pcc_per_spectrum = cov_per_spectrum / (std_test * std_pred)  # 每个光谱的PCC
pcc = np.mean(pcc_per_spectrum)  # 所有光谱的平均PCC
print(f"PCC: {pcc}")

def cosine_similarity(true, pred):
    """
    计算真实值和预测值之间的余弦相似性
    :param true: 真实值数组 (numpy array)
    :param pred: 预测值数组 (numpy array)
    :return: 余弦相似性值 (float)
    """
    # 确保输入是NumPy数组
    true = np.array(true)
    pred = np.array(pred)
    
    # 计算点积
    dot_product = np.dot(true, pred)
    
    # 计算模长
    norm_true = np.linalg.norm(true)
    norm_pred = np.linalg.norm(pred)
    
    # 防止分母为零
    if norm_true == 0 or norm_pred == 0:
        return 0.0  # 如果模长为零，返回0（或根据需求处理）
    
    # 计算余弦相似性
    cosine_sim = dot_product / (norm_true * norm_pred)
    return cosine_sim

def average_cosine_similarity(true_matrix, pred_matrix):
    """
    计算真实值矩阵和预测值矩阵之间的平均余弦相似性
    :param true_matrix: 真实值矩阵 (numpy array, shape=(n, 1024))
    :param pred_matrix: 预测值矩阵 (numpy array, shape=(n, 1024))
    :return: 平均余弦相似性值 (float)
    """
    # 确保输入矩阵的形状一致
    if true_matrix.shape != pred_matrix.shape:
        raise ValueError("真实值矩阵和预测值矩阵的形状不一致。")
    
    n = true_matrix.shape[0]  # 获取数据量
    similarities = []
    
    # 遍历每一对真实值和预测值，计算余弦相似性
    for i in range(n):
        sim = cosine_similarity(true_matrix[i], pred_matrix[i])
        similarities.append(sim)
    
    # 计算平均余弦相似性
    avg_similarity = np.mean(similarities)
    return avg_similarity


# 计算平均余弦相似性
avg_similarity = average_cosine_similarity(y_test, y_pred)
print(f"平均余弦相似性: {avg_similarity:.4f}")

epsilon = 1  # 避免除以零
Y_safe = np.where(Y == 0, epsilon, Y)  # 将 Y 中的零值替换为 epsilon

# 计算相对误差
relative_errors = np.abs(y_pred - Y) / Y_safe
# 计算每个光谱的平均相对误差
average_relative_errors_per_spectrum = np.mean(relative_errors, axis=1)

# 计算总体平均相对误差
overall_average_residual = np.mean(average_relative_errors_per_spectrum)

print(f"Overall Average Relative Error: {overall_average_residual}")
differences = y_pred - Y
# 计算每个光谱的平均差值（对每个光谱的 1024 个数据点求平均）
average_differences_per_spectrum = np.mean(np.abs(differences), axis=1)

# 计算所有光谱的平均差值
overall_average_difference = np.mean(average_differences_per_spectrum)

print(f"Overall Average Difference: {overall_average_difference}")

with open(folder_name + '/评估指标ALL.txt', 'w', encoding='utf-8') as file:
    file.write(f'训练时间t1:{t1:.4f}\n')
    file.write(f'训练集数量:{len(x_train)}\n')
    file.write(f'预测时间t2:{t2:.4f}\n')
    file.write(f'测试集数量{len(X)}\n')
    file.write(f'MSE: {mse:.6f}\n')
    file.write(f'RMSE: {rmse:.6f}\n')
    file.write(f'MAE: {mae:.6f}\n')
    file.write(f'R²: {r2:.6f}\n')
    file.write(f'sam: {sam:.6f}\n')
    file.write(f'pcc: {pcc:.6f}\n')
    file.write(f'overall_average_residual: {overall_average_residual:.6f}\n')
    file.write(f'overall_average_difference: {overall_average_difference:.6f}\n')
    file.write(f'平均余弦相似性: {avg_similarity:.4f}')
    
import matplotlib.gridspec as gridspec

n_samples = Y.shape[0]  # 总样本数量
# 随机选择10个样本
random_indices = np.random.choice(n_samples, size=20, replace=False)

# 遍历随机选择的样本并绘制图像
for i in random_indices:
    cos_sim = cosine_similarity(y_test[i], y_pred[i])
    # 创建一个图形对象
    plt.figure(figsize=(10, 6))  # 调整整体图像大小

    # 使用 gridspec 定义子图布局
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0])  # 高度比例为 2:1

    # 上半部分：折线图
    ax1 = plt.subplot(gs[0])  # 占据前两行（高度比例为2）
    plt.plot(Y[i], label='Actual Spectra', color='blue')
    plt.plot(y_pred[i], label='Predicted Spectra', color='red', alpha=0.5)
    plt.legend(loc='upper right')
#    plt.title(f'Spectra Comparison for Sample {i}')
#    plt.xlabel('Wavelength Index')
    plt.text(0.05, 0.9, f"Cosine Similarity: {cos_sim:.4f}", 
             transform=ax1.transAxes, fontsize=12, color='black', 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.ylabel('Arbitrary flux')
    plt.grid(True)

    # 下半部分：残差图
    ax2 = plt.subplot(gs[1])  # 占据第三行（高度比例为1）
#    residuals = np.abs(y_pred[i] - Y[i]) / Y_safe[i]  # 计算残差
    residuals = y_pred[i] - Y[i]
    plt.plot(residuals, label='Residuals', color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # 添加零线
#    plt.ylim(-0.07, 0.07)
#    plt.legend(loc='upper right')
#    plt.ylim(0.5, 1.5)  # 设置 y 轴范围为 [0.5, 1.5]

#    plt.title(f'Residual Plot for Sample {i}')
#    plt.xlabel('Wavelength Index')

    plt.ylabel('Residuals')
    plt.grid(True)

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(f'{folder_name}/images_to_fits{i}.jpg', dpi=900, bbox_inches='tight')
    plt.close()
