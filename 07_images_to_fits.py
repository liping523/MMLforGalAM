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

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
matplotlib.use('Agg')
folder_name = 'model_2_2/100_3'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 已创建。")
else:
    print(f"文件夹 '{folder_name}' 已存在。")

df = pd.read_csv("normalized_data_3W.csv")  #,nrows=30000)
#df = pd.read_csv('normalized_data_1453.csv')
images = []
img_names = []
# fits_names = []
images_names = df.iloc[:, 0].values
fits_names = df.iloc[:, 1].values
logages = df.iloc[:, -2].values
MH = df.iloc[:, -1].values

for index, row in df.iterrows():
    img_name = row["photo_name"]
    #img_name = row["images_names"]
    img_names.append(img_name)
    # fits_name = row["fits_names"]
    # fits_names.append(fits_name)
    img = load_img(f'0_images_crop/{row["photo_name"]}', target_size=(128, 128))
    #img = load_img(f'images_crop/{row["images_names"]}', target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    images.append(img_array)


X_fits = df.iloc[:, 2:4173].values

# 定义R^2系数的计算函数
def coeff_determination(y_true, y_pred):
    """
    The coefficient of determination R^2 is often used in linear regression to represent the percentage of the dependent variable's variance explained by the regression line. If R^2 = 1, it indicates that the model perfectly predicts the target variable.
    Formula: R^2 = SSR/SST = 1 - SSE/SST
    Where: SST (total sum of squares) is the total sum of squares, SSR (regression sum of squares) is the regression sum of squares, and SSE (error sum of squares) is the residual sum of squares.
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

np.random.seed(42)
tf.random.set_seed(42)


model_B = load_model('model_1/fits_model_2para.h5', custom_objects={'coeff_determination': coeff_determination})
model_B_modified = tf.keras.models.Model(inputs=model_B.input, outputs=model_B.layers[-2].output)
fits_features = model_B_modified.predict(X_fits.reshape(-1, X_fits.shape[1], 1))
#print("模型B的倒数第二层的前五个输出:", fits_features.flatten()[:5])

X = np.array(images)
print("X.shape:", X.shape)
print("X.max:",X[0].max())
print("X.min:",X[0].min())

# 初始化存储最小值和最大值的数组
min_vals = np.zeros(fits_features.shape[0])
max_vals = np.zeros(fits_features.shape[0])

Y = np.zeros_like(fits_features)

for i in range(fits_features.shape[0]):
    min_val = np.min(fits_features[i])
    max_val = np.max(fits_features[i])
    
    min_vals[i] = min_val
    max_vals[i] = max_val

    if max_val == min_val:
        Y[i] = np.zeros_like(fits_features[i])  # 防止除0错误
    else:
        Y[i] = (fits_features[i] - min_val) / (max_val - min_val)

print("Y.shape:", Y.shape)
print("Y.max:", Y[0].max())
print("Y.min:", Y[0].min())
print("fits_features[0][:5]:", fits_features[0][:5])
print("Y[0][:5]:", Y[0][:5])



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
'''
# ##########################  new:best   #########################
def build_model(input_shape=(128, 128, 3),output_dim=1024):  #1024
    model = models.Sequential()
    
    # 第一层卷积层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))   # , padding='same'
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第二层卷积层
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第三层卷积层
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第四层卷积层
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 平坦化
    model.add(layers.Flatten())
    
    # 全连接层
    model.add(layers.Dense(1024, activation='relu'))  #512
    
    # 输出层
    model.add(layers.Dense(output_dim))
    
    return model

'''
''' 111
def build_model(input_shape=(128, 128, 3), output_dim=2048):
    model = models.Sequential()
    
    # 第一层卷积层，使用padding='same'保持尺寸不变
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape,kernel_regularizer=l2(0.001)))
    # 池化层，步长和窗口大小相同，图像尺寸减半
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # 第二层卷积层，使用padding='same'
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    # 池化层，步长和窗口大小相同，图像尺寸减半
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # 第三层卷积层，使用padding='same'
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.001)))
    # 池化层，步长和窗口大小相同，图像尺寸减半
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # 第四层卷积层，使用padding='same'
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.001)))
    
    # 池化层，步长和窗口大小相同，图像尺寸减半
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # 平坦化
    model.add(layers.Flatten())
    
    # 全连接层
    model.add(layers.Dense(2048, activation='relu'))  # 512
    
    # 输出层
    model.add(layers.Dense(output_dim,activation='linear'))
    
    return model
    
# 构建模型
input_shape = (128, 128, 3)
output_dim = 2048
model = build_model(input_shape, output_dim)
# 编译模型
#model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[coeff_determination])
model.summary()
'''

model = Sequential()

# 卷积层, padding='same'
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten层将卷积层的输出展平为一维
model.add(Flatten())

# 全连接层
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

# 输出层，输出为1024维的模拟光谱特征
model.add(Dense(1024, activation='linear'))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 模型结构
model.summary()

# 训练模型，使用X_train和y_train作为训练数据，进行100个epochs的训练，使用1/9的数据作为验证集，verbose=2表示打印训练过程。
start = time.time()
history = model.fit(x_train, y_train, epochs=100, validation_split=1/9, verbose=2)
end = time.time()
# 保存模型
model.save(folder_name + '/galaxy_image_to_spectrum_model_new.h5')

# Convert history.history to DataFrame 将训练历史数据转换为DataFrame并保存为CSV文件
history_df = pd.DataFrame(history.history)
history_df.to_csv(folder_name + '/history_2para.csv', index=False)


'''
##分批次训练
batch_size = 16  # 根据显存大小调整
epochs = 30
validation_split = 1/9

# 计算训练和验证集的大小
val_size = int(x_train.shape[0] * validation_split)
train_size = x_train.shape[0] - val_size

x_val = x_train[-val_size:]
y_val = y_train[-val_size:]

x_train = x_train[:train_size]
y_train = y_train[:train_size]

num_batches = train_size // batch_size

# 存储每个 epoch 的损失和验证损失
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    epoch_train_loss = []
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]
        
        # 对每个批次进行训练
        loss = model.train_on_batch(x_batch, y_batch)
        epoch_train_loss.append(loss)
    
    # 如果数据不能整除batch_size，处理剩余部分
    if train_size % batch_size != 0:
        x_batch = x_train[num_batches * batch_size:]
        y_batch = y_train[num_batches * batch_size:]
        loss = model.train_on_batch(x_batch, y_batch)
        epoch_train_loss.append(loss)
    
    # 计算当前 epoch 的平均训练损失
    train_loss = np.mean(epoch_train_loss)
    train_loss_history.append(train_loss)
    
    # 在验证集上进行验证
    val_loss = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
    val_loss_history.append(val_loss)
    
    print(f'Train loss: {train_loss} - Val loss: {val_loss}')
end = time.time()
# 转换为 DataFrame 并保存到 CSV
history_df = pd.DataFrame({
    'epoch': np.arange(1, epochs + 1),
    'train_loss': train_loss_history,
    'val_loss': val_loss_history
})

history_df.to_csv(folder_name + '/history_2para.csv', index=False)
'''

# 绘制并保存训练过程中的损失图表
val_loss = history.history['val_loss']
loss = history.history['loss']
# # 从CSV文件中读取进行图表的绘制
#val_loss = history_df['val_loss']
#loss = history_df['loss']

plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(folder_name + '/loss.png')
plt.close()

with open(folder_name + '/modelsummary_images_to_fits1.txt', 'w') as f:
    model.summary(print_fn=lambda x:f.write(x+'\n'))

model = tf.keras.models.load_model(folder_name + "/galaxy_image_to_spectrum_model_new.h5", custom_objects={'coeff_determination': coeff_determination})

#y_pred_ = model.predict_on_batch(X)

#y_pred_ = model.predict(X)
batch_size = 16  
print("X.shape[0]:", X.shape[0])
num_batches = X.shape[0] // batch_size

y_pred_ = []

for i in range(num_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X[start:end]
    
    # 对每个批次进行预测
    y_pred_batch = model.predict_on_batch(X_batch)
    
    y_pred_.append(y_pred_batch)

# 如果数据不能整除batch_size，处理剩余部分
if X.shape[0] % batch_size != 0:
    X_batch = X[num_batches * batch_size:]
    y_pred_batch = model.predict_on_batch(X_batch)
    y_pred_.append(y_pred_batch)

# 合并所有批次的预测结果
y_pred_ = np.concatenate(y_pred_, axis=0)
y_pred_[y_pred_ < 0] = 0

print("y_pred_.max:",y_pred_[10].max())
print("y_pred_.min:",y_pred_[10].min())

y_pred = np.zeros_like(y_pred_)

for i in range(y_pred_.shape[0]):
    min_val = np.min(y_pred_[i])
    max_val = np.max(y_pred_[i])
    y_pred[i] = (y_pred_[i] - min_val) / (max_val - min_val)
    

# y_pred_ = model.predict(x_test)

print("y_pred.shape:", y_pred.shape)
#print("x_test_name.shape:", x_test_name.shape)
#print("y_test_name.shape:", y_test_name.shape)
print("logages.shape:", logages.shape)
print("MH.shape:", MH.shape)
print("y_test.max:",Y[10].max())
print("y_test.min:",Y[10].min())
print("y_pred_.max:",y_pred_[10].max())
print("y_pred_.min:",y_pred_[10].min())
print("y_pred.max:",y_pred[10].max())
print("y_pred.min:",y_pred[10].min())


csv_file = folder_name + "/pre_fits_1024.csv"

# 检查文件是否存在
file_exists = os.path.isfile(csv_file)
header = ["images_names", "fits_names"] + [f"{col}" for col in range(1, 1025)] + ["logAge", "MH"]
# 写入CSV文件
with open(csv_file, mode='w', newline='') as file:
    #if not file_exists:
        #writer = csv.writer(file)
        #writer.writerow(header)
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range(len(images_names)):
        row1 = [images_names[i], fits_names[i]]
        row1 += y_pred[i].tolist()
        row1.append(logages[i]) 
        row1.append(MH[i])
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

def evaluate_mse(y_true, y_pred):
    # 计算每个样本的 MSE
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=1)
    # 对所有样本的 MSE 求平均值
    mse = np.mean(mse_per_sample)
    return mse
def evaluate_rmse(y_true, y_pred):
    # 计算每个样本的 MSE
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=1)
    # 计算每个样本的 RMSE
    rmse_per_sample = np.sqrt(mse_per_sample)
    # 对所有样本的 RMSE 求平均值
    rmse = np.mean(rmse_per_sample)
    return rmse

def evaluate_mae(y_true, y_pred):
    # 计算每个样本的 MAE
    mae_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)
    # 对所有样本的 MAE 求平均值
    mae = np.mean(mae_per_sample)
    return mae


def evaluate_r2(y_true, y_pred):
    # 初始化一个列表来存储每个样本的 R²
    r2_per_sample = []
    for i in range(y_true.shape[0]):
        r2 = r2_score(y_true[i], y_pred[i])
        r2_per_sample.append(r2)
    # 对所有样本的 R² 求平均值
    r2 = np.mean(r2_per_sample)
    return r2

mse = evaluate_mse(Y, y_pred)
rmse = evaluate_rmse(Y, y_pred)
mae = evaluate_mae(Y, y_pred)
r2 = evaluate_r2(Y, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
t = end - start
print('Running time: %s Seconds' % (end - start))
# 打开文件以写入模式，如果文件不存在会自动创建
with open(folder_name + '/评估指标.txt', 'w', encoding='utf-8') as file:
    file.write(f'Running time: {t} Seconds\n')
    file.write(f'MSE: {mse:.6f}\n')
    file.write(f'RMSE: {rmse:.6f}\n')
    file.write(f'MAE: {mae:.6f}\n')
    file.write(f'R²: {r2:.6f}\n')

i = 289
plt.figure(figsize=(10, 4))
plt.plot(Y[i], label='Actual Spectra', color='blue')
plt.plot(y_pred[i], label='Predicted Spectra', color='red', alpha=0.5)
# plt.title(f'Actual vs Predicted Spectra for Image ID: {i}')
plt.legend()
# 调整布局
plt.tight_layout()
plt.savefig(f'{folder_name}/images_to_fits{i}1.jpg', dpi=900, bbox_inches='tight')
i = 289
plt.figure(figsize=(10, 4))
plt.plot(Y[i], label='Actual Spectra', color='blue')
plt.plot(y_pred_[i], label='Predicted Spectra', color='red', alpha=0.5)
# plt.title(f'Actual vs Predicted Spectra for Image ID: {i}')
plt.legend()
# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig(f'{folder_name}/images_to_fits{i}.jpg', dpi=900, bbox_inches='tight')
i = 239
plt.figure(figsize=(10, 4))
plt.plot(Y[i], label='Actual Spectra', color='blue')
plt.plot(y_pred[i], label='Predicted Spectra', color='red', alpha=0.5)
# plt.title(f'Actual vs Predicted Spectra for Image ID: {i}')
plt.legend()
# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig(f'{folder_name}/images_to_fits{i}.jpg', dpi=900, bbox_inches='tight')
i = 602
plt.figure(figsize=(10, 4))
plt.plot(Y[i], label='Actual Spectra', color='blue')
plt.plot(y_pred[i], label='Predicted Spectra', color='red', alpha=0.5)
# plt.title(f'Actual vs Predicted Spectra for Image ID: {i}')
plt.legend()
# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig(f'{folder_name}/images_to_fits{i}.jpg', dpi=900, bbox_inches='tight')

