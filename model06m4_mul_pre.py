import time
import os
import gc
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from matplotlib import colors
from tensorflow.keras.layers import MultiHeadAttention, Layer, LayerNormalization
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.linear_model import LinearRegression
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
matplotlib.use('Agg')


band = 'irg_Band_42'
folder_name = f'model_4/{band}'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 已创建。")
else:
    print(f"文件夹 '{folder_name}' 已存在。")
    
data = np.load(f"feature_{band}.npz", allow_pickle=True)
#df = pd.read_csv('fits_features_normalized.csv')
# 提取数据
img_names = data["images_name"]
fits_names = data["fits_name"]
logAge = data["logAge"]
MH = data["MH"]
X_fits = data["fits_feature"]
#X_fits = df.iloc[:, 4:].values
#X_fits = np.array(X_fits, dtype='float32')
X_images = data["images_feature"]
print("X_fits.shape:", X_fits.shape)
print("X_fits[23].max:",X_fits[23].max())
print("X_fits[23].min:",X_fits[23].min())
print("X_images[23].max:",X_images[23].max())
print("X_images[23].min:",X_images[23].min())

 
Y = np.column_stack((logAge, MH))
print("Y[0].min():",Y[:,0].min())
print("Y[0].max():",Y[:,0].max())
print("Y[1].min():",Y[:,1].min())
print("Y[1].max():",Y[:,1].max())
print("Y.shape:",Y.shape)

#scaler = MinMaxScaler()
#Y = scaler.fit_transform(Y)
min_vals = np.min(Y, axis=0)
max_vals = np.max(Y, axis=0)

Y = (Y - min_vals) / (max_vals - min_vals)

print("Y[0].min():",Y[:,0].min())
print("Y[0].max():",Y[:,0].max())
print("Y[1].min():",Y[:,1].min())
print("Y[1].max():",Y[:,1].max())
 
# 定义R^2系数的计算函数
def coeff_determination(y_true, y_pred):
    SS_res =  tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# 定义异常值移除函数
def remove_abnor(x, y):
    result_two = y - x
    σ = 3
    std_two = np.std(abs(result_two))
    cc = np.where((result_two < result_two.mean() + σ * std_two) & (result_two > result_two.mean() - (σ * std_two)))
    x = x[cc]
    y = y[cc]
    return x, y

def plot_hexbin(x, y, savefigdir, savefigname1, savefigname2):
    plt.close()
    plt.hexbin(x, y, gridsize=25, mincnt=0, vmax=25, cmap=plt.cm.gray_r)
    plt.colorbar()
    min1 = x.min()
    min2 = y.min()
    max1 = x.max()
    max2 = y.max()
    min_num = min(min1, min2)
    max_num = max(max1, max2)
    plt.xlim((min_num, max_num))
    plt.ylim((min_num, max_num))
    mu = (y - x).mean()
    sigma = (y - x).std()
    N = len(x)
    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu,),
        r'$\sigma=%.5f$' % (sigma,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(min_num+0.1, max_num-0.1, textstr, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel(" True  " + savefigname1)
    plt.ylabel("Predicted  " + savefigname1)
    plt.plot((min_num, max_num), (min_num, max_num), "b--", linewidth=3)
    plt.savefig(savefigdir + '/' + savefigname2 + '.png', dpi=600)
    plt.close()
    
def plot_combined_chart(x, y, savefigname,image_name):
    
    plt.rc('axes', titlesize=18, labelsize=12)  # 坐标轴标题字号
    plt.rc('xtick', labelsize=16)  # x轴刻度字号
    plt.rc('ytick', labelsize=16)  # y轴刻度字号
    plt.rc('axes', linewidth=1.5)  # 坐标轴加粗
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams.update({'font.size':16})
    fig = plt.figure(figsize=(5, 7))
    min1 = x.min()
    min2 = y.min()
    max1 = x.max()
    max2 = y.max()
    min_num = min(min1, min2)
    max_num = max(max1, max2)
    # 创建一个网格布局，定义两行的宽度比为2:1
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # 第一行的子图（占据较大宽度）
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 上方的散点图
    xy = np.vstack([x, y])  # 将两个维度的数据进行叠加
    kenal = gaussian_kde(xy)  # 建立概率密度分布
    z = kenal.evaluate(xy)  # 得到每个样本点的概率密度
    idx = z.argsort()  # 对概率密度进行排序，获取密集点
   
    # 使用Normalize来设置颜色柱的范围
    norm = colors.Normalize(vmin=z.min(), vmax=(z.max()-z.max()/4))#
    
    # 绘制散点图并添加密集区的边缘直线
    #scatter = ax1.scatter(x, y, marker='o', c=z, edgecolors='none', s=50, cmap='coolwarm', norm=norm)
    #scatter = ax1.scatter(x, y, marker='o', c=z, edgecolors='none', s=50, cmap='plasma', norm=norm)  # 修改颜色映射为 'plasma'
    #scatter = ax1.scatter(x, y, marker='o', c=z, edgecolors='none', s=50, cmap='viridis', norm=norm)
    scatter = ax1.scatter(x, y, marker='o', c=z, edgecolors='none', s=50, cmap='Spectral_r', norm=norm)
    cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    
    mu = (y - x).mean()
    sigma = (y - x).std()
    N = len(x)
    textstr = '\n'.join((
        r'Bias=%.4f' % (mu,),
        r'SD=%.4f' % (sigma,)))
    props = dict(boxstyle='round',  facecolor='none', edgecolor='black')
    ax1.text(min_num+0.1, max_num-0.1, textstr, fontsize=12, verticalalignment='top', bbox=props)

    ax1.set_xlabel('True ' + savefigname, fontsize=14)
    ax1.set_ylabel('Predicted ' + savefigname, fontsize=14)
    #ax1.set_title(f'Scatter plot of True data and Model Estimated_{savefigname}')
    ax1.tick_params(axis='both', which='major', width=1.5, labelsize=12)
    ax1.set_xlim((min_num, max_num))
    ax1.set_ylim((min_num, max_num))
    
    ax1.plot((min_num, max_num), (min_num, max_num), color='r', linewidth=2, linestyle='--')

    
    # 第二行的子图（占据较小宽度）
    ax2 = fig.add_subplot(gs[1, 0])
    res_0 = y - x  # 计算残差
    bins = np.arange(res_0.min()-0.2, res_0.max()+0.2, 0.1)
    ax2.hist(res_0, bins)
    ax2.set_xlabel(r'$\Delta$'+savefigname, fontsize=14)
    ax2.set_ylabel('Number', fontsize=14)
    #ax2.set_title(f'Histogram of Residuals_{savefigname}')
    ax2.xaxis.label.set_size(14)
    ax2.yaxis.label.set_size(14)
    ax2.tick_params(axis='both', which='major', width=1.5, labelsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(folder_name, image_name)
    # 保存图像
    plt.savefig(f'{save_path}.jpg', dpi=900, bbox_inches='tight')
    plt.close()

'''        
class AttentionFusion(Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.1, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

        # Linear layers to project inputs to key_dim dimensions
        self.query_dense = Dense(key_dim)
        self.key_dense = Dense(key_dim)
        self.value_dense = Dense(key_dim)

        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        # Fully connected layer to project back to original dimension
        self.dense = Dense(key_dim)

        # Layer normalization
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        # Dropout to prevent overfitting
        self.dropout = Dropout(dropout_rate)

    def call(self, image_features, spectral_features):
        # Ensure input is 3D
        if len(image_features.shape) == 2:
            image_features = tf.expand_dims(image_features, axis=1)
        if len(spectral_features.shape) == 2:
            spectral_features = tf.expand_dims(spectral_features, axis=1)

        # Project features to shared subspace
        query = self.query_dense(spectral_features)
        key = self.key_dense(image_features)
        value = self.value_dense(image_features)

        # Multi-head attention
        attention_output = self.attention(query=query, key=key, value=value)

        # Residual connection and layer normalization
        residual_output = self.layer_norm(spectral_features + attention_output)

        # Apply dropout
        fused_features = self.dropout(residual_output)

        # Remove seq_len dimension if it exists
        fused_features = tf.squeeze(fused_features, axis=1)

        # Final dense layer to ensure output dimension matches
        output_features = self.dense(fused_features)
        return output_features
def batch_feature_fusion(X_images, X_fits, batch_size=32, attention_params={"num_heads": 8, "key_dim": 1024}):
    """
    分批进行特征融合。

    Args:
        X_images (np.ndarray): Images 特征.
        X_fits (np.ndarray): Fits 特征.
        batch_size (int): 批次大小.
        attention_params (dict): AttentionFusion 层的参数.

    Returns:
        np.ndarray: 融合后的特征.
    """
    num_samples = X_images.shape[0]
    attention_fusion_layer = AttentionFusion(**attention_params)
    fused_features = []

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_X_images = X_images[i:i+batch_size]
        batch_X_fits = X_fits[i:i+batch_size]

        # 特征融合
        data = attention_fusion_layer(batch_X_images, batch_X_fits)
        fused_features.append(data)

    # 将分批融合的特征拼接在一起
    fused_features = np.concatenate(fused_features, axis=0)
    return fused_features
    
#data = batch_feature_fusion(X_images, X_fits, batch_size=1024)

attention_fusion_layer = AttentionFusion(num_heads=8, key_dim=1024)
data = attention_fusion_layer(X_images, X_fits)
data = np.array(data)
print("data.shape:",data.shape)
'''
class AttentionFusion(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(AttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.norm_image = LayerNormalization()
        self.norm_spectral = LayerNormalization()
        self.dropout = Dropout(0.1)

    def process_attention(self, spectral_features, image_features):
        # 扩展维度以匹配多头注意力的期望输入
        spectral_features = tf.expand_dims(spectral_features, axis=1)
        image_features = tf.expand_dims(image_features, axis=1)

        # 使用光谱特征作为查询，图像特征作为键和值
        attention_output = self.multi_head_attention(
            query=spectral_features,  # 光谱特征作为查询
            key=image_features,       # 图像特征作为键
            value=image_features      # 图像特征作为值
        )

        # 残差连接和归一化
        attention_output = self.norm_spectral(attention_output + spectral_features)
        return attention_output

    def call(self, image_features, spectral_features):
        # 使用光谱特征作为查询，进行跨模态注意力
        attention_output = self.process_attention(spectral_features, image_features)
        
        # 融合后的特征
        fused_features = tf.squeeze(attention_output, axis=1)
        
        # 加入dropout
        fused_features = self.dropout(fused_features)

        return fused_features

        

# 定义分批处理的调用  分批处理，没有中间结果保存
def process_with_batching(image_features, spectral_features, batch_size=2048):
    # 实例化 AttentionFusion 层
    fusion_layer = AttentionFusion(num_heads=8, key_dim=1024)

    # 将数据集分批处理
    dataset = tf.data.Dataset.from_tensor_slices((image_features, spectral_features)).batch(batch_size)

    # 初始化一个空列表存储每个批次的输出
    fused_features_list = []

    # 对每个批次进行处理
    for batch_image_features, batch_spectral_features in dataset:
        batch_fused_features = fusion_layer(batch_image_features, batch_spectral_features)
        fused_features_list.append(batch_fused_features)

    # 将所有批次的输出拼接到一起
    fused_features = tf.concat(fused_features_list, axis=0)

    return fused_features

# 调用分批处理，设定批次大小为 2048
data = process_with_batching(X_images, X_fits, batch_size=2048)
data = np.array(data)
# 输出结果形状
print(f"Fused features shape (with batching): {data.shape}")

def min_max_normalize(array):
    min_vals = array.min(axis=1, keepdims=True)
    max_vals = array.max(axis=1, keepdims=True)
    normalized_array = (array - min_vals) / (max_vals - min_vals)
    return normalized_array

X = min_max_normalize(data)



indices = np.arange(len(X))
x_train, x_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(
    X, Y, indices, test_size=0.2, random_state=2)

x_val, x_test, y_val, y_test, val_indices, test_indices = train_test_split(
    x_temp, y_temp, temp_indices, test_size=0.5, random_state=2)
'''    
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(1024,), kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='linear'))
#optimizer = optimizers.Adam(learning_rate=0.0001, clipnorm=1.0) 
#model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[coeff_determination]) 

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[coeff_determination]) 

#delta = 1.0  # 你可以根据数据调整这个值
#huber_loss = Huber(delta=delta)  # 定义 Huber 损失函数
#model.compile(optimizer=Adam(learning_rate=0.0001), loss=huber_loss, metrics=['mae'])
model.summary()
'''
np.random.seed(42)
tf.random.set_seed(42)
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(1024,)))
#model.add(Dropout(0.3)) 
model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(2, activation='linear'))
optimizer = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[coeff_determination])
model.summary()
history = []
start = time.time()
history = model.fit(
    x_train, y_train,
    epochs=100,  # 训练100个epoch
    validation_data=(x_val, y_val),
    verbose=2 
)

end = time.time()
t1 = end - start
model.save(folder_name + '/galaxy_mul_model.h5')

history_df = pd.DataFrame(history.history)
history_df.to_csv(folder_name + '/history_2para.csv', index=False)


val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.ylim(top=0.06)
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(folder_name + '/loss.pdf')
plt.close()
print('Running time: %s Seconds' % (end - start))
with open(folder_name + '/modelsummary_mul.txt', 'w') as f:
    model.summary(print_fn=lambda x:f.write(x+'\n'))

model = tf.keras.models.load_model(folder_name + '/galaxy_mul_model.h5', custom_objects={'coeff_determination': coeff_determination})
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
t2 = end - start

print("反归一化前:")
print("y_test[:,0].max:",y_test[:,0].max())
print("y_test[:,0].min:",y_test[:,0].min())
print("y_test[:,1].max:",y_test[:,1].max())
print("y_test[:,1].min:",y_test[:,1].min())
print("y_pred_[:,0].max:",y_pred[:,0].max())
print("y_pred_[:,0].min:",y_pred[:,0].min())
print("y_pred_[:,1].max:",y_pred[:,1].max())
print("y_pred_[:,1].min:",y_pred[:,1].min())
'''
# 反归一化
y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)
'''


y_test = y_test * (max_vals - min_vals) + min_vals
y_pred = y_pred * (max_vals - min_vals) + min_vals
print("反归一化:")

print("y_test[:,0].max:",y_test[:,0].max())
print("y_test[:,0].min:",y_test[:,0].min())
print("y_test[:,1].max:",y_test[:,1].max())
print("y_test[:,1].min:",y_test[:,1].min())
print("y_pred_[:,0].max:",y_pred[:,0].max())
print("y_pred_[:,0].min:",y_pred[:,0].min())
print("y_pred_[:,1].max:",y_pred[:,1].max())
print("y_pred_[:,1].min:",y_pred[:,1].min())


test_image_names = img_names[test_indices]
test_fits_names = fits_names[test_indices]
test_df = pd.DataFrame({
    "images_name": test_image_names,
    "fits_name": test_fits_names,
    "true_logAge": y_test[:, 0],
    "pred_logAge": y_pred[:, 0],
    "true_MH": y_test[:, 1],
    "pred_MH": y_pred[:, 1],
})

test_df.to_csv(folder_name + "/True_Predict_2para.csv", index=False)

mse_values = []
mae_values = []
r2_values = []
#mape_values = []
sd_values = []
mu_values = []


para1 = ['logAge', '[M/H]']
para2 = ['logAge', 'mh']
for i in range(2):
    x = y_test[:, i]
    y = y_pred[:, i]
    x = x.flatten()
    y = y.flatten()
    print(para1[i], "before:", len(x), len(y))
#    x_ = denormalize(x, mmin[i], mmax[i])
#    y_ = denormalize(y, mmin[i], mmax[i])
    x_,y_ = remove_abnor(x,y)  #去掉异常值
    print(para1[i], "after:", len(x_), len(y_))
    plot_hexbin(x_, y_, folder_name, para2[i], f'去掉异常值_{para2[i]}')
    plot_hexbin(x, y, folder_name, para2[i], f'原始_{para2[i]}')
    plot_combined_chart(x_, y_, para2[i], f'去掉异常值_{para2[i]}')
    plot_combined_chart(x, y, para2[i], f'原始_{para2[i]}')
#    print("x.shape",x.shape)
#    print("y.shape",y.shape)
    
    mse_values.append(mean_squared_error(x_, y_))
    mse_values.append(mean_squared_error(x, y))
    mae_values.append(mean_absolute_error(x_, y_))
    mae_values.append(mean_absolute_error(x, y))
    r2_values.append(r2_score(x_, y_))
    r2_values.append(r2_score(x, y))
#    mape_values.append(mean_absolute_percentage_error(x_, y_))
#    mape_values.append(mean_absolute_percentage_error(x, y))
    sd_values.append(np.std(x_ - y_))
    sd_values.append(np.std(x - y))
    mu_values.append(np.mean(x_ - y_))
    mu_values.append(np.mean(x - y))
# 打印每个输出的指标
for i in range(4):
    print(f'Output {i + 1}:')
    print(f'MSE: {mse_values[i]:.4f}')
    print(f'MAE: {mae_values[i]:.4f}')
    print(f'R²: {r2_values[i]:.4f}')
#    print(f'MAPE: {mape_values[i]:.4f}')
    print(f'SD: {sd_values[i]:.4f}')
    print(f'mu: {mu_values[i]:.4f}')
with open(folder_name + '/评估指标.txt', 'w', encoding='utf-8') as file:
    file.write(f'训练时间t1:{t1:.4f}\n')
    file.write(f'训练集数量:{len(x_train)}\n')
    file.write(f'预测时间t2:{t2:.4f}\n')
    file.write(f'测试集数量{len(x_test)}\n')
    for i in range(4):
        if i % 2 == 1:
            file.write('原始数据\n')
        else:
            file.write('去掉异常值\n')
        file.write(f'Output {i + 1}:(1、2:Age;3、4:MH)\n')
        file.write(f'MSE: {mse_values[i]:.4f}\n')
        file.write(f'MAE: {mae_values[i]:.4f}\n')
        file.write(f'R²: {r2_values[i]:.4f}\n')
#        file.write(f'MAPE: {mape_values[i]:.4f}\n')
        file.write(f'SD: {sd_values[i]:.4f}\n')
        file.write(f'mu: {mu_values[i]:.4f}\n') 
        file.write('\n')


