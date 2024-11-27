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
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg') 

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
        r'$\mu=%.5f$' % (mu,),
        r'$\sigma=%.5f$' % (sigma,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(min_num+0.1, max_num-0.1, textstr, fontsize=8, verticalalignment='top', bbox=props)

    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Estimated Values')
    #ax1.set_title(f'Scatter plot of True data and Model Estimated_{savefigname}')
    
    ax1.set_xlim((min_num, max_num))
    ax1.set_ylim((min_num, max_num))
    
    ax1.plot((min_num, max_num), (min_num, max_num), color='r', linewidth=1.5, linestyle='--')

    
    # 第二行的子图（占据较小宽度）
    ax2 = fig.add_subplot(gs[1, 0])
    res_0 = y - x  # 计算残差
    bins = np.arange(res_0.min()-0.2, res_0.max()+0.2, 0.1)
    ax2.hist(res_0, bins)
    '''
    for rect in ax2.patches:
        yval = rect.get_height()
        yval_int = int(np.round(yval))
        if yval_int != 0:
            xval = rect.get_x() + rect.get_width() / 2
            ax2.annotate(f'{yval_int}', (xval, yval),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points", ha='center', va='bottom')
    '''
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Number')
    #ax2.set_title(f'Histogram of Residuals_{savefigname}')
    ax2.xaxis.label.set_size(10)
    ax2.yaxis.label.set_size(10)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(folder_name, image_name)
    # 保存图像
    plt.savefig(f'{save_path}.png', dpi=900, bbox_inches='tight')
    plt.close()	
def denormalize(predicted_values, min_val, max_val):
    original_values = predicted_values * (max_val - min_val) + min_val
    return original_values


folder_name = 'model_4_2/att3_model2_2/1_6_conv'
i=1
# 检查文件夹是否存在
if not os.path.exists(folder_name):
    # 如果文件夹不存在，则创建它
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 已创建。")
else:
    print(f"文件夹 '{folder_name}' 已存在。")
# 数据预处理
df = pd.read_csv("model_2_2/100_3/pre_fits_1024.csv")#, nrows=30000)  #, skiprows=range(1, 3000),nrows=30000) #    ,
#df = pd.read_csv("04images_to_fits_result/pre_fits_4171.csv",nrows=20000)
#df = pd.read_csv("normalized_data_3W.csv")
#df = pd.read_csv('normalized_data_1453.csv')
images = []
img_names = []
images_names = df.iloc[:, 0].values
print(images_names.shape)

for index, row in df.iterrows():
    #img_name = row["photo_name"]
    img_name = row["images_names"]
    img_names.append(img_name)
    #img = load_img(f'0_images_crop/{row["photo_name"]}', target_size=(128, 128))
    img = load_img(f'0_images_crop/{row["images_names"]}', target_size=(128, 128))
    #img = load_img(f'images_crop/{row["images_names"]}', target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    images.append(img_array)

X_images = np.array(images)
print("X_images.shape:", X_images.shape)

X_fits = df.iloc[:, 2:1026].values
Y = df.iloc[:, [1026, 1027]].values
#X_fits = df.iloc[:, 2:2050].values
#Y = df.iloc[:, [2050, 2051]].values

#X_fits = df.iloc[:, 2:4173].values  #1026  2050 3074
#Y = df.iloc[:, [4173, 4174]].values
print("X_fits.shape:", X_fits.shape)

def min_max_normalize(arr):
    # 计算每个特征的最小值和最大值
    min_val = arr.min(axis=0)
    max_val = arr.max(axis=0)
    
    # 应用最小-最大归一化
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr
Y = min_max_normalize(Y)

'''
Y = np.array(Y).reshape(-1, Y.shape[1], 1).astype("float32")
# 对标签值进行归一化处理
scaler = list(range(Y.shape[1]))
for i in range(Y.shape[1]):
    scaler[i] = MinMaxScaler().fit(Y[:, i])
    Y[:, i] = scaler[i].transform(Y[:, i])

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

# 模型加载与特征提取
model_A = load_model("02images_pre_result/galaxy_image_model.h5", custom_objects={'coeff_determination': coeff_determination})
#model_A = load_model("model_3_1024_2/galaxy_image_model.h5", custom_objects={'coeff_determination': coeff_determination})
#model_B = load_model('model_1/fits_model_2para.h5', custom_objects={'coeff_determination': coeff_determination})

model_A_modified = tf.keras.models.Model(inputs=model_A.input, outputs=model_A.layers[-2].output)
#model_B_modified = tf.keras.models.Model(inputs=model_B.input, outputs=model_B.layers[-2].output)

image_features = model_A_modified.predict(X_images)
del X_images
gc.collect()
#X_fit = model_B_modified.predict(X_fits)


image_features = tf.cast(image_features, tf.float32)
spectral_features = tf.cast(X_fits, tf.float32)

# 定义分批处理的调用
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
fused_features = process_with_batching(image_features, spectral_features, batch_size=2048)

# 输出结果形状
print(f"Fused features shape (with batching): {fused_features.shape}")

'''
# 逐个归一化
normalized_arr = np.zeros_like(data)  # 初始化一个归一化后的数组
for i in range(data.shape[0]):
    min_val = np.min(data[i])
    max_val = np.max(data[i])
    normalized_arr[i] = (data[i] - min_val) / (max_val - min_val)
X = normalized_arr
'''
data = np.array(fused_features)
# 逐个归一化
def min_max_normalize(array):
    # 对每个样本进行归一化
    min_vals = array.min(axis=1, keepdims=True)
    max_vals = array.max(axis=1, keepdims=True)
    normalized_array = (array - min_vals) / (max_vals - min_vals + 1e-10)  # 加上一个小的数值避免除以零
    return normalized_array

# 归一化所有特征向量
X = min_max_normalize(data)
print(X[123].max())
print(X[123].min())


####模型
np.random.seed(42)
tf.random.set_seed(42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=i)


# 构建新的CNN模型1
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(1024,)))  #2048
#model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu')) 
#model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(2, activation='linear'))

# 编译模型
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
              loss='mean_squared_error', 
              metrics=[coeff_determination])

model.summary()



# 加载模型并对测试集进行预测 
model = tf.keras.models.load_model(folder_name +"/galaxy_mul_model.h5", custom_objects={'coeff_determination': coeff_determination})
y_pred_ = model.predict(x_test)

'''
# 创建一个标准化 (StandardScaler) 对象
y_test = np.array(y_test).reshape(-1, y_test.shape[1], 1).astype("float32")
y_pred_ = np.array(y_pred_).reshape(-1, y_pred_.shape[1], 1).astype("float32")
for i in range(y_test.shape[1]):
    y_test[:, i] = scaler[i].inverse_transform(y_test[:, i])
    y_pred_[:, i] = scaler[i].inverse_transform(y_pred_[:, i])
'''
'''
## Save as CSV
train, test = train_test_split(df, test_size=0.1, random_state=i)
df_c = pd.DataFrame()
df_c['imagesname'] = test['images_names'].values
df_c['fitsname'] = test['fits_names'].values
#df_c['imagesname'] = test['photo_name'].values
#df_c['fitsname'] = test['fits_name'].values
df_c['True_Age'] = y_test[:, 0]
df_c['True_mh'] = y_test[:, 1]

df_c['Pre_Age'] = y_pred_[:, 0]
df_c['Pre_mh'] = y_pred_[:, 1]
df_c.to_csv(folder_name + "/True_Predict_2para.csv", index=False)
'''

data = pd.read_csv('model_4_2/att3_model2_2/1_6/True_Predict_2para_1.csv.csv')

# 提取True_Age和True_MH列到y_test
y_test = data[['True_Age', 'True_MH']].to_numpy()

# 提取Pre_Age和Pre_MH列到y_pred_
y_pred_ = data[['Pre_Age', 'Pre_MH']].to_numpy()

age_max = 10.2
age_min = 7.8
mh_max = 0.22
mh_min = -1.71
mmin = [7.8,-1.71]
mmax = [10.2,0.22]
mse_values = []
mae_values = []
r2_values = []
mape_values = []
sd_values = []

para1 = ['logAge', '[M/H]']
para2 = ['logAge', 'mh']
for i in range(2):
    x = y_test[:, i]
    y = y_pred_[:, i]
    x = x.flatten()
    y = y.flatten()
    print(para1[i], "before:", len(x), len(y))
    x_ = denormalize(x, mmin[i], mmax[i])
    y_ = denormalize(y, mmin[i], mmax[i])
    x,y = remove_abnor(x_,y_)  #去掉异常值
    print(para1[i], "after:", len(x), len(y))
    plot_hexbin(x, y, folder_name, para2[i], f'去掉异常值_{para2[i]}')
    plot_hexbin(x_, y_, folder_name, para2[i], f'原始_{para2[i]}')
    plot_combined_chart(x, y, para2[i], f'去掉异常值_{para2[i]}')
    plot_combined_chart(x_, y_, para2[i], f'原始_{para2[i]}')
    print("x.shape",x.shape)
    print("y.shape",y.shape)
    
    mse_values.append(mean_squared_error(x, y))
    mse_values.append(mean_squared_error(x_, y_))
    mae_values.append(mean_absolute_error(x, y))
    mae_values.append(mean_absolute_error(x_, y_))
    r2_values.append(r2_score(x, y))
    r2_values.append(r2_score(x_, y_))
    mape_values.append(mean_absolute_percentage_error(x, y))
    mape_values.append(mean_absolute_percentage_error(x_, y_))
    sd_values.append(np.std(x - y))
    sd_values.append(np.std(x_ - y_))

# 打印每个输出的指标
for i in range(4):
    print(f'Output {i + 1}:')
    print(f'MSE: {mse_values[i]:.4f}')
    print(f'MAE: {mae_values[i]:.4f}')
    print(f'R²: {r2_values[i]:.4f}')
    print(f'MAPE: {mape_values[i]:.4f}')
    print(f'SD: {sd_values[i]:.4f}')

# 打开文件以写入模式，如果文件不存在会自动创建
with open(folder_name + '/评估指标.txt', 'w', encoding='utf-8') as file:
    file.write(f'time: {t}s\n')
    for i in range(4):
        if i%2==1:
            file.write('原始数据\n')
        else:
            file.write('去掉异常值\n')
        file.write(f'Output {i + 1}:(1、2:Age;3、4:MH)\n')
        file.write(f'MSE: {mse_values[i]:.4f}\n')
        #file.write(f'RMSE: {rmse_values[i]:.4f}\n')
        file.write(f'MAE: {mae_values[i]:.4f}\n')
        file.write(f'R²: {r2_values[i]:.4f}\n')
        file.write(f'MAPE: {mape_values[i]:.4f}\n')
        file.write(f'SD: {sd_values[i]:.4f}\n')
        file.write('\n')
