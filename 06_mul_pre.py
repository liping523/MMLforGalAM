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
'''
#######第一种自注意力机制
# 定义自定义注意力层
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_dense = tf.keras.layers.Dense(d_model)
        self.k_dense = tf.keras.layers.Dense(d_model)
        self.v_dense = tf.keras.layers.Dense(d_model)

        self.attention_dropout = tf.keras.layers.Dropout(attention_dropout)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 3, 1])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_scores = tf.matmul(q, k, transpose_b=True) / (self.d_model ** 0.5)

        if mask is not None:
            attention_scores = attention_scores * (1 - tf.cast(mask, dtype=attention_scores.dtype))

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context = tf.matmul(attention_probs, v)

        context = tf.transpose(context, perm=[0, 3, 1, 2])
        context = tf.reshape(context, (batch_size, -1, self.d_model))

        return context

class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, attention_dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        self.multi_head_attention = ScaledDotProductAttention(d_model, num_heads, attention_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None):
        attention_output = self.multi_head_attention(inputs, inputs, inputs, mask)
        return self.layer_norm(attention_output + inputs)

class SelfAttentionModel(tf.keras.Model):
    def __init__(self, d_model, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.self_attention = SelfAttentionLayer(d_model, num_heads)

    def call(self, inputs):
        return self.self_attention(inputs)
        

# 特征融合
model = SelfAttentionModel(512, 4)
model2 = SelfAttentionModel(1024, 8)
image_features_3 = np.reshape(image_features, (1, image_features.shape[0], 512))  #1024
output_image = model(image_features_3)
fits_features_3 = np.reshape(fits_features, (1, fits_features.shape[0], 1024))
output_fits = model2(fits_features_3)
print(output_image.shape)
print(output_fits.shape)
image_features_2 = np.reshape(output_image, (output_image.shape[1], 512))  #1024
fits_features_2 = np.reshape(output_fits, (output_fits.shape[1], 1024))   #1024
print(image_features_2.shape)
print(fits_features_2.shape)
merged_features = tf.keras.layers.Concatenate()([image_features_2, fits_features_2])


merged_feature = tf.keras.layers.Concatenate()([image_features, fits_features])
model = SelfAttentionModel(1536,8)
out_features = np.reshape(merged_feature, (1, merged_feature.shape[0], 1536))
output_mer = model(out_features)
merged_features = np.reshape(output_mer, (output_mer.shape[1], 1536))

print("Output merged_features Shape:", merged_features.shape)

'''

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
'''
######使用3w数据 结果为0.16
class AttentionFusion(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(AttentionFusion, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm_image = LayerNormalization()
        self.norm_spectral = LayerNormalization()
        self.dropout = Dropout(0.1)

    def call(self, image_features, spectral_features):
        # 扩展维度以适应多头自注意力的输入
        image_features_proj = tf.expand_dims(image_features, axis=1)  # (batch_size, 1, 1024)
        spectral_features_proj = tf.expand_dims(spectral_features, axis=1)  # (batch_size, 1, 1024)

        # 对图像特征应用自注意力
        attention_output_image = self.multi_head_attention(query=image_features_proj, key=spectral_features_proj, value=spectral_features_proj)
        attention_output_image = self.norm_image(attention_output_image + image_features_proj)

        # 对光谱特征应用自注意力
        attention_output_spectral = self.multi_head_attention(query=spectral_features_proj, key=image_features_proj, value=image_features_proj)
        attention_output_spectral = self.norm_spectral(attention_output_spectral + spectral_features_proj)

        # 融合图像和光谱特征（加权求和）
        fused_features = attention_output_image + attention_output_spectral

        # 融合后的特征取消扩展的维度
        fused_features = tf.squeeze(fused_features, axis=1)

        # 加入 dropout 进行正则化
        fused_features = self.dropout(fused_features)

        return fused_features
image_features = tf.cast(image_features, tf.float32)
X_fits = tf.cast(X_fits, tf.float32)


# 注意力融合
num_heads = 8  # 设置多头注意力的头数
key_dim = 1024  # 特征维度保持 1024
fusion_layer = AttentionFusion(num_heads=num_heads, key_dim=key_dim)
merged_features = fusion_layer(image_features, X_fits)
'''
'''
### 分开处理
class AttentionFusion(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(AttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.norm_image = LayerNormalization()
        self.norm_spectral = LayerNormalization()
        self.dropout = Dropout(0.1)
    
    def compute_attention_weights(self, features):
        # 计算每个特征中的非0值数量
        non_zero_counts = tf.reduce_sum(tf.cast(features != 0, dtype=tf.float32), axis=-1, keepdims=True)
        # 计算注意力权重，权重是非0值的比例
        attention_weights = non_zero_counts / tf.reduce_sum(non_zero_counts)
        return attention_weights
    
    def process_image_attention(self, image_features, spectral_features):
        image_features_proj = tf.expand_dims(image_features, axis=1)  
        spectral_features_proj = tf.expand_dims(spectral_features, axis=1)
        attention_output_image = self.multi_head_attention(query=image_features_proj, key=spectral_features_proj, value=spectral_features_proj)
        #attention_output_image = tf.squeeze(attention_output_image, axis=1)
     
        # 计算图像特征的注意力权重
        #attention_weights = self.compute_attention_weights(image_features)
        # 将注意力权重应用到输出上
        #attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1) 
        #attention_output_image = attention_output_image * attention_weights_expanded
        attention_output_image = self.norm_image(attention_output_image + image_features_proj)
        return attention_output_image

    def process_spectral_attention(self, image_features, spectral_features):
        image_features_proj = tf.expand_dims(image_features, axis=1)
        spectral_features_proj = tf.expand_dims(spectral_features, axis=1)

        attention_output_spectral = self.multi_head_attention(query=spectral_features_proj, key=image_features_proj, value=image_features_proj)
        # 计算光谱特征的注意力权重
        #attention_weights = self.compute_attention_weights(spectral_features)
        
        # 将注意力权重应用到输出上
        #attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1) 
        #attention_output_spectral = attention_output_spectral * attention_weights_expanded
        attention_output_spectral = self.norm_spectral(attention_output_spectral + spectral_features_proj)

        return attention_output_spectral

    def call(self, image_features, spectral_features):
        # 图像注意力处理
        #attention_output_image = self.process_image_attention(image_features, spectral_features)
        # 光谱注意力处理
        #attention_output_spectral = self.process_spectral_attention(image_features, spectral_features)

        # 融合图像和光谱特征
        #fused_features = attention_output_image + attention_output_spectral
        fused_features = image_features + spectral_features
        #print("fused_features shape:", fused_features.shape)
        fused_features = tf.squeeze(fused_features, axis=1)
        
        # 加入dropout
        fused_features = self.dropout(fused_features)

        return fused_features
'''
'''
#交互注意力
class AttentionFusion(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(AttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.norm_image = LayerNormalization()
        self.norm_spectral = LayerNormalization()
        self.dropout = Dropout(0.1)

    def compute_attention_weights(self, features):
        # 计算每个特征中的非0值数量
        non_zero_counts = tf.reduce_sum(tf.cast(features != 0, dtype=tf.float32), axis=-1, keepdims=True)
        # 计算注意力权重，权重是非0值的比例
        attention_weights = non_zero_counts / tf.reduce_sum(non_zero_counts)
        return attention_weights

    def process_image_attention(self, image_features, spectral_features):
        image_features_proj = tf.expand_dims(image_features, axis=1)  
        spectral_features_proj = tf.expand_dims(spectral_features, axis=1)
        attention_output_image = self.multi_head_attention(
            query=image_features_proj,
            key=spectral_features_proj,
            value=spectral_features_proj
        )
        
        # 计算图像特征的注意力权重
        #attention_weights = self.compute_attention_weights(image_features)
        # 将注意力权重应用到输出上
        #attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1) 
        #attention_output_image = attention_output_image * attention_weights_expanded
        
        attention_output_image = self.norm_image(attention_output_image + image_features_proj)
        return attention_output_image

    def process_spectral_attention(self, image_features, spectral_features):
        image_features_proj = tf.expand_dims(image_features, axis=1)
        spectral_features_proj = tf.expand_dims(spectral_features, axis=1)

        attention_output_spectral = self.multi_head_attention(
            query=spectral_features_proj,
            key=image_features_proj,
            value=image_features_proj
        )
        
        # 计算光谱特征的注意力权重
        #attention_weights = self.compute_attention_weights(spectral_features)
        # 将注意力权重应用到输出上
        #attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1) 
        #attention_output_spectral = attention_output_spectral * attention_weights_expanded
        
        attention_output_spectral = self.norm_spectral(attention_output_spectral + spectral_features_proj)
        return attention_output_spectral

    def call(self, image_features, spectral_features):
        # 图像注意力处理
        attention_output_image = self.process_image_attention(image_features, spectral_features)
        
        # 光谱注意力处理
        attention_output_spectral = self.process_spectral_attention(image_features, spectral_features)
        
        # 融合图像和光谱特征
        fused_features = attention_output_image + attention_output_spectral
        
        fused_features = tf.squeeze(fused_features, axis=1)
        
        # 加入dropout
        fused_features = self.dropout(fused_features)

        return fused_features

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
fused_features = process_with_batching(image_features, spectral_features, batch_size=2048)

# 输出结果形状
print(f"Fused features shape (with batching): {fused_features.shape}")


'''
###分开处理,将中间结果进行保存
num_heads = 8
key_dim = 1024

fusion_layer = AttentionFusion(num_heads=num_heads, key_dim=key_dim)
attention_output_image_1 = fusion_layer.process_image_attention(images_features[:30000], spectral_features[:30000])
attention_output_image_2 = fusion_layer.process_image_attention(images_features[30000:], spectral_features[30000:])
attention_output_image = tf.concat([attention_output_image_1,attention_output_image_2],axis=0)
#attention_output_image = fusion_layer.process_image_attention(images_features, spectral_features)
np.save(f"{folder_name}/attention_output_image.npy", attention_output_image.numpy())  # 保存图像特征

# 释放内存
del attention_output_image
gc.collect() 

# 第二步: 处理光谱特征
attention_output_spectral = fusion_layer.process_spectral_attention(images_features, spectral_features)
np.save(f"{folder_name}/attention_output_spectral.npy", attention_output_spectral.numpy())  # 保存光谱特征
# 释放内存
del attention_output_spectral
gc.collect()

# 从文件中加载保存的结果
image_attention_output = np.load(f"{folder_name}/attention_output_image.npy")
spectral_attention_output = np.load(f"{folder_name}/attention_output_spectral.npy")

# 将加载的数据转换为Tensor
image_attention_output = tf.convert_to_tensor(image_attention_output, dtype=tf.float32)
spectral_attention_output = tf.convert_to_tensor(spectral_attention_output, dtype=tf.float32)
print("image_attention_output shape:", image_attention_output.shape)
print("spectral_attention_output shape:", spectral_attention_output.shape)

merged_features = fusion_layer.call(image_attention_output, spectral_attention_output)

print("Fused features shape:", merged_features.shape)
'''
'''
# 注意力融合
image_features = tf.cast(image_features, tf.float32)
X_fits = tf.cast(X_fits, tf.float32)
num_heads = 8  # 设置多头注意力的头数
key_dim = 1024  # 特征维度保持 1024
fusion_layer_image = Attention_image(num_heads=num_heads, key_dim=key_dim)
#  不进行批处理
#merged_features = fusion_layer(image_features, X_fits)
'''

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
'''
from sklearn.preprocessing import RobustScaler
from scipy import stats

# 计算每个特征的中位数
X = np.copy(normalized_arr)
for i in range(normalized_arr.shape[1]):
    median = np.median(normalized_arr[:, i])
    # 将离群值替换为中位数
    X[np.abs(stats.zscore(normalized_arr[:, i])) > 3, i] = median

# 再次检测离群值
z_scores = np.abs(stats.zscore(X))
outliers = np.where(z_scores > 3)

print(f'Number of outliers after replacing: {len(outliers[0])}')
'''
'''
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
'''

'''

model = models.Sequential()
model.add(layers.Conv1D(64,3, input_shape=((X.shape[1], 1)), activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(128,3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(2))
#model.add(Dropout(0.1)
model.add(layers.Conv1D(256,3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(2))
# 展平层
model.add(layers.Flatten())
# 全连接层
model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dense(512, activation='relu'))
# 输出层 (2维)
model.add(layers.Dense(2, activation='linear'))

# 编译模型
#model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[coeff_determination])
# 打印模型摘要
model.summary()
'''
#x_train = np.array(x_train).reshape(-1, X.shape[1], 1).astype("float32")
#x_test = np.array(x_test).reshape(-1, X.shape[1], 1).astype("float32")

'''
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
history = []
start = time.time()
history = model.fit(x_train, y_train, epochs=100, batch_size=64,validation_split=1/9, verbose=2)

end = time.time()

model.save(folder_name + '/galaxy_mul_model.h5')

history_df = pd.DataFrame(history.history)
history_df.to_csv(folder_name + '/history_2para.csv', index=False)

# 绘制并保存训练过程中的损失图表
val_loss = history.history['val_loss']
loss = history.history['loss']
# # 从CSV文件中读取进行图表的绘制
# val_loss = history_df['val_loss']
# loss = history_df['loss']

plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(folder_name + '/loss.png')
plt.close()
t = end - start
print('Running time: %s Seconds' % (end - start))
with open(folder_name + '/modelsummary_mul.txt', 'w') as f:
    model.summary(print_fn=lambda x:f.write(x+'\n'))
'''
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
    #file.write(f'time: {t}s\n')
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


