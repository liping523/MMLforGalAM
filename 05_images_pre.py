import os
import time
import csv
import numpy as np
import matplotlib
import tensorflow as tf
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split  # 从sklearn库中导入数据集划分函数
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import optimizers
import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from matplotlib import colors

matplotlib.use('Agg')  
folder_name = 'new_m3'
i = 1
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 已创建。")
else:
    print(f"文件夹 '{folder_name}' 已存在。")

def coeff_determination(y_true, y_pred):
    """
    The coefficient of determination R^2 is often used in linear regression to represent the percentage of the dependent variable's variance explained by the regression line. If R^2 = 1, it indicates that the model perfectly predicts the target variable.
    Formula: R^2 = SSR/SST = 1 - SSE/SST
    Where: SST (total sum of squares) is the total sum of squares, SSR (regression sum of squares) is the regression sum of squares, and SSE (error sum of squares) is the residual sum of squares.
    """
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return 1 - SS_res / (SS_tot + K.epsilon())

# 定义异常值移除函数
def remove_abnor(x,y):
    '''
       Remove outliers beyond 3 sigma for two variables
    '''
    result_two = y - x  # 计算y和x之间的差值 result_two = y - x
    σ = 3 # 定义一个阈值 σ = 3，表示异常值的判断标准是超过3个标准差
    std_two = np.std(abs(result_two))  # 计算 result_two 的绝对值的标准差
    # 使用where()函数找到符合条件的索引，即差值在3个标准差范围内的索引。
    cc = np.where((result_two < result_two.mean() + σ * std_two) & (result_two > result_two.mean() - (σ * std_two)))
    # 根据索引从x和y中提取出符合条件的数据
    x = x[cc]
    y = y[cc]
    return x,y

def plot_hexbin(x, y, savefigdir, savefigname1, savefigname2):
    plt.close()
    plt.hexbin(x, y, gridsize=25, mincnt=0, vmax=20, cmap=plt.cm.gray_r)
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
        r'$\mu=%.4f$' % (mu,),
        r'$\sigma=%.4f$' % (sigma,)))
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
        r'Bias=%.4f' % (mu,),
        r'SD=%.4f' % (sigma,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(min_num+0.1, max_num-0.1, textstr, fontsize=8, verticalalignment='top', bbox=props)

    ax1.set_xlabel('True ' + savefigname)
    ax1.set_ylabel('Predicted ' + savefigname)
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
    plt.savefig(f'{save_path}.jpg', dpi=900, bbox_inches='tight')
    plt.close()	
    	
def denormalize(predicted_values, min_val, max_val):
    original_values = predicted_values * (max_val - min_val) + min_val
    return original_values
    
# 读取CSV文件
df = pd.read_csv('normalized_data_3W.csv')
#df = pd.read_csv('normalized_data_1453.csv')
# 根据CSV中的数据加载图像和对应的光谱特征
images = []
labels_data = []
img_names = []
labels_data = df.iloc[:, [4173, 4174]].values
images_names = df.iloc[:, 0].values

for index, row in df.iterrows():
    # 加载图像并转换为数组
    img_name = row["photo_name"]
    #img_name = row["images_names"]
    img_names.append(img_name)
    img = load_img(f'0_images_crop/{row["photo_name"]}', target_size=(128, 128))
    #img = load_img(f'images_crop/{row["images_names"]}', target_size=(128, 128))
    img_array = img_to_array(img)
    # 归一化图像
    img_array /= 255.0
    images.append(img_array)

X = np.array(images)
Y = labels_data
'''
Y = np.array(Y).reshape(-1, Y.shape[1], 1).astype("float32")
# 对标签值进行归一化处理
scaler = list(range(Y.shape[1]))
for i in range(Y.shape[1]):
    scaler[i] = MinMaxScaler().fit(Y[:, i])
    Y[:, i] = scaler[i].transform(Y[:, i])
'''
def min_max_normalize(arr):
    # 计算每个特征的最小值和最大值
    min_val = arr.min(axis=0)
    max_val = arr.max(axis=0)
    
    # 应用最小-最大归一化
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr
Y = min_max_normalize(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=i)

np.random.seed(42)
tf.random.set_seed(42)
'''
start = time.time()
# 利用二维卷积网络进行回归预测
# 构建新的CNN模型
model = Sequential()
# 第一个二维卷积层，使用L2正则化
model.add(Conv2D(32, (3, 3), input_shape=(X.shape[1], X.shape[2], X.shape[3]), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))

# 第二个二维卷积层
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))

# 最大池化层
model.add(MaxPooling2D((2, 2)))


# 第三组卷积层
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling2D((2, 2)))

# 第四组卷积层
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling2D((2, 2)))

# 第四组卷积层
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling2D((2, 2)))
# 展平层
model.add(Flatten())

# 全连接层
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
# 输出层，使用线性激活函数进行回归
model.add(Dense(Y.shape[1], activation='linear'))

# 编译模型，使用Adam优化器和均方误差作为损失函数
#model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[coeff_determination])
# 查看模型结构
model.summary()

# 训练模型，使用X_train和y_train作为训练数据，进行100个epochs的训练，使用1/9的数据作为验证集，verbose=2表示打印训练过程。
history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, verbose=2)
end = time.time()
history_df = pd.DataFrame(history.history)
# 保存模型
model.save(folder_name +'/galaxy_image_model.h5')
history_df.to_csv(folder_name +'/history_2para.csv', index=False)


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
plt.savefig(folder_name +'/loss.png')
plt.close()

t = end - start
print('Running time: %s Seconds' % (end - start))
with open(folder_name +'/modelsummary_images.txt', 'w') as f:
    model.summary(print_fn=lambda x:f.write(x+'\n'))
'''

# 加载模型并对测试集进行预测folder_name +
model = tf.keras.models.load_model("02images_pre_result/galaxy_image_model.h5", custom_objects={'coeff_determination': coeff_determination})
y_pred_ = model.predict(x_test)

'''
# Restore the scaled target variables and prediction results back to their original ranges
# 将缩放后的目标变量和预测结果恢复到原来的范围
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
df_c['imagesname'] = test['photo_name'].values
df_c['fitsname'] = test['fits_name'].values
df_c['True_Age'] = y_test[:, 0]
df_c['True_mh'] = y_test[:, 1]

df_c['Pre_Age'] = y_pred_[:, 0]
df_c['Pre_mh'] = y_pred_[:, 1]
df_c.to_csv(folder_name + "/True_Predict_2para.csv", index=False)
'''

mse_values = [mean_squared_error(y_test[:, i], y_pred_[:, i]) for i in range(2)]
rmse_values = [mse ** 0.5 for mse in mse_values]
mae_values = [mean_absolute_error(y_test[:, i], y_pred_[:, i]) for i in range(2)]
r2_values = [r2_score(y_test[:, i], y_pred_[:, i]) for i in range(2)]
mape_values = [mean_absolute_percentage_error(y_test[:, i], y_pred_[:, i]) for i in range(2)]

# 计算标准差
sd_values = [np.std(y_test[:, i] - y_pred_[:, i]) for i in range(2)]

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
    plot_combined_chart(x, y, para1[i], f'去掉异常值_{para2[i]}')
    plot_combined_chart(x_, y_, para1[i], f'原始_{para2[i]}')
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
    #file.write(f't:{t:.4f}\n')
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

