import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from tensorflow.keras.regularizers import l1, l2, L1L2
from sklearn.model_selection import train_test_split  # 从sklearn库中导入数据集划分函数
from sklearn.preprocessing import MinMaxScaler # 从sklearn库中导入MinMaxScaler数据标准化器
from tensorflow.keras import backend as K  # 从tensorflow.keras库中导入backend
from tensorflow.keras import models, layers  # 从tensorflow.keras库中导入模型和层定义部分
from tensorflow.keras import optimizers  
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
from matplotlib import colors

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

matplotlib.use('Agg')  
folder_name = 'model_1_2048'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 已创建。")
else:
    print(f"文件夹 '{folder_name}' 已存在。")
    
# 定义R^2系数的计算函数
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
    idx = z.argsort()
    
    # 使用Normalize来设置颜色柱的范围
    norm = colors.Normalize(vmin=0, vmax=5)
    
    scatter = ax1.scatter(x, y, marker='o', c=z, edgecolors='none', s=50, cmap='Spectral_r', norm=norm)
    cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Estimated Values')
    ax1.set_title(f'Scatter plot of True data and Model Estimated_{savefigname}')
    
    ax1.set_xlim((min_num, max_num))
    ax1.set_ylim((min_num, max_num))
    
    ax1.plot((min_num, max_num), (min_num, max_num), color='r', linewidth=1.5, linestyle='--')

    # 第二行的子图（占据较小宽度）
    ax2 = fig.add_subplot(gs[1, 0])
    res_0 = y - x  # 计算残差
    bins = np.arange(res_0.min()-0.2, res_0.max()+0.2, 0.1)
    ax2.hist(res_0, bins)
    for rect in ax2.patches:
        yval = rect.get_height()
        yval_int = int(np.round(yval))
        if yval_int != 0:
            xval = rect.get_x() + rect.get_width() / 2
            ax2.annotate(f'{yval_int}', (xval, yval),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points", ha='center', va='bottom')

    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Number')
    ax2.set_title(f'Histogram of Residuals_{savefigname}')
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

#df = pd.read_csv("data_normalized.csv")
#df = pd.read_csv('normalized_data_1453.csv')
df = pd.read_csv('normalized_data_3W.csv')

# 提取光谱数据和标签值
X = df.iloc[:, 2:4173].values
Y = df.iloc[:, [4173, 4174]].values  # 年龄、金属丰度
# Normalize Y
Y = np.array(Y).reshape(-1, Y.shape[1], 1).astype("float32")
# 对标签值进行归一化处理
scaler = list(range(Y.shape[1]))
for i in range(Y.shape[1]):
    scaler[i] = MinMaxScaler().fit(Y[:, i])
    Y[:, i] = scaler[i].transform(Y[:, i])

# Split the dataset into training and testing sets 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
X_train = np.array(X_train).reshape(-1, X.shape[1], 1).astype("float32")
X_test = np.array(X_test).reshape(-1, X.shape[1], 1).astype("float32")

start = time.time()
# Build the CNN model 构建CNN模型
model = models.Sequential()
# 向模型中添加一个一维卷积层，该层有16个滤波器，卷积核大小为3，输入形状为(X.shape[1], 1)，激活函数为ReLU，使用L2正则化（权重衰减）防止过拟合。
model.add(layers.Conv1D(16,3, input_shape=((X.shape[1], 1)), activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(16, 3, activation='relu',kernel_regularizer=l2(0.001)))

model.add(layers.MaxPooling1D(2))

model.add(layers.Conv1D(32, 3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(32, 3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(2))

model.add(layers.Conv1D(64, 3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(64, 3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(2))

model.add(layers.Conv1D(128, 3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(128, 3, activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(4171, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='linear'))
# 编译模型，指定优化器为Adam，学习率为0.0001，损失函数为均方误差（Mean Squared Error），评估指标为R^2系数。
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[coeff_determination])
model.summary()

# 训练模型，使用X_train和y_train作为训练数据，进行100个epochs的训练，使用1/9的数据作为验证集，verbose=2表示打印训练过程。
print(X_train.shape)
print(y_train.shape)
history = model.fit(X_train, y_train, epochs=100, validation_split=1/9, verbose=2)
# Convert history.history to DataFrame 将训练历史数据转换为DataFrame并保存为CSV文件
history_df = pd.DataFrame(history.history)
# Save as CSV file
history_df.to_csv(folder_name + '/history_2para.csv', index=False)
# Save the model
model.save(folder_name + "/fits_model_2para.h5")
end = time.time()
t = end - start
print('Running time: %s Seconds' % (end - start))
with open(folder_name + '/modelsummary_fits.txt', 'w') as f:
    model.summary(print_fn=lambda x:f.write(x+'\n'))

# 绘制并保存训练过程中的损失图表
val_loss = history.history['val_loss']
loss = history.history['loss']
# # 从CSV文件中读取进行图表的绘制
#history_df = pd.read_csv('01fits_pre_result/history_4para_L2.csv')
#val_loss = history_df['val_loss']
#loss = history_df['loss']

plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(folder_name +'/loss.png')
plt.close()

# 加载模型并对测试集进行预测            
model = tf.keras.models.load_model(folder_name + "/fits_model_2para.h5", custom_objects={'coeff_determination': coeff_determination})
y_pred_ = model.predict(X_test)

# Restore the scaled target variables and prediction results back to their original ranges
# 将缩放后的目标变量和预测结果恢复到原来的范围
# 创建一个标准化 (StandardScaler) 对象
y_test = np.array(y_test).reshape(-1, y_test.shape[1], 1).astype("float32")
y_pred_ = np.array(y_pred_).reshape(-1, y_pred_.shape[1], 1).astype("float32")
for i in range(y_test.shape[1]):
    y_test[:, i] = scaler[i].inverse_transform(y_test[:, i])
    y_pred_[:, i] = scaler[i].inverse_transform(y_pred_[:, i])
## Save as CSV
train, test = train_test_split(df, test_size=0.1, random_state=1)
df_c = pd.DataFrame()
df_c['imagesname'] = test['photo_name'].values
df_c['fitsname'] = test['fits_name'].values
df_c['True_Age'] = y_test[:, 0]
df_c['True_mh'] = y_test[:, 1]

df_c['Pre_Age'] = y_pred_[:, 0]
df_c['Pre_mh'] = y_pred_[:, 1]
df_c.to_csv(folder_name + "/True_Predict_2para.csv", index=False)

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
    file.write(f't:{t:.4f}\n')
    for i in range(4):
        if i%2==1:
            file.write('原始数据\n')
        else:
            file.write('去掉异常值\n')
        file.write(f'Output {i + 1}:(1、2:Age;3、4:MH)\n')
        file.write(f'MSE: {mse_values[i]:.4f}\n')
        file.write(f'MAE: {mae_values[i]:.4f}\n')
        file.write(f'R²: {r2_values[i]:.4f}\n')
        file.write(f'MAPE: {mape_values[i]:.4f}\n')
        file.write(f'SD: {sd_values[i]:.4f}\n')
        file.write('\n')
  

            
