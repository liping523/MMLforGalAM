import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from tensorflow.keras.regularizers import l1, l2, L1L2
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler # 
from tensorflow.keras import backend as K  
from tensorflow.keras import models, layers  
from tensorflow.keras import optimizers  
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
from matplotlib import colors
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import MeanAbsoluteError

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

matplotlib.use('Agg')  
folder_name = 'model_1/1'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"文件夹 '{folder_name}' 已创建。")
else:
    print(f"文件夹 '{folder_name}' 已存在。")
    
data = np.load("data.npz", allow_pickle=True)
# 提取数据
img_names = data["images_name"]
fits_names = data["fits_name"]
logAge = data["logAge"]
MH = data["MH"]
X = data["fits_feature"]
Y = np.column_stack((logAge, MH))
print("Y[0].min():",Y[:,0].min())
print("Y[0].max():",Y[:,0].max())
print("Y[1].min():",Y[:,1].min())
print("Y[1].max():",Y[:,1].max())
print("Y.shape:",Y.shape)

# 创建一个 MinMaxScaler 对象
scaler = MinMaxScaler()
# 对 Y 的每一列进行归一化
Y = scaler.fit_transform(Y)

print("Y[0].min():",Y[:,0].min())
print("Y[0].max():",Y[:,0].max())
print("Y[1].min():",Y[:,1].min())
print("Y[1].max():",Y[:,1].max())
# 定义R^2系数的计算函数
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return 1 - SS_res / (SS_tot + K.epsilon())

def remove_abnor(x,y):
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

indices = np.arange(len(X))
X_train, x_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(
    X, Y, indices, test_size=0.2, random_state=1)

# 第二次分割：从临时集中分割出验证集和测试集（各占临时集的 50%）
X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(
    x_temp, y_temp, temp_indices, test_size=0.5, random_state=1)

np.random.seed(42)
tf.random.set_seed(42)
model = models.Sequential()
model.add(layers.Conv1D(16, 3, input_shape=(X.shape[1], 1), activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(16, 3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(3))

model.add(layers.Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(32, 3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(3))

model.add(layers.Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(3))

model.add(layers.Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling1D(3))

# 添加全连接层
model.add(layers.Flatten())
model.add(layers.Dense(4171, activation='relu'))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(2, activation='linear'))

# 编译模型，指定优化器为Adam，学习率为0.0001，损失函数为均方误差（Mean Squared Error），评估指标为R^2系数。
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=[coeff_determination])
model.summary()

start = time.time()
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=2) 

end = time.time()
model.save(folder_name + "/fits_model.h5")

# Convert history.history to DataFrame 将训练历史数据转换为DataFrame并保存为CSV文件
history_df = pd.DataFrame(history.history)
history_df.to_csv(folder_name + '/history_2para.csv', index=False)

t1 = end - start
print('Running time: %s Seconds' % (end - start))
with open(folder_name + '/modelsummary_fits.txt', 'w') as f:
    model.summary(print_fn=lambda x:f.write(x+'\n'))

# 绘制并保存训练过程中的损失图表
val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(folder_name +'/loss.png')
plt.close()

# 加载模型并对测试集进行预测 
model = tf.keras.models.load_model(folder_name + "/fits_model.h5", custom_objects={'coeff_determination': coeff_determination})           

start = time.time()
y_pred = model.predict(X_test)
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

# 反归一化
y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)
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
    x_,y_ = remove_abnor(x,y)  #去掉异常值
    print(para1[i], "after:", len(x), len(y))
    plot_hexbin(x_, y_, folder_name, para2[i], f'去掉异常值_{para2[i]}')
    plot_hexbin(x, y, folder_name, para2[i], f'原始_{para2[i]}')
    plot_combined_chart(x_, y_, para2[i], f'去掉异常值_{para2[i]}')
    plot_combined_chart(x, y, para2[i], f'原始_{para2[i]}')
    print("x.shape",x.shape)
    print("y.shape",y.shape)
    
    mse_values.append(mean_squared_error(x_, y_))
    mse_values.append(mean_squared_error(x, y))
    mae_values.append(mean_absolute_error(x_, y_))
    mae_values.append(mean_absolute_error(x, y))
    r2_values.append(r2_score(x_, y_))
    r2_values.append(r2_score(x, y))
    sd_values.append(np.std(x_ - y_))
    sd_values.append(np.std(x - y))
    mu_values.append(np.mean(x - y))
    mu_values.append(np.mean(x_ - y_))
# 打印每个输出的指标
for i in range(4):
    print(f'Output {i + 1}:')
    print(f'MSE: {mse_values[i]:.4f}')
    print(f'MAE: {mae_values[i]:.4f}')
    print(f'R²: {r2_values[i]:.4f}')
    print(f'SD: {sd_values[i]:.4f}')
    print(f'mu: {mu_values[i]:.4f}')

# 打开文件以写入模式，如果文件不存在会自动创建
with open(folder_name + '/评估指标.txt', 'w', encoding='utf-8') as file:
    file.write(f'训练时间t1:{t1:.4f}\n')
    file.write(f'训练集数量:{len(X_train)}\n')
    file.write(f'预测时间t2:{t2:.4f}\n')
    file.write(f'测试集数量{len(X_test)}\n')
    for i in range(4):
        if i%2==1:
            file.write('原始数据\n')
        else:
            file.write('去掉异常值\n')
        file.write(f'Output {i + 1}:(1、2:Age;3、4:MH)\n')
        file.write(f'MSE: {mse_values[i]:.4f}\n')
        file.write(f'MAE: {mae_values[i]:.4f}\n')
        file.write(f'R²: {r2_values[i]:.4f}\n')
        file.write(f'SD: {sd_values[i]:.4f}\n')
        file.write(f'mu: {mu_values[i]:.4f}\n')        
        file.write('\n')
  

            
