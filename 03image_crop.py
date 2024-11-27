import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import os
from PIL import Image

# 对图像进行裁剪，将黑色区域尽可能地裁剪掉
def crop_and_resize_image(image, target_size=(128, 128)):
    """
    裁剪掉图像中的纯黑区域，并将非黑区域调整大小以适应目标尺寸。

    参数:
    - image: numpy 数组，单个图像数据。
    - target_size: 目标尺寸元组，如 (128, 128)。

    返回:
    - resized_image: 裁剪并调整大小后的图像。
    """
    # 找出非零像素的索引
    ind = np.argwhere(image != 0)

    if ind.size == 0:
        # 如果图像全为零，则返回全零图像
        return np.zeros(target_size, dtype=image.dtype)

    # 计算非零区域的边界
    row_min, row_max = ind[:, 0].min(), ind[:, 0].max()
    col_min, col_max = ind[:, 1].min(), ind[:, 1].max()

    # 裁剪图像
    cropped_image = image[row_min:row_max + 1, col_min:col_max + 1, :]

    # 使用最近邻插值方法进行放缩到目标尺寸
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_NEAREST)

    return resized_image


# 读取CSV文件
df = pd.read_csv('data_normalized.csv')

# 根据CSV中的数据加载图像和对应的光谱特征
images = []
img_names = []
images_names = df.iloc[:, 0].values


for index, row in df.iterrows():
    # 加载图像并转换为数组
    img_name = row["photo_name"]
    img_names.append(img_name)
    img = load_img(f'0_image_pre/{row["photo_name"]}', target_size=(128, 128))
    img_array = img_to_array(img)
    # 归一化图像
    img_array /= 255.0
    images.append(img_array)

# 转换为numpy数组
images = np.array(images)
print(images.shape)
print(len(img_names))

# 遍历数组中的每个图像，进行裁剪和调整大小
cropped_resized_images = np.array([crop_and_resize_image(images[i]) for i in range(images.shape[0])])

# 将裁剪完成的图像进行保存


# 定义保存图像的文件夹
folder_name = "0_images_crop"

# 如果文件夹不存在，则创建它
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 遍历所有图像并保存
for idx, img_name in enumerate(img_names):
    # 获取图像数据
    image_data = cropped_resized_images[idx]

    # 确保图像数据是正确的类型和范围
    if image_data.dtype != np.uint8:
        # 归一化到 [0, 1]，然后转换为 uint8 类型
        image_data = (image_data - image_data.min()) * (255.0 / (image_data.max() - image_data.min()))
        image_data = image_data.astype(np.uint8)

    # 将 numpy 数组转换为 PIL Image 对象并保存
    image = Image.fromarray(image_data)
    image_path = os.path.join(folder_name, img_name)
    image.save(image_path, "JPEG", quality=95)  # 设置高质量保存

print("All images have been saved.")




