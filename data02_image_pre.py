import os
import numpy as np
import cv2
import math
import pandas as pd
import shutil
from skimage.io import imread
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import SourceCatalog
from photutils.segmentation import deblend_sources
from tqdm import tqdm  
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image


# 获取输入图像文件夹的路径
input_folder = '../images_5w_256x256_scale2'  # 输入图像的文件夹

main_folders = '256_scale2' # 输出文件夹-----不同像素的预处理结果放在不同的文件夹下
a = 128  # 图像中心点的坐标,例如:图像为64x64,中心点就是(32,32); 图像为128x128,中心点就是(64,64)
folders = [
    os.path.join(main_folders, 'images_1'), # 经过椭圆拟合处理后的图像（有效目标区域）
    os.path.join(main_folders, 'images_2'), # (最终的合格图像)经过裁剪和调整大小后的非混叠图像
    os.path.join(main_folders, 'images_3_1'),  #没有合适椭圆拟合结果的图像
    os.path.join(main_folders, 'images_3_2'),  # 在混叠判断中未检测到源的图像
    os.path.join(main_folders, 'images_3_3'),  # 混叠的图像（存在多个目标源）
    os.path.join(main_folders, 'images_4'),  # 非混叠的图像（单个清晰目标源）
    os.path.join(main_folders, 'images_3_4')  # 无效图像（背景或前景为空）
]


# 遍历文件夹列表，检查并创建不存在的文件夹
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"文件夹 '{folder}' 创建成功。")
    else:
        print(f"文件夹 '{folder}' 已经存在，跳过创建。")
        
        
# 定义函数 ellipse 处理图像
def ellipse(input_image, filename):
    # 如果图像为空，直接跳过并移动到新的文件夹
    if input_image is None or input_image.size == 0:
        #print(f"无效图像: {filename}")
        shutil.copy(os.path.join(input_folder, filename), os.path.join(folders[6], filename))
        return None

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # 计算最大和最小灰度值
    Cmax = np.max(gray_image).astype(np.uint16)
    Cmin = np.min(gray_image).astype(np.uint16)

    # 初始化阈值
    T0 = (Cmax + Cmin) // 2

    # 迭代计算自适应阈值
    while True:
        foreground = gray_image[gray_image > T0]
        background = gray_image[gray_image <= T0]

        # 如果前景或背景为空，跳过
        if foreground.size == 0 or background.size == 0:
            #print(f"无效图像（前景或背景为空）: {filename}")
            shutil.copy(os.path.join(input_folder, filename), os.path.join(folders[6], filename))
            return None

        Cobj = np.mean(foreground) if foreground.size > 0 else 0
        Cbk = np.mean(background) if background.size > 0 else 0

        T1 = (Cobj + Cbk) // 2

        if np.abs(T1 - T0) < 36:
            break

        T0 = T1

    _, auto_thresh = cv2.threshold(gray_image, T1, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(auto_thresh.astype(np.uint8), 30, 90)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_new = input_image.copy()
    contour_image = cv2.drawContours(image_new, contours, -1, (0, 255, 0), 1)

    # 拟合最优椭圆
    point_x = 128
    point_y = 128
    end_image = input_image.copy()
    end_ellipse = None
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center_x = ellipse[0][0]
            center_y = ellipse[0][1]
            a = ellipse[1][0] / 2
            b = ellipse[1][1] / 2
            angle = ellipse[2]
            cv2.ellipse(end_image, ellipse, (0, 255, 0), thickness=1)
            
            if a != 0 and b != 0:
                result = ((point_x - center_x) ** 2 / a ** 2) + ((point_y - center_y) ** 2 / b ** 2)
                if result <= 1:
                    end_ellipse = ellipse
                    cv2.ellipse(end_image, ellipse, (0, 0, 255), thickness=1)

    if end_ellipse is not None:
        mask = np.zeros(input_image.shape[:2], dtype=np.uint8)
        scale = 1.5
        center = (int(end_ellipse[0][0]), int(end_ellipse[0][1]))
        axes = (int(end_ellipse[1][0] / 2 * scale), int(end_ellipse[1][1] / 2 * scale))
        angle = int(end_ellipse[2])
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        result_image = cv2.bitwise_and(input_image, input_image, mask=mask)
        return result_image
    else:
        #print(f"无效图像（没有合适的椭圆）: {filename}")
        shutil.copy(os.path.join(input_folder, filename), os.path.join(folders[2], filename))
        return None
        
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
    
def get_dynamic_box_size(image_shape, base_size=(50, 50), base_image_size=128):
    """
    动态计算背景框大小，根据图像尺寸等比例调整。

    参数:
    - image_shape: 输入图像的形状 (height, width)。
    - base_size: 基准图像尺寸下的背景框大小，默认为 (50, 50)。
    - base_image_size: 基准图像的最小维度，默认为 128。

    返回:
    - 动态调整后的背景框大小 (box_height, box_width)。
    """
    min_dim = min(image_shape)  # 获取图像的最小维度
    scale_factor = min_dim / base_image_size  # 计算缩放比例
    box_height = int(base_size[0] * scale_factor)
    box_width = int(base_size[1] * scale_factor)
    return (box_height, box_width)
        
def is_blended_improved(image):
    tqdm.disable = True
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = img

    bkg_estimator = MedianBackground()
    data = data.astype(np.float64)
    # 动态调整背景框大小
    box_size = get_dynamic_box_size(data.shape, base_size=(50, 50), base_image_size=128)
    bkg = Background2D(data, box_size, filter_size=(3, 3), bkg_estimator=bkg_estimator, exclude_percentile=30)
    bkg.background = bkg.background.astype(np.float64)
    data -= bkg.background.astype(np.uint8)

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)  
    convolved_data = convolve(data, kernel)
    half_x = int(data.shape[0] / 2)
    half_y = int(data.shape[1] / 2)
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    if segment_map is None:
        return "未检测到源"

    pre_label = segment_map.data[half_x][half_y]
    pre_cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    pre_object = pre_cat.get_labels(pre_label).to_table()

    segm_deblend = deblend_sources(convolved_data, segment_map, npixels=10, nlevels=90, contrast=0.000001)
    cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)

    tbl = cat.to_table()
    label = segm_deblend.data[half_x][half_y]
    object = cat.get_labels(label).to_table()

    if len(tbl) == 1:
        return "不是混叠"
    else:
        return "是混叠"
        

#####1. 遮掩
# 初始化列表
no_ellipse = []
# 遍历输入文件夹中的图像文件
for filename in tqdm(os.listdir(input_folder), desc="处理图像（遮掩）", unit="file"):
    if filename.endswith('.jpg'):
        input_image = cv2.imread(os.path.join(input_folder, filename))
        processed_image = ellipse(input_image, filename)

        # 如果处理后的图像无效，则跳过
        if processed_image is None or processed_image.size == 0:
            no_ellipse.append(filename)
            continue

        output_path = os.path.join(folders[0], filename)
        cv2.imwrite(output_path, processed_image)

# 没有合适椭圆的图像数量
print("没有合适椭圆的图像数量：", len(no_ellipse))

### 混叠判断
images_pre = []  # 保存非混叠图像的文件名
notyuan_images = []
blended_image = []

# 使用 tqdm 显示进度条
with tqdm(os.listdir(folders[0]), desc="混叠判断", unit="image") as pbar:
    for filename in pbar:
        if filename.endswith('.jpg'):
            input_image = cv2.imread(os.path.join(folders[0], filename))
            result = is_blended_improved(input_image)
            if result == "不是混叠":
                images_pre.append(filename)  # 保存非混叠图像的文件名
                shutil.copy(os.path.join(folders[0], filename), os.path.join(folders[5], filename))
            elif result == "未检测到源":
                notyuan_images.append(filename)
                shutil.copy(os.path.join(folders[0], filename), os.path.join(folders[3], filename))
            else:
                blended_image.append(filename)
                shutil.copy(os.path.join(folders[0], filename), os.path.join(folders[4], filename))

print("没有检测到源的图像数量：", len(notyuan_images))
print("混叠的图像数量：", len(blended_image))
print("非混叠的图像数量：", len(images_pre))

### 裁剪
# 从非混叠图像文件夹中读取图像进行裁剪
img_names = os.listdir(folders[5])
img_names = [name for name in img_names if name.endswith('.jpg') or name.endswith('.png')]

# 遍历
for img_name in tqdm(img_names, desc="裁剪并保存"):
    # 使用PIL打开图像并调整大小
    img = Image.open(os.path.join(folders[5], img_name))
    img = img.resize((256, 256))  # 目标大小

    # 将图像转换为 NumPy 数组并归一化
    img_array = np.array(img) / 255.0

    # 裁剪和调整大小
    cropped_resized_image = crop_and_resize_image(img_array)

    # 确保图像数据是正确的类型和范围
    if cropped_resized_image.dtype != np.uint8:
        cropped_resized_image = (cropped_resized_image - cropped_resized_image.min()) * (255.0 / (cropped_resized_image.max() - cropped_resized_image.min()))
        cropped_resized_image = cropped_resized_image.astype(np.uint8)

    # 将 numpy 数组转换为 PIL Image 对象并保存
    image = Image.fromarray(cropped_resized_image)
    image_path = os.path.join(folders[1], img_name)
    image.save(image_path, "JPEG", quality=95)  # 设置高质量保存
    

# 遍历文件夹 folders[2] 和 folders[6]
no_ellipse = os.listdir(folders[2])  # 获取没有合适椭圆的图像文件名
no_back = os.listdir(folders[6])  # 获取无效图像文件名

# 过滤文件类型，仅保留图像文件（.jpg 和 .png）
no_ellipse = [name for name in no_ellipse if name.endswith('.jpg') or name.endswith('.png')]
no_back = [name for name in no_back if name.endswith('.jpg') or name.endswith('.png')]


# 保存不合格图像名称
file_name = os.path.join(main_folders, '剔除的图像名称.txt')
with open(file_name, "w") as file:
    file.write("没有合适椭圆的图像\n")
    for item in no_ellipse:
        file.write(f"{item}\n")
        
    file.write("背景或前景为空的图像\n")
    for item in no_back:
        file.write(f"{item}\n")

    file.write("没有检测到源的图像\n")
    for item in notyuan_images:
        file.write(f"{item}\n")

    file.write("混叠的图像\n")
    for item in blended_image:
        file.write(f"{item}\n")

print(f"文件 {file_name} 已写入完成。")


