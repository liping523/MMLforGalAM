import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import pandas as pd
import os
import shutil
from skimage.io import imread
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import SourceCatalog
from photutils.segmentation import deblend_sources
'''
# 定义函数ellipse处理图像，并保存处理后的结果
def ellipse(input_image):
    # 灰度
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # 1.自适应阈值进行二值化操作，进而绘制轮廓。

    # 计算最大和最小灰度值
    Cmax = np.max(gray_image).astype(np.uint16)
    Cmin = np.min(gray_image).astype(np.uint16)

    # 初始化阈值
    T0 = (Cmax + Cmin) // 2

    # 迭代计算自适应阈值
    while True:
        # 将图像分为前景和背景
        foreground = gray_image[gray_image > T0]
        background = gray_image[gray_image <= T0]

        # 计算前景和背景的平均灰度值
        Cobj = np.mean(foreground)
        Cbk = np.mean(background)

        # 更新阈值
        T1 = (Cobj + Cbk) // 2

        # 判断是否收敛
        if np.abs(T1 - T0) < 36:
            break

        # 更新阈值并继续迭代
        T0 = T1
    # print(T1)
    # 自适应二值化
    _, auto_thresh = cv2.threshold(gray_image, T1, 255, cv2.THRESH_BINARY)

    # 边缘检测
    edges = cv2.Canny(auto_thresh.astype(np.uint8), 30, 90)

    # 轮廓提取
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 在图像上绘制轮廓
    image_new = input_image.copy()
    contour_image = cv2.drawContours(image_new, contours, -1, (0, 255, 0), 1)

    # 拟合最优椭圆
    point_x = 64
    point_y = 64
    end_image = input_image.copy()
    end_ellipse = None
    for contour in contours:
        if len(contour) >= 5:
            # 拟合最小椭圆
            ellipse = cv2.fitEllipse(contour)
            # print(ellipse)
            # 椭圆参数
            center_x = ellipse[0][0]
            center_y = ellipse[0][1]
            a = ellipse[1][0] / 2
            b = ellipse[1][1] / 2
            angle = ellipse[2]
            cv2.ellipse(end_image, ellipse, (0, 255, 0), thickness=1)  # 在图像上绘制椭圆
            # 检查长轴和短轴是否为零
            if a != 0 and b != 0:
                # 将给定点的坐标代入椭圆方程计算左侧结果
                result = ((point_x - center_x) ** 2 / a ** 2) + ((point_y - center_y) ** 2 / b ** 2)

                # 判断给定点是否在椭圆内
                if result <= 1:
                    # print("给定点在椭圆内")

                    end_ellipse = ellipse  # 保存符合的椭圆

                    cv2.ellipse(end_image, ellipse, (0, 0, 255), thickness=1)  # 在图像上绘制椭圆
    if end_ellipse is not None:
        # 遮盖
        mask = np.zeros(input_image.shape[:2], dtype=np.uint8)
        # 设置扩大的比例
        scale = 1.5  # 扩大的比例
        # 定义椭圆参数
        center = (int(end_ellipse[0][0]), int(end_ellipse[0][1]))
        axes = (int(end_ellipse[1][0] / 2 * scale), int(end_ellipse[1][1] / 2 * scale))
        angle = int(end_ellipse[2])
        # 在掩码上绘制椭圆区域（白色部分）
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        # 将椭圆范围外的部分设置为黑色
        result_image = cv2.bitwise_and(input_image, input_image, mask=mask)

        return result_image


# 获取pro_test文件夹中的jpg文件列表
input_folder = '5w_images'  # 指定输入文件夹
pre_folder = 'image_pre'  # 指定输出文件夹

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(pre_folder):
    os.makedirs(pre_folder)

no_ellipse = []
# 遍历pro_test文件夹中的jpg文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # 读取图像
        input_image = cv2.imread(os.path.join(input_folder, filename))

        # 调用ellipse函数处理图像
        processed_image = ellipse(input_image)

        # 保存处理后的图像到文件夹中，保持与原始图像名字一致
        output_path = os.path.join(pre_folder, filename)
        # print(filename)
        if processed_image is None or processed_image.size == 0:
            # print(f"Processed image is empty for file: {filename}")
            no_ellipse.append(filename)
            continue

        cv2.imwrite(output_path, processed_image)
# 没有合适椭圆的图像数量
print("没有合适椭圆的图像数量", len(no_ellipse))

def is_blended_improved(image):
    ###fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    # image = to_rgb.dr2_rgb(file,['g','r','z'])
    # image = cv2.imread(file)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = img

    bkg_estimator = MedianBackground()
    data = data.astype(np.float64)  # 将图像数据转换为 float64 类型
    bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator, exclude_percentile=30)
    bkg.background = bkg.background.astype(np.float64)  # 将背景数据转换为 float64 类型
    data -= bkg.background.astype(np.uint8)  # 将背景数据转换为 uint8 类型进行减法操作

    threshold = 1.5 * bkg.background_rms  # detect_sources的参数，高于此阈值的会认定是目标，否则就认为是背景
    # 调低阈值可以识别出来更暗的星系，但是也容易产生噪点降低识别率

    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    half_x = int(data.shape[0] / 2)
    half_y = int(data.shape[1] / 2)
    # 检测到的源x   npixels代表表示一个被检测到的源或目标必须具有的最小像素数量。只有连接在一起的像素点数大于阈值的源才会被检测到
    segment_map = detect_sources(convolved_data, threshold, npixels=10)  # 此处就是像素点数大于10才会被检测到
    if segment_map is None:
        # 处理未检测到源的情况
        return "未检测到源"

    pre_label = segment_map.data[half_x][half_y]
    pre_cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    pre_object = pre_cat.get_labels(pre_label).to_table()

    segm_deblend = deblend_sources(convolved_data, segment_map,  # npixels含义与detect_sources中的相同
                                   npixels=10, nlevels=90, contrast=0.000001,
                                   progress_bar=False)
    cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)

    tbl = cat.to_table()
    label = segm_deblend.data[half_x][half_y]
    object = cat.get_labels(label).to_table()

    if len(tbl) == 1:
        return "不是混叠"

    else:
        return "是混叠"  # 重叠


output_eli_folder = 'eli_image'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_eli_folder):
    os.makedirs(output_eli_folder)

images_pre = []
notfound_images = []
blended_image = []
# 遍历pro_test文件夹中的jpg文件
for filename in os.listdir(pre_folder):
    if filename.endswith('.jpg'):
        # print(filename)
        # 读取图像
        input_image = cv2.imread(os.path.join(pre_folder, filename))

        # 调用函数判断图像是否混叠
        result = is_blended_improved(input_image)
        # print(result)
        if result == "不是混叠":
            images_pre.append(filename)
        elif result == "未检测到源":
            notfound_images.append(filename)
            shutil.move(os.path.join(pre_folder, filename), os.path.join(output_eli_folder, filename))
        else:
            blended_image.append(filename)
            shutil.move(os.path.join(pre_folder, filename), os.path.join(output_eli_folder, filename))

print("非混叠的图像数量：", len(images_pre))
print("没有检测到源的图像数量：", len(notfound_images))
print("混叠的图像数量：", len(blended_image))
# 定义文件名
file_name = "剔除的图像名称.txt"

# 打开文件进行写入
with open(file_name, "w") as file:
    # 写入数组一的标题和内容
    file.write("没有合适椭圆的图像\n")
    for item in no_ellipse:
        file.write(f"{item}\n")

    # 写入数组二的标题和内容
    file.write("没有检测到源的图像\n")
    for item in notfound_images:
        file.write(f"{item}\n")

    # 写入数组三的标题和内容
    file.write("混叠的图像\n")
    for item in blended_image:
        file.write(f"{item}\n")

print(f"文件 {file_name} 已写入完成。")
ex_filenames = no_ellipse + notfound_images + blended_image

'''

txt_filename = '剔除的图像名称.txt'

# 预定义的标题行数组
header_lines = ["没有合适椭圆的图像","没有检测到源的图像", "混叠的图像"]

# 初始化一个空列表来存储合并后的文件名
ex_filenames = []

# 打开并读取txt文件
with open(txt_filename, 'r') as file:
    lines = file.readlines()  # 读取所有行到一个列表

# 遍历文件的每一行
for line in lines:
    # 去除每行的首尾空白字符，包括换行符
    stripped_line = line.strip()
    # 检查当前行是否在标题行数组中
    if stripped_line in header_lines:
        # 如果是标题行，跳过
        continue
    else:
        # 如果不是标题行，则添加到列表中
        ex_filenames.append(stripped_line)

# 输出合并后的文件名列表
print("ex_filenames_num:",len(ex_filenames))


csv_filename = 'pre_data.csv'

# 读取CSV文件
df_1 = pd.read_csv(csv_filename)

# 检查ex_filenames中的文件名是否在df的photo_name列中
rows_to_delete = df_1.index[df_1['photo_name'].isin(ex_filenames)]

# 如果rows_to_delete不为空，则删除这些行
if rows_to_delete.size > 0:
    df_1.drop(rows_to_delete, inplace=True)

# 计算删除后CSV文件的行数
remaining_rows = len(df_1)

# 输出剩余的行数
print(f"After deletion, the CSV file has {remaining_rows} rows.")

# 根据不符合要求的光谱文件名找到对应的图像名
# 读取CSV文件
df = pd.read_csv("5w_data.csv")

# 创建空列表用于存储光谱名
ex_fits_names = []

# 遍历列表中的姓名
for name in ex_filenames:
    # 在CSV文件中查找对应的姓名
    row = df.loc[df["photo_name"] == name]
    # 提取
    ex_fits = row["fitsname"].values[0]
    ex_fits_names.append(ex_fits)
print(f'对应光谱数量：{len(ex_fits_names)}')

# 源文件夹和目标文件夹路径
source_image_folder = '5w_fits'
target_folder = 'ex_fits'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
# 遍历文件名列表
for file_name in ex_fits_names:
    source_path = os.path.join(source_image_folder, file_name)
    target_path = os.path.join(target_folder, file_name)
    # 检查源文件是否存在
    if not os.path.exists(source_path):
        print(f"文件 {source_path} 不存在，跳过剪切。")
        continue  # 跳过剪切，继续下一个文件
    shutil.move(source_path, target_path)

print("文件剪切完成")
