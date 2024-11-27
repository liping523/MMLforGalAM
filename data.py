'''
import pandas as pd
import os
import shutil

# 读取CSV文件
csv_file_path = '../10w_data.csv'  
df = pd.read_csv(csv_file_path)
successful_rows = 0

# 设置文件夹路径
photo_folder = '../photo'  # 图像文件夹
spec_folder = '../spec'  # 光谱文件文件夹
images_folder = 'images'  # 新的图像文件夹
fits_folder = 'fits'  # 新的光谱文件文件夹

# 确保目标文件夹存在
os.makedirs(images_folder, exist_ok=True)
os.makedirs(fits_folder, exist_ok=True)

# 新的CSV文件路径
new_csv_file_path = 'new_images_and_fits.csv'

# 写入列名（只在文件不存在时写入）
if not os.path.isfile(new_csv_file_path):
    df.to_csv(new_csv_file_path, mode='a', index=False)

# 遍历CSV文件中的每一行
for index, row in df.iterrows():
    photo_name = row['photo_name']
    fits_name = row['fitsname']

    # 检查图像文件和光谱文件是否存在
    if os.path.isfile(os.path.join(photo_folder, photo_name)) and os.path.isfile(os.path.join(spec_folder, fits_name)):
        # 复制图像文件到新的文件夹
        shutil.copy(os.path.join(photo_folder, photo_name), os.path.join(images_folder, photo_name))
        # 复制光谱文件到新的文件夹
        shutil.copy(os.path.join(spec_folder, fits_name), os.path.join(fits_folder, fits_name))
        # 将这一行保存到新的CSV文件中
        row.to_csv(new_csv_file_path, mode='a', header=False, index=False)
        successful_rows += 1  

print(f"Successfully data {successful_rows}.")
'''
import os
import pandas as pd

# 定义文件夹路径和CSV文件路径
folder_path = '0_5w_fits'
output_csv_path = '3w_data.csv'
not_found_list = []
csv_file_path = '../10w_data.csv' 
df = pd.read_csv(csv_file_path, low_memory=False)

# 遍历文件夹中的文件名
for filename in os.listdir(folder_path):
    # 检查文件名是否在10w_data.csv的fitsname列中
    if filename in df['fitsname'].values:
        # 如果找到，复制这一行的内容到新的csv文件中
        matched_rows = df[df['fitsname'] == filename]
        with open(output_csv_path, 'a') as f:
            matched_rows.to_csv(f, index=False, header=f.tell() == 0)
    else:
        # 如果没有找到，将文件名添加到数组中
        not_found_list.append(filename)

# 将未找到的文件名写入一个新的csv文件
not_found_csv_path = '3w_not_found.csv'
pd.DataFrame(not_found_list, columns=['filename']).to_csv(not_found_csv_path, index=False)

