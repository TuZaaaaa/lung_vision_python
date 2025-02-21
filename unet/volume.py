from PIL import Image
import numpy as np
import os
import pandas as pd

def calculate_white_pixel_count(image_path):
    # 读取图像
    image = Image.open(image_path)
    # 将图像转换为二值模式（1表示白色，0表示黑色）
    image = image.convert('1')
    # 将图像转换为NumPy数组
    image_array = np.array(image)
    # 计算白色像素的数量（非零像素表示白色区域）
    white_pixel_count = np.count_nonzero(image_array)
    return white_pixel_count

def process_images(image_folder, output_file):
    file_names = []
    white_pixel_counts = []

    for file_name in os.listdir(image_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            white_pixel_count = calculate_white_pixel_count(file_path)
            file_names.append(file_name)
            white_pixel_counts.append(white_pixel_count)

    # 创建DataFrame
    data = {
        'File Name': file_names,
        'White Pixel Count': white_pixel_counts
    }
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    # 设置图像文件夹路径
    image_folder = "input" # 替换为您的图像文件夹路径
    # 设置输出Excel文件路径
    output_file = "output/output.xlsx"   # 替换为您希望保存结果的路径

    # 执行图像处理和白色像素点数计算
    process_images(image_folder, output_file)

if __name__ == "__main__":
    main()