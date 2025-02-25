import numpy as np
from PIL import Image


class PixelTool:
    @staticmethod
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
