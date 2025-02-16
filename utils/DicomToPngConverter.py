import io
import os
import cv2
import pydicom
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

class DicomToPngConverter:
    """
    处理 DICOM 转 PNG，使用流式方式存储，不落地磁盘。
    """

    @staticmethod
    def convert_stream(dicom_files: Dict[str, bytes]) -> Dict[str, bytes]:
        """
        将 DICOM 文件转换为 PNG，并返回字节数据。

        :param dicom_files: 包含 DICOM 文件数据的字典，键为文件名，值为文件的字节数据。
        :return: 一个字典，键为转换后的 PNG 文件名，值为 PNG 文件的字节数据。
        """
        converted_images = {}

        def process_single_file(file_name, file_bytes):
            try:
                dicom_data = pydicom.dcmread(io.BytesIO(file_bytes), force=True)

                # 检查 Pixel Data 是否存在
                if not hasattr(dicom_data, "PixelData"):
                    print(f"Warning: {file_name} has no Pixel Data.")
                    return

                # 解码 Pixel Data
                dicom_data.decompress()

                # 读取像素数据并转换为浮点型
                image_array = dicom_data.pixel_array.astype(np.float32)

                # 若存在 RescaleSlope 和 RescaleIntercept，则进行重标定
                if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                    image_array = image_array * float(dicom_data.RescaleSlope) + float(dicom_data.RescaleIntercept)

                # 针对 MONOCHROME1 类型的图像进行反转（MONOCHROME1 的高像素值代表暗部）
                if hasattr(dicom_data, 'PhotometricInterpretation') and dicom_data.PhotometricInterpretation == "MONOCHROME1":
                    image_array = np.max(image_array) - image_array

                # 归一化处理
                min_val = np.min(image_array)
                max_val = np.max(image_array)
                if max_val - min_val != 0:
                    image_array = (image_array - min_val) / (max_val - min_val)
                image_array = (image_array * 255).astype(np.uint8)

                # 转换为 PNG
                img_stream = io.BytesIO()
                Image.fromarray(image_array).save(img_stream, format="PNG")
                converted_images[file_name + ".png"] = img_stream.getvalue()

            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")

        # 并发处理
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(lambda item: process_single_file(*item), dicom_files.items())


        return converted_images
    @staticmethod
    def convert_to_disk(dicom_files: Dict[str, bytes], output_dir: str) -> None:
        """
        将 DICOM 文件转换为 PNG，并将结果直接保存到指定的磁盘目录中。

        :param dicom_files: 包含 DICOM 文件数据的字典，键为文件名，值为文件的字节数据。
        :param output_dir: 输出目录路径，转换后的 PNG 文件将保存在此目录中。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def process_single_file(file_name, file_bytes):
            try:
                dicom_data = pydicom.dcmread(io.BytesIO(file_bytes), force=True)

                if not hasattr(dicom_data, "PixelData"):
                    print(f"Warning: {file_name} has no Pixel Data.")
                    return

                dicom_data.decompress()
                image_array = dicom_data.pixel_array.astype(np.float32)

                if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                    image_array = image_array * float(dicom_data.RescaleSlope) + float(dicom_data.RescaleIntercept)

                if hasattr(dicom_data, 'PhotometricInterpretation') and dicom_data.PhotometricInterpretation == "MONOCHROME1":
                    image_array = np.max(image_array) - image_array

                min_val = np.min(image_array)
                max_val = np.max(image_array)
                if max_val - min_val != 0:
                    image_array = (image_array - min_val) / (max_val - min_val)
                image_array = (image_array * 255).astype(np.uint8)

                output_file = os.path.join(output_dir, file_name.replace("\\", "").replace("/", "") + ".png")
                Image.fromarray(image_array).save(output_file, format="PNG")

            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(lambda item: process_single_file(*item), dicom_files.items())
