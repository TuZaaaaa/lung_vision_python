import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from loguru import logger
import numpy as np
import pydicom
from PIL import Image


class DicomToPngConverter:
    """
    处理 DICOM 转 PNG，支持流式转换以及从目录直接读取 DICOM 文件后转换保存到新目录。
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

                # 清理文件名
                clean_name = os.path.basename(file_name)

                # 检查 Pixel Data
                if not hasattr(dicom_data, "PixelData"):
                    logger.warning(f"Skipped {clean_name}: No Pixel Data")
                    return

                # 仅在需要时解压
                try:
                    if not dicom_data.is_decompressed:
                        dicom_data.decompress()
                except Exception as e:
                    if "already uncompressed" in str(e):
                        pass
                    else:
                        raise

                # 读取像素数据并转换为浮点型
                image_array = dicom_data.pixel_array.astype(np.float32)

                # 重排序
                instance_number = dicom_data.get("InstanceNumber")
                file_name = str(instance_number)

                # 若存在 RescaleSlope 和 RescaleIntercept，则进行重标定
                if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                    image_array = image_array * float(dicom_data.RescaleSlope) + float(dicom_data.RescaleIntercept)

                # 针对 MONOCHROME1 类型的图像进行反转（高像素值代表暗部）
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

    @staticmethod
    def convert_directory(input_dir: str, output_dir: str) -> None:
        """
        从输入目录读取 DICOM 文件，并将其转换为 PNG，保存到输出目录中。

        :param input_dir: 包含未转换 DICOM 文件的目录路径。
        :param output_dir: 输出目录路径，转换后的 PNG 文件将保存在此目录中。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def process_file(file_name: str):
            file_path = os.path.join(input_dir, file_name)
            # 如果不是文件则跳过（例如目录）
            if not os.path.isfile(file_path):
                return
            try:
                # 直接使用文件路径读取 DICOM 文件
                dicom_data = pydicom.dcmread(file_path, force=True)

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

                # 构造输出文件名，去掉原文件扩展名，并追加 .png
                output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".png")
                Image.fromarray(image_array).save(output_file, format="PNG")
                print(f"Converted {file_name} to {output_file}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")

        # 获取目录下所有文件列表
        file_list = os.listdir(input_dir)
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(process_file, file_list)
