import os
import pydicom
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import List

@dataclass
class DicomToPngConverter:
    input_dir: str
    output_dir: str
    workers: int = 4
    dicom_files: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """初始化时创建输出目录，并查找所有 DICOM 文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.dicom_files = self._find_dicom_files()

    def _find_dicom_files(self) -> List[str]:
        """递归查找所有 DICOM 文件"""
        dicom_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                dicom_files.append(os.path.join(root, file))
        return dicom_files

    def _convert_single_file(self, dicom_path: str):
        """转换单个 DICOM 文件为 PNG"""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image_array = dicom_data.pixel_array.astype(np.float32)

            # 归一化处理
            image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
            image_array = (image_array * 255).astype(np.uint8)

            # 生成 PNG 输出路径，保持目录层级
            relative_path = os.path.relpath(dicom_path, self.input_dir)
            png_path = os.path.join(self.output_dir, relative_path) + ".png"
            os.makedirs(os.path.dirname(png_path), exist_ok=True)

            # 保存 PNG
            Image.fromarray(image_array).save(png_path)
            print(f"Converted: {dicom_path} -> {png_path}")

        except Exception as e:
            print(f"Failed to convert {dicom_path}: {e}")

    def convert_all(self):
        """批量转换所有 DICOM 文件"""
        print(f"Found {len(self.dicom_files)} DICOM files to convert.")
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            executor.map(self._convert_single_file, self.dicom_files)


# 示例用法
if __name__ == "__main__":
    input_directory = "DICOM"
    output_directory = "output"

    converter = DicomToPngConverter(input_directory, output_directory, workers=8)
    converter.convert_all()
