import io
import zipfile
import tarfile
import rarfile
from typing import Dict

class ArchiveExtractor:
    """
    处理压缩文件，不落地磁盘，直接解压缩并返回文件内容。
    """

    @staticmethod
    def extract_stream(file_stream: io.BytesIO) -> Dict[str, bytes]:
        """
        直接从 BytesIO 解压缩，并返回所有文件的字节数据。
        """
        extracted_files = {}

        # ZIP 处理
        if zipfile.is_zipfile(file_stream):
            with zipfile.ZipFile(file_stream, 'r') as zip_ref:
                for name in zip_ref.namelist():
                    with zip_ref.open(name) as f:
                        extracted_files[name] = f.read()

        # TAR 处理
        elif tarfile.is_tarfile(file_stream):
            with tarfile.open(fileobj=file_stream, mode='r:*') as tar_ref:
                for member in tar_ref.getmembers():
                    if member.isfile():
                        extracted_files[member.name] = tar_ref.extractfile(member).read()

        # RAR 处理
        elif file_stream.name.endswith(".rar"):
            with rarfile.RarFile(file_stream, 'r') as rar_ref:
                for name in rar_ref.namelist():
                    with rar_ref.open(name) as f:
                        extracted_files[name] = f.read()

        return extracted_files
