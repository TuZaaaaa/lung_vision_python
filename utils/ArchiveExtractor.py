import os
import zipfile
import tarfile
import rarfile

class ArchiveExtractor:
    """
    解压缩工具类，支持 ZIP、TAR.GZ、RAR 格式的解压。
    """

    @staticmethod
    def extract_zip(file_path, extract_to):
        """
        解压 ZIP 文件。
        :param file_path: ZIP 文件路径
        :param extract_to: 解压目标目录
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"ZIP 文件已解压到 {extract_to}")

    @staticmethod
    def extract_tar_gz(file_path, extract_to):
        """
        解压 TAR.GZ 文件。
        :param file_path: TAR.GZ 文件路径
        :param extract_to: 解压目标目录
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"TAR.GZ 文件已解压到 {extract_to}")

    @staticmethod
    def extract_rar(file_path, extract_to):
        """
        解压 RAR 文件。
        :param file_path: RAR 文件路径
        :param extract_to: 解压目标目录
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(extract_to)
        print(f"RAR 文件已解压到 {extract_to}")

    @staticmethod
    def extract(file_path, extract_to):
        """
        自动识别文件格式并解压。
        :param file_path: 压缩文件路径
        :param extract_to: 解压目标目录
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        if file_path.endswith('.zip'):
            ArchiveExtractor.extract_zip(file_path, extract_to)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            ArchiveExtractor.extract_tar_gz(file_path, extract_to)
        elif file_path.endswith('.rar'):
            ArchiveExtractor.extract_rar(file_path, extract_to)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")


# 示例用法
if __name__ == "__main__":
    # 解压 ZIP 文件
    ArchiveExtractor.extract('example.zip', 'extracted_files')

    # 解压 TAR.GZ 文件
    ArchiveExtractor.extract('example.tar.gz', 'extracted_files')

    # 解压 RAR 文件
    ArchiveExtractor.extract('example.rar', 'extracted_files')

    # 自动识别文件格式并解压
    ArchiveExtractor.extract('example.zip', 'extracted_files')
