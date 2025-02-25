import io
import os
import time
import tempfile
from bson import ObjectId
from flask import Flask, request, jsonify
from loguru import logger

from common.Result import Result
from unet.unet_process import process_stream
from utils.ArchiveExtractor import ArchiveExtractor
from utils.DicomToPngConverter import DicomToPngConverter
from utils.MongoDBTool import MongoDBTool
from utils.MySQLTool import MySQLTool
from utils.PixelTool import PixelTool

app = Flask(__name__)

# 配置日志
logger.add("app.log", rotation="500 MB", retention="10 days", compression="zip")


@app.route('/lvs-py-api/upload', methods=['POST'])
def upload_file():
    start_time = time.time()  # 记录开始时间
    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")
    """ 处理上传的 DICOM 文件压缩包 """
    logger.info("Received file upload request")

    file = request.files.get('file')
    study_id = request.form.get('studyId')

    if not file:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    # 直接使用 BytesIO 避免文件写入磁盘
    file_stream = io.BytesIO(file.read())

    # 处理压缩包，提取 DICOM 文件
    extracted_files = ArchiveExtractor.extract_stream(file_stream)

    if not extracted_files:
        logger.error('没有合法文件可以进行压缩')
        return Result.error('没有合法文件可以进行压缩').to_response()

    logger.info(f"Extracted {len(extracted_files)} files")

    # 转换 DICOM 文件
    converted_images = DicomToPngConverter.convert_stream(extracted_files)

    # test
    # DicomToPngConverter.convert_to_disk(extracted_files, 'test_dicom_output')
    # return Result.success().to_response()

    # 存入 MongoDB
    mongo_id_to_filename_dict = {}
    for image_name, image_bytes in converted_images.items():
        mongo_result = mongo_tool.insert_one({
            "study_id": study_id,
            "filename": image_name,
            "image_data": image_bytes
        })
        if mongo_result["success"]:
            mongo_id_to_filename_dict[mongo_result["inserted_id"]] = image_name

    # 存储 ID 到 MySQL
    for mongo_id, image_name in mongo_id_to_filename_dict.items():
        mysql_tool.insert(
            "INSERT INTO file (study_id, file_name, image_mongo_id) VALUES (%s, %s, %s);",
            (study_id, image_name, mongo_id)
        )


    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('数据导入', 'IMPORT', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )

    # 关闭连接
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    logger.info(f"已存储 {len(mongo_id_to_filename_dict)} 张图片至 MongoDB 和 MySQL，执行时长：{round(end_time - start_time, 2)} 秒")
    return Result.success().to_response()

@app.route('/lvs-py-api/image_process', methods=['POST'])
def image_process():
    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")

    start_time = time.time()  # 记录开始时间
    json_data = request.get_json()

    study_id = json_data.get('studyId')
    if not study_id:
        return Result.error('检查id 不存在').to_response()

    # Retrieve files from MySQL using the study_id
    res = mysql_tool.execute_query('SELECT * FROM file WHERE study_id = %s;', (study_id,))

    input_files = {}
    print(res)
    if not 'data' in res.keys():
        return Result.error('暂无数据，请先进行数据导入').to_response()

    for row in res['data']:
        for record in row:
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['image_mongo_id'])})
            if result_doc["success"]:
                # Use a sanitized filename as key
                input_files[result_doc["data"]["filename"].replace("/", "")] = result_doc["data"]["image_data"]

    # Process the images using the UNet model
    result = process_stream(input_files)

    # Insert processed images into MongoDB with white pixel count calculated first
    mongo_id_to_filename_dict = {}
    for image_name, image_bytes in result.items():
        # Write the image bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_bytes)
            tmp_file_path = tmp_file.name

        # Calculate white pixel count using the temporary file
        white_pixel_count = PixelTool.calculate_white_pixel_count(tmp_file_path)
        os.remove(tmp_file_path)  # Remove the temporary file

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": study_id,
            "filename": image_name,
            "image_data": image_bytes,
            "white_pixel_count": white_pixel_count
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_dict[mongo_result["inserted_id"]] = {
                "filename": image_name,
                "white_pixel_count": white_pixel_count
            }

    # Insert the processed image details into MySQL, including the white pixel count
    pixel_sum = 0
    for mongo_id, info in mongo_id_to_filename_dict.items():
        mysql_tool.update(
            "update file set process_mongo_id = %s, pixel_num = %s where study_id = %s;",
            (mongo_id, info["white_pixel_count"], study_id)
        )
        pixel_sum += info["white_pixel_count"]

    mysql_tool.update(
        "update study set pixel_sum = %s where id = %s",
        (pixel_sum, study_id)
    )


    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('图像处理', 'PROCESS', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )


    # Close database connections
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    logger.info(f"已处理 {len(result)} 张图片，执行时长：{round(end_time - start_time, 2)} 秒")

    return Result.success().to_response()


if __name__ == '__main__':
    app.run()
