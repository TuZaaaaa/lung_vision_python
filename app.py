import base64
import io
import os
import tempfile
import time
from datetime import datetime

import numpy as np
from PIL import Image as PILImage
from bson import ObjectId
from flask import Flask, request, jsonify
from loguru import logger
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PlatypusImage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

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

    file_number = 1
    # 存储 ID 到 MySQL
    for mongo_id, image_name in mongo_id_to_filename_dict.items():
        mysql_tool.insert(
            "INSERT INTO file (study_id, file_name, file_number, image_mongo_id) VALUES (%s, %s, %s, %s);",
            (study_id, image_name.replace("/", ""), file_number, mongo_id)
        )
        file_number += 1


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
            "update file set process_mongo_id = %s, pixel_num = %s where study_id = %s and file_name = %s;",
            (mongo_id, info["white_pixel_count"], study_id, info["filename"])
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

@app.route('/lvs-py-api/report_generate', methods=['POST'])
def report_generate():
    study_id = request.get_json().get('studyId')
    if not study_id:
        return Result.error('检查id 不存在').to_response()

    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")

    res = mysql_tool.execute_query('SELECT * FROM study WHERE id = %s;', (study_id,))
    if not 'data' in res.keys():
        return Result.error('暂无数据，请先进行数据导入').to_response()
    study_info = res['data'][0][0]
    print(study_info)
    patient_id = study_info['patient_id']
    res = mysql_tool.execute_query('SELECT * FROM patient WHERE id = %s;', (patient_id,))
    if not 'data' in res.keys():
        return Result.error('此检查记录无相关联的患者').to_response()
    patient_info = res['data'][0][0]
    print(patient_info)

    # 选取示例图片
    image_example_nums = 4
    res = mysql_tool.execute_query('SELECT * FROM file WHERE study_id = %s;', (study_id,))
    input_files = {}
    for row in res['data']:
        # 挑选平均分布的 4 个编号
        n = len(row)
        indices = np.linspace(0, n-1, image_example_nums, dtype=int)
        row = [row[i] for i in indices]
        for record in row:
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['process_mongo_id'])})
            if result_doc["success"]:
                input_files[result_doc["data"]["filename"].replace("/", "")] = result_doc["data"]["image_data"]

    if len(input_files) < image_example_nums:
        return Result.error('暂无图像数据，请先进行图像处理').to_response()

    # 使用 BytesIO 存储 PDF 文件数据，而不是保存到磁盘
    pdf_output = generate_report(
        patient_name=patient_info['name'],
        age=patient_info['age'],
        gender="男" if patient_info['gender'] == "MALE" else "女",
        exam_date=study_info['study_date'],
        exam_number=study_info['id'],
        pixel_sum=study_info['pixel_sum'],
        input_files=input_files
    )

    # 返回 PDF 内容作为响应
    # return jsonify({'pdf': pdf_output.getvalue().decode('latin1')})  # Use Latin1 for safe encoding
    return Result.success_with_data({'pdf': base64.b64encode(pdf_output.getvalue()).decode('utf-8')}).to_response()


def generate_report(patient_name, age, gender, exam_date, exam_number, pixel_sum, input_files):
    """
    生成检查报告 PDF
    """
    pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttc'))
    styles = getSampleStyleSheet()
    chinese_style = ParagraphStyle(
        'Chinese', parent=styles['Normal'], fontName='SimSun', fontSize=12, leading=15,
    )

    if len(input_files) != 4:
        raise ValueError("必须提供4个检查结果图片的字节数据。")

    # Use BytesIO to create an in-memory PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    title_style = ParagraphStyle(
        'TitleChinese', parent=styles['Title'], fontName='SimSun', fontSize=24, leading=28,
    )
    elements.append(Paragraph("医疗检查报告", title_style))
    elements.append(Spacer(1, 12))

    info_text = (
        f"报告生成日期：{report_date}<br/><br/>"
        f"姓名：{patient_name}<br/>"
        f"年龄：{age}<br/>"
        f"性别：{gender}<br/><br/>"
        f"检查日期：{exam_date}<br/>"
        f"检查编号：{exam_number}<br/>"
        f"像素总数：{pixel_sum}<br/><br/>"
        f"检查示例：<br/>"
    )
    elements.append(Paragraph(info_text, chinese_style))
    elements.append(Spacer(1, 12))

    # 创建一个表格数据列表，每行两个图像
    table_data = []

    row = []
    for image_name, image_bytes in input_files.items():
        image_stream = io.BytesIO(image_bytes)
        pil_img = PILImage.open(image_stream)

        temp_io = io.BytesIO()
        pil_img.save(temp_io, format='PNG')
        temp_io.seek(0)

        img = PlatypusImage(temp_io, width=200, height=150)
        row.append(img)

        if len(row) == 2:
            table_data.append(row)
            row = []

    if row:
        table_data.append(row)

    row_heights = [160] * len(table_data)
    col_widths = [220, 220]

    table = Table(
        table_data,
        colWidths=col_widths,
        rowHeights=row_heights,
    )

    table.setStyle(TableStyle([
        ('PAD', (0, 0), (-1, -1), 10),
    ]))

    elements.append(table)
    doc.build(elements)

    # Rewind the buffer to the beginning of the BytesIO object
    buffer.seek(0)
    return buffer

if __name__ == '__main__':
    app.run()
