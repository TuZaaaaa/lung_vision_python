import base64
import io
import os
import tempfile
import threading
import time
import numpy as np
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
from playwright.sync_api import sync_playwright
from io import BytesIO
from PIL import Image
app = Flask(__name__)

# 配置日志
logger.add("app.log", rotation="500 MB", retention="10 days", compression="zip")

tasks = {}

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
def create_task():
    task_id = str(len(tasks) + 1)  # 任务 ID
    tasks[task_id] = {"status": "processing", "result": None}
    # 启动后台线程
    thread = threading.Thread(target=image_process, args=(request.get_json(), task_id))
    thread.start()
    return Result.success_with_data({"task_id": task_id, "status": "processing" ,"message": "处理中"}).to_response()
@app.route('/lvs-py-api/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = tasks[task_id]
    if task["status"] == "processing":
        return Result.success_with_data({"task_id": task_id, "status": "processing" ,"message": "处理中"}).to_response()
    elif task["status"] == "finished":
        return Result.success_with_data({"task_id": task_id, "status": "finished" ,"message": "处理完成"}).to_response()
    else:
        return Result.error("任务不存在").to_response()
def image_process(json_data, task_id):
    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")

    start_time = time.time()  # 记录开始时间

    study_id = json_data.get('studyId')
    if not study_id:
        tasks[task_id] = {"status": "error", "result": "检查id 不存在"}
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

    tasks[task_id] = {"status": "finished", "result": "处理完成"}
    logger.info(f"已处理 {len(result)} 张图片，执行时长：{round(end_time - start_time, 2)} 秒")
    tasks[task_id] = {"status": "finished", "result": "处理完成"}
    return


@app.route('/lvs-py-api/report_generate', methods=['POST'])
def report_generate():
    study_id = request.get_json().get('studyId')
    if not study_id:
        return Result.error('检查id 不存在').to_response()

    start_time = time.time()  # 记录结束时间
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

    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('报告生成', 'PROCESS', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )

    # Close database connections
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    # 返回 PDF 内容作为响应
    # return jsonify({'pdf': pdf_output.getvalue().decode('latin1')})  # Use Latin1 for safe encoding
    return Result.success_with_data({'pdf': base64.b64encode(pdf_output.getvalue()).decode('utf-8')}).to_response()

def generate_report(patient_name, age, gender, exam_date, exam_number, pixel_sum, input_files):
    """
    使用 Playwright 生成检查报告 PDF
    """

    if len(input_files) != 4:
        raise ValueError("必须提供4个检查结果图片的字节数据。")

    # 获取当前时间
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 将字节数据转换为base64编码的字符串
    def encode_image_to_base64(image_bytes):
        image_stream = BytesIO(image_bytes)
        pil_img = Image.open(image_stream)

        # 将图片保存到内存并转换为Base64编码
        temp_io = BytesIO()
        pil_img.save(temp_io, format='PNG')
        temp_io.seek(0)

        # 转换为Base64编码字符串
        encoded_string = base64.b64encode(temp_io.read()).decode('utf-8')
        return encoded_string

    def encode_file_to_base64(file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode('utf-8')
    # 获取所有图片的Base64编码
    base64_images = {}
    for i, (image_name, image_bytes) in enumerate(input_files.items(), 1):
        base64_images[f'image{i}'] = encode_image_to_base64(image_bytes)
    logo_base64 = encode_file_to_base64("assets/images/吉大二院logo.jpg")
    # HTML模板字符串
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <base href="file:///D:/Python/PycharmProjects/lung_vision_python/">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.bootcdn.net/ajax/libs/twitter-bootstrap/4.5.2/css/bootstrap.min.css">

    <title>Document</title>
    <style>
        body {{
            width: 1000px;
            height: 800px;
            margin: 0 auto;
        }}
        p {{
            font-size: 20px;
        }}
        img {{
            margin: 0 auto;
        }}
        .study-img {{
            width: 70%;
            height: 100%;
        }}
        .container {{
            width: 100%;
            height: 100%;
            margin: 0 auto;
        }}
        .row {{
            text-align: center;
            position: relative;
            top: 30px;
        }}  
        .flex1 {{
            display: flex;
            justify-content: space-between;
            width: 100%;
        }} 
        .flex2 {{
            display: flex;
            justify-content: space-around;
            width: 100%;
        }} 
        .col-12 {{
            text-align: left;
        }}
        .col-6, .col-4 {{
            flex: 1;
            text-align: left;
        }}
        .title-large {{
            font-family: "宋体", SimSun, serif;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: -5px;
        }}
        .title-medium {{
            font-family: "宋体", SimSun, serif;
            font-size: 25px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <div class="col-12 text-center">
                <p class="title-large">吉林大学白求恩第二医院</p>
            </div>
        </div>
        <div class="row">
            <div class="col-12 text-center">
                <p class="title-medium">病理诊断报告</p>
            </div>
        </div>
        <img src="data:image/jpeg;base64,{logo_base64}" alt="吉大二院logo" style="position: relative; width: 100px; height: 100px; bottom: 11%; left: 12%;">
        <div class="row flex1">
            <div class="col-6">
                <p>检查编号:{exam_number}</p>
            </div>
            <div class="col-6">
                <p>申请科室：重症监护室</p>
            </div>
        </div>
        <hr>
        <div class="row flex1">
            <div class="col-4">
                <p>姓名：{patient_name}</p>
            </div>
            <div class="col-4">
                <p>性别：{gender}</p>
            </div>
            <div class="col-4">
                <p>年龄：{age}</p>
            </div>
        </div>
        <div class="row flex1">
            <div class="col-4">
                <p>像素总数：{pixel_sum}</p>
            </div>
            <div class="col-4">
                <p>检查日期：{exam_date}</p>
            </div>
            <div class="col-4">
                <p>生成日期：{now}</p>
            </div>
        </div>
        <hr>
        <div class="row flex1">
            <div class="col-12">
                <p>临床诊断：活得很好</p>
            </div>
        </div>
        <hr>
        <div class="row flex1">
            <div class="col-12">
                <p>检查示例：</p>
            </div>
        </div>
        <div class="row flex2">
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_images['image1']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_images['image2']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
            </div>
        </div>
        <div class="row flex2">
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_images['image3']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_images['image4']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
            </div>
        </div>
    </div>
</body>
</html>
"""

    # 使用 Playwright 将 HTML 转换为 PDF
    buffer = BytesIO()
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(args=["--allow-file-access-from-files"])
        # 创建一个新的页面
        page = browser.new_page()
        # 设置页面内容
        page.set_content(html_template)
        # 等待页面加载完成
        page.wait_for_load_state('networkidle')

        pdf_data = page.pdf(format='A4', margin={'top': '20px', 'bottom': '20px'})
        # 将数据写入 BytesIO 对象
        buffer = BytesIO(pdf_data)
        # 关闭浏览器
        browser.close()

    # 将缓冲区的内容返回
    buffer.seek(0)
    return buffer


if __name__ == '__main__':
    app.run()