import base64
import io
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from bson import ObjectId
from flask import Flask, request, jsonify
from loguru import logger

from common.Result import Result
from unet.unet_process import process_stream
from utils import MaskProcessTool
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

    res = mysql_tool.execute_query('SELECT process_status FROM study WHERE id = %s;', (study_id,))
    for row in res['data']:
        for record in row:
            if record['process_status'] != '未导入':
                return Result.error('重复导入请先清空数据').to_response()

    # 直接使用 BytesIO 避免文件写入磁盘
    file_stream = io.BytesIO(file.read())

    # 处理压缩包，提取 DICOM 文件
    extracted_files = ArchiveExtractor.extract_stream(file_stream)

    if not extracted_files:
        logger.error('没有合法文件可以进行解压缩')
        return Result.error('没有合法文件可以进行解压缩').to_response()
    if 'data/dicom/' not in extracted_files.keys():
        return Result.error('dicom 目录缺失').to_response()
    if 'data/img_p/' not in extracted_files.keys():
        return Result.error('img_p 目录缺失').to_response()
    if 'data/img_v/' not in extracted_files.keys():
        return Result.error('img_v 目录缺失').to_response()

    # 去除多余的目录文件
    extracted_files = {k:v for k,v in extracted_files.items() if k not in ['data/', 'data/dicom/', 'data/img_p/', 'data/img_v/'] }

    dicom_files = {name:data for name, data in extracted_files.items() if name.startswith("data/dicom/")}
    img_p_files = {name:data for name, data in extracted_files.items() if name.startswith("data/img_p/")}
    img_v_files = {name:data for name, data in extracted_files.items() if name.startswith("data/img_v/")}

    if len(dicom_files) != len(img_p_files) | len(dicom_files) != len(img_v_files):
        return Result.error('3 个目录文件数量不一致').to_response()


    logger.info(f"Extracted {len(extracted_files)} files")

    converted_images = DicomToPngConverter.convert_stream(dicom_files)

    # test
    # DicomToPngConverter.convert_to_disk(dicom_files, 'test_dicom_output')
    # return Result.success().to_response()

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # 存入 MongoDB
    image_mongo_id_to_filename_dict = {}
    i = 1
    for image_name, image_bytes in sorted(converted_images.items(), key=lambda x: natural_sort_key(x[0])):
        path = Path(image_name)
        mongo_result = mongo_tool.insert_one({
            "study_id": study_id,
            "filename": f"{path.stem}_{i}{path.suffix}",
            "image_data": image_bytes
        })
        i += 1
        if mongo_result["success"]:
            image_mongo_id_to_filename_dict[mongo_result["inserted_id"]] = path.stem


    image_p_mongo_id_to_filename_dict = {}
    i = 1
    for image_name, image_bytes in img_p_files.items():
        path = Path(image_name)
        mongo_result = mongo_tool.insert_one({
            "study_id": study_id,
            "filename": f"{path.stem}_{i}{path.suffix}",
            "image_data": image_bytes
        })
        i += 1
        if mongo_result["success"]:
            image_p_mongo_id_to_filename_dict[mongo_result["inserted_id"]] = path.stem

    image_v_mongo_id_to_filename_dict = {}
    i = 1
    for image_name, image_bytes in img_v_files.items():
        path = Path(image_name)
        mongo_result = mongo_tool.insert_one({
            "study_id": study_id,
            "filename": f"{path.stem}_{i}{path.suffix}",
            "image_data": image_bytes
        })
        i += 1
        if mongo_result["success"]:
            image_v_mongo_id_to_filename_dict[mongo_result["inserted_id"]] = path.stem

    file_number = 1
    for dk, pk, vk in zip(image_mongo_id_to_filename_dict.keys(), image_p_mongo_id_to_filename_dict.keys(), image_v_mongo_id_to_filename_dict.keys()):
        mysql_tool.insert(
            "INSERT INTO file (study_id, file_name, file_number, image_mongo_id, image_p_mongo_id, image_v_mongo_id) VALUES (%s, %s, %s, %s, %s, %s);",
            (study_id, image_mongo_id_to_filename_dict[dk], file_number, dk, pk, vk)
        )
        file_number += 1


    # 更新状态
    mysql_tool.update(
        "update study set process_status = %s, file_num = %s where id = %s",
        ("已导入", len(dicom_files), study_id)
    )

    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('数据导入', 'IMPORT', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )

    # 关闭连接
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    logger.info(f"已存储 {len(extracted_files)} 张图片至 MongoDB 和 MySQL，执行时长：{round(end_time - start_time, 2)} 秒")
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
        tasks[task_id] = {"status": "error", "mark_images": "检查id 不存在"}
        return Result.error('检查id 不存在').to_response()

    # Retrieve files from MySQL using the study_id
    res = mysql_tool.execute_query('SELECT * FROM file WHERE study_id = %s;', (study_id,))
    print(res)
    if not 'data' in res.keys():
        return Result.error('暂无数据，请先进行数据导入').to_response()

    image_files = {}
    image_p_images = {}
    image_v_images = {}
    for row in res['data']:
        for record in row:
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['image_mongo_id'])})
            if result_doc["success"]:
                image_files[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['image_p_mongo_id'])})
            if result_doc["success"]:
                image_p_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['image_v_mongo_id'])})
            if result_doc["success"]:
                image_v_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]


    # Process the images using the UNet model
    mark_images = process_stream(image_files)

    # 对齐彩图与掩码图
    process_p_result_dict = MaskProcessTool.process_ct_images(ct_images=image_p_images, mask_images=mark_images)
    process_v_result_dict = MaskProcessTool.process_ct_images(ct_images=image_v_images, mask_images=mark_images)

    # Insert processed images into MongoDB with white pixel count calculated first
    mongo_id_to_filename_p_dict = {}
    mongo_id_to_filename_v_dict = {}
    file_number = 1
    for mk, pk, vk in zip(mark_images.keys(), process_p_result_dict.keys(), process_v_result_dict.keys()):
        # Write the image bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(mark_images[str(mk)])
            tmp_file_path = tmp_file.name

        # Calculate white pixel count using the temporary file
        white_pixel_count = PixelTool.calculate_white_pixel_count(tmp_file_path)
        os.remove(tmp_file_path)  # Remove the temporary file

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk,
            "image_data": process_p_result_dict[str(pk)],
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_p_dict[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk,
            "image_data": process_v_result_dict[str(vk)],
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_v_dict[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }
        file_number += 1

    # Insert the processed image details into MySQL, including the white pixel count
    pixel_sum = 0

    for mongo_id, info in mongo_id_to_filename_p_dict.items():
        mysql_tool.update(
            "update file set process_p_mongo_id = %s, pixel_num_p = %s where study_id = %s and file_number = %s;",
            (mongo_id, info["white_pixel_count"], study_id, info["file_number"])
        )
        pixel_sum += info["white_pixel_count"]

    mysql_tool.update(
        "update study set pixel_p_sum = %s where id = %s",
        (pixel_sum, study_id)
    )

    pixel_sum = 0
    for mongo_id, info in mongo_id_to_filename_v_dict.items():
        mysql_tool.update(
            "update file set process_v_mongo_id = %s, pixel_num_v = %s where study_id = %s and file_number = %s;",
            (mongo_id, info["white_pixel_count"], study_id, info["file_number"])
        )
        pixel_sum += info["white_pixel_count"]

    end_time = time.time()  # 记录结束时间
    mysql_tool.update(
        "update study set pixel_v_sum = %s, process_status = %s, execute_time = %s where id = %s",
        (pixel_sum, "已处理", round(end_time - start_time, 2), study_id)
    )


    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('图像处理', 'PROCESS', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )


    # Close database connections
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    tasks[task_id] = {"status": "finished", "mark_images": "处理完成"}
    logger.info(f"已处理 {len(mark_images)} 张图片，执行时长：{round(end_time - start_time, 2)} 秒")
    tasks[task_id] = {"status": "finished", "mark_images": "处理完成"}
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
    # 挑选平均分布的 2 个编号
    image_example_nums = 2
    res = mysql_tool.execute_query('SELECT * FROM file WHERE study_id = %s;', (study_id,))
    input_p_files = {}
    input_v_files = {}

    indices = []
    for row in res['data']:
        n = len(row)
        indices = np.linspace(60, n-131, image_example_nums, dtype=int)
        row = [row[i] for i in indices]
        for record in row:
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['process_p_mongo_id'])})
            if result_doc["success"]:
                input_p_files[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['process_v_mongo_id'])})
            if result_doc["success"]:
                input_v_files[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
    # test for save full output
    # os.makedirs('p_output', exist_ok=True)
    # for filename, image_data in input_p_files.items():
    #     with open(os.path.join('p_output', filename), 'wb') as f:
    #         f.write(image_data)
    # os.makedirs('v_output', exist_ok=True)
    # for filename, image_data in input_v_files.items():
    #     with open(os.path.join('v_output', filename), 'wb') as f:
    #         f.write(image_data)

    print(len(input_p_files))
    if len(input_p_files) < image_example_nums:
        return Result.error('暂无图像数据，请先进行图像处理').to_response()

    # 使用 BytesIO 存储 PDF 文件数据，而不是保存到磁盘
    pdf_output = generate_report(
        patient_name=patient_info['name'],
        age=patient_info['age'],
        gender="男" if patient_info['gender'] == "MALE" else "女",
        exam_date=study_info['study_date'],
        exam_number=study_info['id'],
        ventilation_perfusion_ratio=study_info['ventilation_perfusion_ratio'],
        description=study_info['description'],
        input_p_files=input_p_files,
        input_v_files=input_v_files,
        idx_arr=indices
    )

    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('报告生成', 'REPORT', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )

    # Close database connections
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    # 返回 PDF 内容作为响应
    # return jsonify({'pdf': pdf_output.getvalue().decode('latin1')})  # Use Latin1 for safe encoding
    return Result.success_with_data({'pdf': base64.b64encode(pdf_output.getvalue()).decode('utf-8')}).to_response()

def generate_report(patient_name, age, gender, exam_date, exam_number, ventilation_perfusion_ratio, description, input_p_files, input_v_files, idx_arr):
    """
    使用 Playwright 生成检查报告 PDF
    """

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
    base64_p_images = {}
    for i, (image_name, image_bytes) in enumerate(input_p_files.items(), 1):
        base64_p_images[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_v_images = {}
    for i, (image_name, image_bytes) in enumerate(input_v_files.items(), 1):
        base64_v_images[f'image{i}'] = encode_image_to_base64(image_bytes)
    logo_base64 = encode_file_to_base64("assets/images/吉大二院logo.jpg")
    # HTML模板字符串
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
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
                <p>检查编号：{exam_number}</p>
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
                <p>通气血流比：{ventilation_perfusion_ratio}</p>
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
                <p>临床诊断：{description}</p>
            </div>
        </div>
        <hr>
        <div class="row flex1">
            <div class="col-12">
                <p>血灌注检查示例：</p>
            </div>
        </div>
        <div class="row flex2">
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_p_images['image1']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[0] + 1}/385 冠状面</p>
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_p_images['image2']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[1] + 1}/385 冠状面</p>
            </div>
        </div>

        
        <div class="row flex1">
            <div class="col-12">
                <p>通气检查示例：</p>
            </div>
        </div>
        <div class="row flex2">
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_v_images['image1']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[0] + 1}/385 冠状面</p>
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_v_images['image2']}" style="width: 400px; height: 300px;margin-top:50px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[1] + 1}/385 冠状面</p>
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

@app.route('/lvs-py-api/data_clear', methods=['POST'])
def data_clear():
    study_id = request.get_json().get('studyId')
    if not study_id:
        return Result.error('检查id 不存在').to_response()

    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")
    mongo_tool.delete_many({"study_id": str(study_id)})
    # 删除 mysql file 记录, 检查记录像素值归零
    mysql_tool.delete("delete from file where study_id = %s", (study_id,))
    mysql_tool.update("update study set pixel_p_sum = 0, pixel_v_sum = 0, process_status = '未导入', file_num = 0, execute_time = 0 where id = %s", (study_id,))

    mongo_tool.close_connection()
    mysql_tool.close_connection()

    return Result.success().to_response()

if __name__ == '__main__':
    app.run()