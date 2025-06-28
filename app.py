import base64
import io
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
import torch
import torchvision
from utils.VQAnalyzer import VQAnalyzer
import sys
import os

os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"

if getattr(sys, 'frozen', False):
    # 打包后的临时路径
    os.environ['PATH'] = os.path.join(sys._MEIPASS, 'torch', 'lib') + ';' + os.environ['PATH']

app = Flask(__name__)

# 配置日志
logger.add("app.log", rotation="500 MB", retention="10 days", compression="zip")
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

tasks = {}


@app.route('/lvs-py-api/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")

    logger.info("Received file upload request")

    file = request.files.get('file')
    study_id = request.form.get('studyId')

    if not file:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    res = mysql_tool.execute_query('SELECT process_status FROM study WHERE id = %s;', (study_id,))
    if 'data' in res.keys() and res['data']:
        for row in res['data']:
            for record in row:
                if record['process_status'] != '未导入':
                    return Result.error('重复导入请先清空数据').to_response()

    # 处理文件流
    file_stream = io.BytesIO(file.read())
    extracted_files = ArchiveExtractor.extract_stream(file_stream)

    if not extracted_files:
        logger.error('没有合法文件可以进行解压缩')
        return Result.error('没有合法文件可以进行解压缩').to_response()

    # 验证所有必要目录存在
    required_dirs = [
        'data/dicom_coronal/', 'data/dicom_sagittal/', 'data/dicom_axial/',
            'data/img_p_coronal/', 'data/img_p_sagittal/', 'data/img_p_axial/',
        'data/img_v_coronal/', 'data/img_v_sagittal/', 'data/img_v_axial/'
    ]

    for dir_path in required_dirs:
        if not any(k.startswith(dir_path) for k in extracted_files.keys()):
            return Result.error(f'{dir_path} 目录缺失').to_response()

    # 分离不同切面的文件
    dicom_coronal_files = {name: data for name, data in extracted_files.items() if
                           name.startswith("data/dicom_coronal/") and data != b'' }
    dicom_sagittal_files = {name: data for name, data in extracted_files.items() if
                            name.startswith("data/dicom_sagittal/") and data != b''}
    dicom_axial_files = {name: data for name, data in extracted_files.items() if name.startswith("data/dicom_axial/") and data != b''}
    img_p_coronal_files = {name: data for name, data in extracted_files.items() if
                           name.startswith("data/img_p_coronal/") and data != b''}
    img_p_sagittal_files = {name: data for name, data in extracted_files.items() if
                            name.startswith("data/img_p_sagittal/") and data != b''}
    img_p_axial_files = {name: data for name, data in extracted_files.items() if name.startswith("data/img_p_axial/") and data != b''}
    img_v_coronal_files = {name: data for name, data in extracted_files.items() if
                           name.startswith("data/img_v_coronal/") and data != b''}
    img_v_sagittal_files = {name: data for name, data in extracted_files.items() if
                            name.startswith("data/img_v_sagittal/") and data != b''}
    img_v_axial_files = {name: data for name, data in extracted_files.items() if name.startswith("data/img_v_axial/") and data != b''}

    # 定义自然排序函数
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # 定义处理每个切面的函数
    def process_orientation(orientation, dicom_files, img_p_files, img_v_files):
        if orientation == 'r':
            converted_images = DicomToPngConverter.convert_stream(dicom_files)
        else:
            converted_images = dicom_files

        # 排序所有图像
        sorted_dicom = sorted(converted_images.items(), key=lambda x: natural_sort_key(x[0]))
        sorted_img_p = sorted(img_p_files.items(), key=lambda x: natural_sort_key(x[0]))
        sorted_img_v = sorted(img_v_files.items(), key=lambda x: natural_sort_key(x[0]))

        # 检查数量一致性
        n = len(sorted_dicom)
        if len(sorted_img_p) != n or len(sorted_img_v) != n:
            raise ValueError(
                f"{orientation}方向文件数量不一致: DICOM={n}, P={len(sorted_img_p)}, V={len(sorted_img_v)}")

        file_records = []
        file_number = 1

        for (dicom_name, dicom_data), (img_p_name, img_p_data), (img_v_name, img_v_data) in zip(
                sorted_dicom, sorted_img_p, sorted_img_v
        ):
            # 存储DICOM图像
            dicom_path = Path(dicom_name)
            dicom_result = mongo_tool.insert_one({
                "study_id": study_id,
                "filename": f"{dicom_path.stem}_{file_number}{dicom_path.suffix}",
                "image_data": dicom_data,
                "orientation": orientation,
                "type": "original"
            })

            # 存储P图像
            p_path = Path(img_p_name)
            p_result = mongo_tool.insert_one({
                "study_id": study_id,
                "filename": f"{p_path.stem}_{file_number}{p_path.suffix}",
                "image_data": img_p_data,
                "orientation": orientation,
                "type": "perfusion"
            })

            # 存储V图像
            v_path = Path(img_v_name)
            v_result = mongo_tool.insert_one({
                "study_id": study_id,
                "filename": f"{v_path.stem}_{file_number}{v_path.suffix}",
                "image_data": img_v_data,
                "orientation": orientation,
                "type": "ventilation"
            })

            # 检查插入结果
            if not dicom_result["success"] or not p_result["success"] or not v_result["success"]:
                raise RuntimeError(f"存储{orientation}方向图像失败")

            # 插入MySQL记录
            mysql_result = mysql_tool.insert(
                "INSERT INTO file (study_id, orientation, file_number, original_image_id, perfusion_image_id, ventilation_image_id) VALUES (%s, %s, %s, %s, %s, %s);",
                (study_id, orientation, file_number,
                 dicom_result["inserted_id"],
                 p_result["inserted_id"],
                 v_result["inserted_id"])
            )

            if not mysql_result.get("success"):
                raise RuntimeError(f"存储{orientation}方向元数据失败")

            file_records.append({
                "file_number": file_number,
                "original_id": dicom_result["inserted_id"],
                "perfusion_id": p_result["inserted_id"],
                "ventilation_id": v_result["inserted_id"]
            })

            file_number += 1

        return file_records

    # 处理各个方向
    try:
        # 冠状面处理
        coronal_records = process_orientation('r', dicom_coronal_files, img_p_coronal_files, img_v_coronal_files)
        sagittal_records = process_orientation('s', dicom_sagittal_files, img_p_sagittal_files, img_v_sagittal_files)
        axial_records = process_orientation('a', dicom_axial_files, img_p_axial_files, img_v_axial_files)

        # 更新研究状态
        mysql_tool.update(
            "UPDATE study SET process_status = %s, file_num = %s WHERE id = %s",
            ("已导入", len(coronal_records) + len(sagittal_records) + len(axial_records), study_id)
        )

        # 记录任务信息
        end_time = time.time()
        mysql_tool.insert(
            "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
            ('数据导入', 'IMPORT',
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
             'FINISHED', 'ENABLE')
        )

        # 日志记录
        logger.info(
            f"已存储 {len(extracted_files)} 张图片至 MongoDB 和 MySQL，执行时长：{round(end_time - start_time, 2)} 秒")
        return Result.success().to_response()

    except Exception as e:
        # 错误处理
        logger.error(f"导入失败: {str(e)}")
        return Result.error(f"导入失败: {str(e)}").to_response()

    finally:
        # 确保资源关闭
        mongo_tool.close_connection()
        mysql_tool.close_connection()


@app.route('/lvs-py-api/image_process', methods=['POST'])
def create_image_process_task():
    task_id = str(len(tasks) + 1)  # 任务 ID
    tasks[task_id] = {"status": "processing", "result": None}
    # 启动后台线程
    thread = threading.Thread(target=image_process, args=(request.get_json(), task_id))
    thread.start()
    return Result.success_with_data(
        {"task_id": task_id, "task_name": "image_process", "status": "processing", "message": "处理中"}).to_response()


@app.route('/lvs-py-api/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    task = tasks[task_id]
    if task["status"] == "processing":
        return Result.success_with_data({"task_id": task_id, "status": "processing", "message": "处理中"}).to_response()
    elif task["status"] == "finished":
        return Result.success_with_data({"task_id": task_id, "status": "finished", "message": "处理完成"}).to_response()
    else:
        return Result.error("任务不存在").to_response()


def image_process(json_data, task_id):
    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    # mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")
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
    image_sagittal_files = {}
    image_p_sagittal_images = {}
    image_v_sagittal_images = {}
    image_axial_files = {}
    image_p_axial_images = {}
    image_v_axial_images = {}
    for row in res['data']:
        for record in row:
            if record['orientation'] == 'r':
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['original_image_id'])})
                if result_doc["success"]:
                    image_files[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['perfusion_image_id'])})
                if result_doc["success"]:
                    image_p_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['ventilation_image_id'])})
                if result_doc["success"]:
                    image_v_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            elif record['orientation'] == 's':
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['original_image_id'])})
                if result_doc["success"]:
                    image_sagittal_files[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['perfusion_image_id'])})
                if result_doc["success"]:
                    image_p_sagittal_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['ventilation_image_id'])})
                if result_doc["success"]:
                    image_v_sagittal_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            elif record['orientation'] == 'a':
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['original_image_id'])})
                if result_doc["success"]:
                    image_axial_files[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['perfusion_image_id'])})
                if result_doc["success"]:
                    image_p_axial_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
                result_doc = mongo_tool.find_one({"_id": ObjectId(record['ventilation_image_id'])})
                if result_doc["success"]:
                    image_v_axial_images[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]


    # 裁切+resize 返回 p和v 的图片 dict
    def asd111666(ct_images):
        import cv2
        result_dict = {}
        # 处理每对图像
        for ct_filename, ct_data in ct_images.items():
            # 找到对应的mask（假设文件名有某种关联性，这里简单匹配）

            print(ct_filename)

            try:
                # 将bytes转换为numpy数组
                ct_nparr = np.frombuffer(ct_data, np.uint8)
                # 解码图像
                original_ct = cv2.imdecode(ct_nparr, cv2.IMREAD_COLOR)
                # 预处理
                # original_ct_rgb = cv2.cvtColor(original_ct, cv2.COLOR_BGR2RGB)
                # if direction == 'up':
                #     cropped_img = original_ct[0:405, 145:620]
                # else:
                cropped_img = original_ct[477:950, 145:620]
                resize_img = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_NEAREST)
                # 测试 ct 和 p/v 裁切
                if ct_filename == 'IMG-094221-0269_269.png':
                    pass
                cv2.imwrite('ct_preprocess.png', resize_img)
                success, encoded_img = cv2.imencode('.png', resize_img.copy())
                if not success:
                    raise ValueError(f"图像编码失败: {ct_filename}")
                result_dict[ct_filename] = encoded_img.tobytes()

            except Exception as e:
                print(f"Error processing image {ct_filename}: {e}")
        return result_dict
    image_p_sagittal_images = asd111666(image_p_sagittal_images)
    image_v_sagittal_images = asd111666(image_v_sagittal_images)
    # image_sagittal_files = asd111666(image_sagittal_files)
    image_p_axial_images = asd111666(image_p_axial_images)
    image_v_axial_images = asd111666(image_v_axial_images)
    # image_axial_files = asd111666(image_axial_files)

    # Process the images using the UNet model
    mark_images_c = process_stream(image_files, 'c')
    mark_images_s = process_stream(image_v_sagittal_images, 's')
    mark_images_a = process_stream(image_v_axial_images, 'a')
    # 对齐彩图与掩码图
    process_p_result_dict = MaskProcessTool.process_ct_images(ct_images=image_p_images, mask_images=mark_images_c, flag='c')
    process_v_result_dict = MaskProcessTool.process_ct_images(ct_images=image_v_images, mask_images=mark_images_c, flag='c')
    process_p_sagittal_result_dict = MaskProcessTool.process_ct_images(ct_images=image_p_sagittal_images, mask_images=mark_images_s, flag='s')
    process_v_sagittal_result_dict = MaskProcessTool.process_ct_images(ct_images=image_v_sagittal_images, mask_images=mark_images_s, flag='s')
    process_p_axial_result_dict = MaskProcessTool.process_ct_images(ct_images=image_p_axial_images, mask_images=mark_images_a, flag='a')
    process_v_axial_result_dict = MaskProcessTool.process_ct_images(ct_images=image_v_axial_images, mask_images=mark_images_a, flag='a')
    analyzer = VQAnalyzer()
    ventilation_perfusion_ratio_r = analyzer.analyze_dicts(process_v_result_dict, process_p_result_dict)
    ventilation_perfusion_ratio_s = analyzer.analyze_dicts(process_v_sagittal_result_dict, process_p_sagittal_result_dict)
    ventilation_perfusion_ratio_a = analyzer.analyze_dicts(process_v_axial_result_dict, process_p_axial_result_dict)
    print("冠切平均V/Q比值：", ventilation_perfusion_ratio_r)
    print("矢切切平均V/Q比值：", ventilation_perfusion_ratio_s)
    print("轴切平均V/Q比值：", ventilation_perfusion_ratio_a)
    ###############################################
    # Insert processed images into MongoDB with white pixel count calculated first
    mongo_id_to_filename_p_dict_r = {}
    mongo_id_to_filename_v_dict_r = {}
    mongo_id_to_filename_p_dict_s = {}
    mongo_id_to_filename_v_dict_s = {}
    mongo_id_to_filename_p_dict_a = {}
    mongo_id_to_filename_v_dict_a = {}
    file_number = 1
    for mk_r, pk_r, vk_r in zip(mark_images_c.keys(), process_p_result_dict.keys(), process_v_result_dict.keys()):
        # Write the image bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(mark_images_c[str(mk_r)])
            tmp_file_path = tmp_file.name

        # Calculate white pixel count using the temporary file
        white_pixel_count = PixelTool.calculate_white_pixel_count(tmp_file_path)
        os.remove(tmp_file_path)  # Remove the temporary file

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk_r,
            "image_data": process_p_result_dict[str(pk_r)],
            "orientation": 'r',
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_p_dict_r[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk_r,
            "image_data": process_v_result_dict[str(vk_r)],
            "orientation": 'r',
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_v_dict_r[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }
        file_number += 1
    file_number = 1
    for mk_s, pk_s, vk_s in zip(mark_images_s.keys(), process_p_sagittal_result_dict.keys(), process_v_sagittal_result_dict.keys()):
        # Write the image bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(mark_images_s[str(mk_s)])
            tmp_file_path = tmp_file.name

        # Calculate white pixel count using the temporary file
        white_pixel_count = PixelTool.calculate_white_pixel_count(tmp_file_path)
        os.remove(tmp_file_path)  # Remove the temporary file

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk_s,
            "image_data": process_p_sagittal_result_dict[str(pk_s)],
            "orientation": 's',
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_p_dict_s[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk_s,
            "image_data": process_v_sagittal_result_dict[str(vk_s)],
            "orientation": 's',
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_v_dict_s[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }
        file_number += 1
    file_number = 1
    for mk_a, pk_a, vk_a in zip(mark_images_a.keys(), process_p_axial_result_dict.keys(), process_v_axial_result_dict.keys()):
        # Write the image bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(mark_images_a[str(mk_a)])
            tmp_file_path = tmp_file.name

        # Calculate white pixel count using the temporary file
        white_pixel_count = PixelTool.calculate_white_pixel_count(tmp_file_path)
        os.remove(tmp_file_path)  # Remove the temporary file

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk_a,
            "image_data": process_p_axial_result_dict[str(pk_a)],
            "orientation": 'a',
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_p_dict_a[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }

        # Insert image along with white_pixel_count into MongoDB
        mongo_result = mongo_tool.insert_one({
            "study_id": str(study_id),
            "filename": mk_a,
            "image_data": process_v_axial_result_dict[str(vk_a)],
            "orientation": 'a',
        })
        if mongo_result["success"]:
            # Adjust the dict to store both the filename and white_pixel_count
            mongo_id_to_filename_v_dict_a[mongo_result["inserted_id"]] = {
                "white_pixel_count": white_pixel_count,
                "file_number": file_number,
            }
        file_number += 1
    file_number = 1
    # Insert the processed image details into MySQL, including the white pixel count
    pixel_sum_r = 0
    for mongo_id_r, info_r in mongo_id_to_filename_p_dict_r.items():
        mysql_tool.update(
            "update file set processed_perfusion_id = %s, perfusion_pixel_count = %s where study_id = %s and file_number = %s and orientation = 'r';",
            (mongo_id_r, info_r["white_pixel_count"], study_id, info_r["file_number"])
        )
        pixel_sum_r += info_r["white_pixel_count"]
    pixel_sum_s = 0

    for mongo_id_s, info_s in mongo_id_to_filename_p_dict_s.items():
        mysql_tool.update(
            "update file set processed_perfusion_id = %s, perfusion_pixel_count = %s where study_id = %s and file_number = %s and orientation = 's';",
            (mongo_id_s, info_s["white_pixel_count"], study_id, info_s["file_number"])
        )
        pixel_sum_s += info_s["white_pixel_count"]
    pixel_sum_a = 0

    for mongo_id_a, info_a in mongo_id_to_filename_p_dict_a.items():
        mysql_tool.update(
            "update file set processed_perfusion_id = %s, perfusion_pixel_count = %s where study_id = %s and file_number = %s and orientation = 'a';",
            (mongo_id_a, info_a["white_pixel_count"], study_id, info_a["file_number"])
        )
        pixel_sum_a += info_a["white_pixel_count"]

    mysql_tool.update(
        "update `study` set pixel_p_sum = %s, ventilation_perfusion_ratio = %s, pixel_p_sum_sagittal = %s, ventilation_perfusion_ratio_sagittal = %s, pixel_p_sum_axial = %s, ventilation_perfusion_ratio_axial = %s where id = %s",
        (pixel_sum_r, round(float(ventilation_perfusion_ratio_r), 2), pixel_sum_s, round(float(ventilation_perfusion_ratio_s), 2), pixel_sum_a, round(float(ventilation_perfusion_ratio_a), 2), study_id)
    )

    pixel_sum_r = 0
    for mongo_id_r, info_r in mongo_id_to_filename_v_dict_r.items():
        mysql_tool.update(
            "update file set processed_ventilation_id = %s, ventilation_pixel_count = %s where study_id = %s and file_number = %s and orientation = 'r';",
            (mongo_id_r, info_r["white_pixel_count"], study_id, info_r["file_number"])
        )
        pixel_sum_r += info_r["white_pixel_count"]

    pixel_sum_s = 0
    for mongo_id_s, info_s in mongo_id_to_filename_v_dict_s.items():
        mysql_tool.update(
            "update file set processed_ventilation_id = %s, ventilation_pixel_count = %s where study_id = %s and file_number = %s and orientation = 's';",
            (mongo_id_s, info_s["white_pixel_count"], study_id, info_s["file_number"])
        )
        pixel_sum_s += info_s["white_pixel_count"]

    pixel_sum_a = 0
    for mongo_id_a, info_a in mongo_id_to_filename_v_dict_a.items():
        mysql_tool.update(
            "update file set processed_ventilation_id = %s, ventilation_pixel_count = %s where study_id = %s and file_number = %s and orientation = 'a';",
            (mongo_id_a, info_a["white_pixel_count"], study_id, info_a["file_number"])
        )
        pixel_sum_a += info_a["white_pixel_count"]
################################
    end_time = time.time()  # 记录结束时间
    mysql_tool.update(
        "update study set pixel_v_sum = %s, pixel_v_sum_sagittal = %s, pixel_v_sum_axial = %s, process_status = %s, execute_time = %s where id = %s",
        (pixel_sum_r, pixel_sum_s, pixel_sum_a, "已处理", round(end_time - start_time, 2), study_id)
    )

    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('图像处理', 'PROCESS', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )

    # Close database connections
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    tasks[task_id] = {"status": "finished", "mark_images": "处理完成"}
    logger.info(f"已处理 {len(mark_images_c) + len(mark_images_s) + len(mark_images_a)} 张图片，执行时长：{round(end_time - start_time, 2)} 秒")
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
    # mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")
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
    res_r = mysql_tool.execute_query("SELECT * FROM file WHERE study_id = %s and orientation='r';", (study_id,))
    res_s = mysql_tool.execute_query("SELECT * FROM file WHERE study_id = %s and orientation='s';", (study_id,))
    res_a = mysql_tool.execute_query("SELECT * FROM file WHERE study_id = %s and orientation='a';", (study_id,))
    input_p_files_r = {}
    input_v_files_r = {}
    input_p_files_s = {}
    input_v_files_s = {}
    input_p_files_a = {}
    input_v_files_a = {}
    analyzer = VQAnalyzer()
    indices_r = []
    indices_s = []
    indices_a = []

    for row in res_r['data']:
        n_r = len(row)
        indices_r = np.linspace(192, 253, image_example_nums, dtype=int)
        row = [row[i] for i in indices_r]
        for record in row:
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['processed_perfusion_id'])})
            if result_doc["success"]:
                input_p_files_r[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            result_doc2 = mongo_tool.find_one({"_id": ObjectId(record['processed_ventilation_id'])})
            if result_doc2["success"]:
                input_v_files_r[result_doc["data"]["filename"]] = result_doc2["data"]["image_data"]
    for row in res_s['data']:
        n_s = len(row)
        indices_s = np.linspace(284, 388, image_example_nums, dtype=int)
        row = [row[i] for i in indices_s]
        for record in row:
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['processed_perfusion_id'])})
            if result_doc["success"]:
                input_p_files_s[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['processed_ventilation_id'])})
            if result_doc["success"]:
                input_v_files_s[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
    for row in res_a['data']:
        n_a = len(row)
        indices_a = np.linspace(206, 349, image_example_nums, dtype=int)
        row = [row[i] for i in indices_a]
        for record in row:
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['processed_perfusion_id'])})
            if result_doc["success"]:
                input_p_files_a[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
            result_doc = mongo_tool.find_one({"_id": ObjectId(record['processed_ventilation_id'])})
            if result_doc["success"]:
                input_v_files_a[result_doc["data"]["filename"]] = result_doc["data"]["image_data"]
    # test for save full output
    # os.makedirs('p_output', exist_ok=True)
    # for filename, image_data in input_p_files.items():
    #     with open(os.path.join('p_output', filename), 'wb') as f:
    #         f.write(image_data)
    # os.makedirs('v_output', exist_ok=True)
    # for filename, image_data in input_v_files.items():
    #     with open(os.path.join('v_output', filename), 'wb') as f:
    #         f.write(image_data)

    print("冠切：" + str(len(input_p_files_r)))
    print("矢切：" + str(len(input_p_files_s)))
    print("轴切：" + str(len(input_p_files_a)))
    if len(input_p_files_r) < image_example_nums or len(input_p_files_s) < image_example_nums or len(input_p_files_a) < image_example_nums:
        return Result.error('暂无图像数据，请先进行图像处理').to_response()
    indices = [indices_r, indices_s, indices_a]
    result_files_r, result_files_s, result_files_a = analyzer.analyze_and_visualize_dicts(input_v_files_r, input_p_files_r), analyzer.analyze_and_visualize_dicts(input_v_files_s, input_p_files_s), analyzer.analyze_and_visualize_dicts(input_v_files_a, input_p_files_a)
    # 使用 BytesIO 存储 PDF 文件数据，而不是保存到磁盘
    pdf_output = generate_report(
        patient_name=patient_info['name'],
        age=patient_info['age'],
        gender="男" if patient_info['gender'] == "MALE" else "女",
        exam_date=study_info['study_date'],
        exam_number=study_info['id'],
        ventilation_perfusion_ratio=study_info['ventilation_perfusion_ratio'],
        description=study_info['description'],
        input_p_files_r=input_p_files_r,
        input_v_files_r=input_v_files_r,
        result_files_r=result_files_r,
        input_p_files_s=input_p_files_s,
        input_v_files_s=input_v_files_s,
        result_files_s=result_files_s,
        input_p_files_a=input_p_files_a,
        input_v_files_a=input_v_files_a,
        result_files_a=result_files_a,
        idx_arr=indices,
        n_a=n_a,
        n_s=n_s,
        n_r=n_r
    )

    end_time = time.time()  # 记录结束时间
    mysql_tool.insert(
        "INSERT INTO task (name, type, created_at, finished_at, task_status, status) VALUES (%s, %s, %s, %s, %s, %s);",
        ('报告生成', 'REPORT', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), 'FINISHED', 'ENABLE')
    )

    # Close database connections
    mongo_tool.close_connection()
    mysql_tool.close_connection()

    # 返回 PDF 内容作为响应
    # return jsonify({'pdf': pdf_output.getvalue().decode('latin1')})  # Use Latin1 for safe encoding
    return Result.success_with_data({'pdf': base64.b64encode(pdf_output.getvalue()).decode('utf-8')}).to_response()


def generate_report(patient_name, age, gender, exam_date, exam_number, ventilation_perfusion_ratio, description,
                    input_p_files_r, input_v_files_r, result_files_r, input_p_files_s, input_v_files_s, result_files_s,
                    input_p_files_a, input_v_files_a, result_files_a, idx_arr, n_a, n_s, n_r
                    ):
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


    base64_p_images_r = {}
    for i, (image_name, image_bytes) in enumerate(input_p_files_r.items(), 1):
        base64_p_images_r[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_v_images_r = {}
    for i, (image_name, image_bytes) in enumerate(input_v_files_r.items(), 1):
        base64_v_images_r[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_r_images_r = {}
    for i, (image_name, image_bytes) in enumerate(result_files_r.items(), 1):
        base64_r_images_r[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_p_images_s = {}
    for i, (image_name, image_bytes) in enumerate(input_p_files_s.items(), 1):
        base64_p_images_s[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_v_images_s = {}
    for i, (image_name, image_bytes) in enumerate(input_v_files_s.items(), 1):
        base64_v_images_s[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_r_images_s = {}
    for i, (image_name, image_bytes) in enumerate(result_files_s.items(), 1):
        base64_r_images_s[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_p_images_a = {}
    for i, (image_name, image_bytes) in enumerate(input_p_files_a.items(), 1):
        base64_p_images_a[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_v_images_a = {}
    for i, (image_name, image_bytes) in enumerate(input_v_files_a.items(), 1):
        base64_v_images_a[f'image{i}'] = encode_image_to_base64(image_bytes)
    base64_r_images_a = {}
    for i, (image_name, image_bytes) in enumerate(result_files_a.items(), 1):
        base64_r_images_a[f'image{i}'] = encode_image_to_base64(image_bytes)

    logo_base64 = encode_file_to_base64("assets/images/吉大二院logo.jpg")
    with open("assets/css/bootstrap.min.css", "r", encoding="utf-8") as file:
        bootstrap_css_content = file.read()
    # HTML模板字符串
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Document</title>
    <style>
        {bootstrap_css_content}
        body {{
            width: 1000px;
            height: 800px;
            margin: 0 auto;
        }}
        p {{
            font-size: 17px;
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
            position: relative;
            bottom: 50px;
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
            font-size: 14px;
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
        <img src="data:image/jpeg;base64,{logo_base64}" alt="吉大二院logo" style="position: relative; width: 100px; height: 100px; bottom: 11%; left: 8%;">
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
                <p>检查示例：</p>
            </div>
        </div>
        <div class="row flex2">
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_r_images_r['image1']}" style="width: 360px; height: 270px;margin-bottom:15px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[0][0] + 1}/{n_r} 冠切面</p>
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_r_images_r['image2']}" style="width: 360px; height: 270px;margin-bottom:15px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[0][1] + 1}/{n_r} 冠切面</p>
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_r_images_s['image1']}" style="width: 360px; height: 270px;margin-bottom:15px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[1][0] + 1}/{n_s} 矢切面</p>
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_r_images_s['image2']}" style="width: 360px; height: 270px;margin-bottom:15px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[1][1] + 1}/{n_s} 矢切面</p>
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_r_images_a['image1']}" style="width: 360px; height: 270px;margin-bottom:15px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[2][0] + 1}/{n_a} 轴切面</p>
            </div>
            <div class="col-6">
                <img class="study-img" src="data:image/png;base64,{base64_r_images_a['image2']}" style="width: 360px; height: 270px;margin-bottom:15px;position: relative;left: 15%;">
                <p style="text-align: right">{idx_arr[2][1] + 1}/{n_a} 轴切面</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

    # 使用 Playwright 将 HTML 转换为 PDF
    def resource_path(relative_path):
        """ 获取打包后资源的绝对路径 """
        if getattr(sys, 'frozen', False):  # 判断是否是打包后的环境
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    ...

    # 使用 Playwright 将 HTML 转换为 PDF
    buffer = BytesIO()
    with sync_playwright() as p:
        # 指定浏览器路径
        browser_path = resource_path(
            os.path.join("playwright_browser", "chromium_headless_shell-1169", "chrome-win", "headless_shell.exe"))

        if not os.path.exists(browser_path):
            raise FileNotFoundError(f"浏览器未找到: {browser_path}")

        # 启动浏览器并指定路径
        browser = p.chromium.launch(
            executable_path=browser_path,
            args=["--allow-file-access-from-files"]
        )

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
def create_data_clear_task():
    task_id = str(len(tasks) + 1)  # 任务 ID
    tasks[task_id] = {"status": "processing", "result": None}
    # 启动后台线程
    thread = threading.Thread(target=data_clear, args=(request.get_json(), task_id))
    thread.start()
    return Result.success_with_data(
        {"task_id": task_id, "task_name": "data_clear", "status": "processing", "message": "处理中"}).to_response()


def data_clear(json_data, task_id):
    start_time = time.time()
    print('data clear 开始')
    study_id = json_data.get('studyId')
    if not study_id:
        tasks[task_id] = {"status": "error", "msg": "检查id 不存在"}
        return Result.error('检查id 不存在').to_response()

    # 数据库初始化
    mongo_tool = MongoDBTool(db_name="mongo_vision", collection_name="dicom_images")
    # mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")
    mysql_tool = MySQLTool(host="localhost", user="root", password="root", database="db_vision")
    mongo_tool.delete_many({"study_id": str(study_id)})
    # 删除 mysql file 记录, 检查记录像素值归零
    mysql_tool.delete("delete from file where study_id = %s", (study_id,))
    mysql_tool.update(
        "update study set pixel_p_sum = 0, pixel_v_sum = 0, ventilation_perfusion_ratio = 0, process_status = '未导入', file_num = 0, execute_time = 0 where id = %s",
        (study_id,))

    mongo_tool.close_connection()
    mysql_tool.close_connection()

    end_time = time.time()  # 记录结束时间
    print('data clear 结束' + str(round(end_time - start_time, 2)) + 's')
    tasks[task_id] = {"status": "finished", "msg": "处理完成"}
    return


if __name__ == '__main__':
    app.run()
