from flask import Flask, request, jsonify
from loguru import logger

from common.Result import Result
from utils.ArchiveExtractor import ArchiveExtractor

app = Flask(__name__)
# app.config['APPLICATION_ROOT'] = '/lvs-py-api'  # 设置请求前缀

# 配置日志文件轮转
logger.add("app.log", rotation="500 MB", retention="10 days", compression="zip")


@app.route('/lvs-py-api/upload', methods=['POST'])
def upload_file():
    # 获取上传的文件
    logger.info('file upload')
    file = request.files.get('file')
    study_id = request.form.get('studyId')

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # 如果需要，可以保存文件
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)

    extract_to = f'./extract'

    print(study_id)

    ArchiveExtractor.extract(file_path, extract_to)

    # 返回成功的响应
    return Result.success().to_response()


if __name__ == '__main__':
    app.run()
