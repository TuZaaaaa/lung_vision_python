import io

import cv2
import numpy as np
from typing import Dict

from PIL import Image


def process_ct_images(ct_images: Dict[str, bytes], mask_images: Dict[str, bytes], flag: str) -> Dict[str, bytes]:
    """
    处理CT和掩膜图像的流式处理函数
    输入:
        ct_images: Dict[ct_filename, bytes] - CT图像数据
        mask_images: Dict[mask_filename, bytes] - 掩膜图像数据
    输出:
        Dict[filename, bytes] - 处理后的图像数据，以ct_filename命名
    """
    result_dict = {}

    # 确保输入字典数量匹配
    if len(ct_images) != len(mask_images):
        raise ValueError("CT图像和掩膜图像数量不匹配")

    # 处理每对图像
    for ct_filename, ct_data in ct_images.items():
        # 找到对应的mask（假设文件名有某种关联性，这里简单匹配）
        mask_filename = next((k for k in mask_images.keys() if str(ct_filename).split('_')[-1] in k.split("_")[-1]), None)
        print(mask_filename)
        print(ct_filename)
        if not mask_filename:
            raise ValueError(f"未找到与 {ct_filename} 对应的掩膜图像")

        try:
            # 将bytes转换为numpy数组
            ct_nparr = np.frombuffer(ct_data, np.uint8)
            mask_nparr = np.frombuffer(mask_images[mask_filename], np.uint8)

            # 解码图像
            original_ct = cv2.imdecode(ct_nparr, cv2.IMREAD_COLOR)
            segmented_lungs = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)

            if original_ct is None or segmented_lungs is None:
                raise ValueError(f"图像解码失败: {ct_filename}")

            # 预处理
            # original_ct_rgb = cv2.cvtColor(original_ct, cv2.COLOR_BGR2RGB)
            if flag == 'c':
                cropped_img = original_ct[477:950, 145:620]
                resize_img = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_NEAREST)
            # cropped_img = original_ct[0:447, 0:788]
            else:
                resize_img = original_ct


            cv2.imwrite('ct_crop.png', resize_img)
            cv2.imwrite('ct_seg.png', segmented_lungs)
            # test
            # segmented_lungs = cv2.resize(segmented_lungs, (original_ct.shape[1], original_ct.shape[0]))

            # 应用默认变换（简化示例，可根据需要添加特征匹配）
            M = np.float32([[1, 0, 5], [0, 1, -3]])  # 默认平移
            aligned_lungs = cv2.warpAffine(segmented_lungs, M,
                                           (resize_img.shape[1], resize_img.shape[0]))

            # 创建掩膜
            _, lung_mask = cv2.threshold(aligned_lungs, 127, 255, cv2.THRESH_BINARY)
            lung_mask_3ch = cv2.merge([lung_mask] * 3)

            # 提取肺部区域
            result = resize_img.copy()
            result[lung_mask_3ch == 0] = 0

            # 将结果编码为bytes
            success, encoded_img = cv2.imencode('.png', result)
            if not success:
                raise ValueError(f"图像编码失败: {mask_filename}")


            result_dict[mask_filename] = encoded_img.tobytes()

            # test
            # cv2.imwrite("save.png", result)
            # return result_dict

        except Exception as e:
            print(f"处理 {mask_filename} 时出错: {str(e)}")
            continue
    return result_dict


# 示例使用方式
if __name__ == "__main__":
    # 模拟输入数据
    with open('ct.png', 'rb') as f:
        ct_data = f.read()
    with open('mask.png', 'rb') as f:
        mask_data = f.read()

    ct_images = {'ct.png': ct_data}
    mask_images = {'ct.png': mask_data}

    # 调用函数
    results = process_ct_images(ct_images, mask_images)
    img = Image.open(io.BytesIO(results['ct.png']))
    img.save('save.png')

    # 检查结果
    for filename, data in results.items():
        print(f"处理完成: {filename}, 数据大小: {len(data)} bytes")
