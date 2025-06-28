import cv2
import numpy as np
from skimage import exposure
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable


class VQAnalyzer:
    def __init__(self, black_threshold=15, registration_threshold=2):
        self.black_threshold = black_threshold
        self.registration_threshold = registration_threshold

    def _validate_images(self, vent_img, flow_img):
        """验证通气和血流图像是否适合分析"""
        if vent_img is None or flow_img is None:
            raise ValueError("有一张或多张图像未能加载")
        if vent_img.shape != flow_img.shape:
            raise ValueError(f"图像尺寸不匹配 通气图: {vent_img.shape} vs 血流图: {flow_img.shape}")
        vent_gray = cv2.cvtColor(vent_img, cv2.COLOR_BGR2GRAY)
        flow_gray = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)
        shift, error, _ = phase_cross_correlation(vent_gray, flow_gray)
        if np.linalg.norm(shift) > self.registration_threshold:
            raise ValueError(f"图像配准偏差过大 (Δx={shift[1]:.1f}, Δy={shift[0]:.1f} pixels)")
        ssim_score = ssim(vent_gray, flow_gray,
                          data_range=flow_gray.max() - flow_gray.min())
        if ssim_score < 0.7:
            raise ValueError(f"图像结构相似性过低 (SSIM={ssim_score:.2f})")

    def _calculate_vq_ratio(self, vent_img, flow_img):
        """计算整体V/Q比值"""
        # TODO 暂时未找到加权代码
        try:
            gray_vent = cv2.cvtColor(vent_img, cv2.COLOR_BGR2GRAY)
            mask_vent = gray_vent > self.black_threshold
            gray_flow = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)
            mask_flow = gray_flow > self.black_threshold
            valid_pixels = np.logical_and(mask_vent, mask_flow)
            if np.sum(valid_pixels) < 10:
                print(np.sum(valid_pixels))
                raise ValueError("有效像素不足")
            vent_values = gray_vent[valid_pixels].astype(np.float32) / 255.0
            flow_values = gray_flow[valid_pixels].astype(np.float32) / 255.0
            v_mean = np.mean(vent_values)
            q_mean = np.mean(flow_values)
            if np.isnan(v_mean) or np.isnan(q_mean):
                raise ValueError("出现NaN值")
            return float(v_mean / (q_mean + 1e-6))
        except Exception as e:
            print(f"计算失败: {str(e)}")
            return 0.0

    def _calculate_vq_from_images(self, vent_img, flow_img, eps=1e-6):
        """计算非零像素区域的V/Q比值"""
        try:
            if vent_img is None or flow_img is None:
                raise ValueError("图像数据为空")

            # 生成联合有效像素掩膜（排除绝对零值）
            mask_vent = cv2.inRange(vent_img, (1, 1, 1), (255, 255, 255))
            mask_flow = cv2.inRange(flow_img, (1, 1, 1), (255, 255, 255))
            combined_mask = cv2.bitwise_and(mask_vent, mask_flow)

            # 有效性校验
            if np.count_nonzero(combined_mask) == 0:
                raise ValueError("无有效像素")
            # 转换为归一化灰度图
            vent_gray = cv2.cvtColor(vent_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            flow_gray = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            # 提取有效像素
            vent_values = vent_gray[combined_mask > 0]
            flow_values = flow_gray[combined_mask > 0]

            # 计算比值
            v_mean = np.mean(vent_values)
            q_mean = np.mean(flow_values)
            return float(v_mean / (q_mean + eps))

        except Exception as e:
            print(f"区域计算错误: {str(e)}")
            return 0.0

    def _separate_lungs(self, image):
        """分割左右肺部区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 获取最大的两个轮廓（左右肺）
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # 按x坐标排序确定左右肺
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        # 创建左右肺掩膜
        left_mask = np.zeros_like(gray)
        right_mask = np.zeros_like(gray)

        if len(contours) >= 1:
            cv2.drawContours(right_mask, [contours[0]], -1, 255, -1)
        if len(contours) >= 2:
            cv2.drawContours(left_mask, [contours[1]], -1, 255, -1)

        return left_mask, right_mask

    def _divide_lung_into_regions(self, mask, lung_name):
        """将单个肺部划分为上中下三个区域"""
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        x, y, w, h = cv2.boundingRect(contours[0])
        region_height = h // 3
        regions = []

        for i in range(3):
            region_mask = np.zeros_like(mask)
            y_start = y + i * region_height
            y_end = y + (i + 1) * region_height if i < 2 else y + h

            region_rect = np.zeros_like(mask)
            cv2.rectangle(region_rect, (x, y_start), (x + w, y_end), 255, -1)
            region_mask = cv2.bitwise_and(region_rect, mask)

            region_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not region_contours:
                continue

            rx, ry, rw, rh = cv2.boundingRect(region_contours[0])
            centroid = (rx + rw // 2, ry + rh // 2)

            regions.append({
                'name': f"{lung_name}_region_{i + 1}",
                'coords': (rx, ry, rx + rw, ry + rh),
                'mask': region_mask,
                'centroid': centroid,
                'level': i  # 0:上, 1:中, 2:下
            })

        return regions

    def _process_lung(self, image, mask, lung_name):
        """处理单个肺部区域并划分为上中下三个区域"""
        regions = self._divide_lung_into_regions(mask, lung_name)
        final_regions = []

        for region in regions:
            region_mask = region['mask']
            contours, _ = cv2.findContours(region_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            x, y, w, h = cv2.boundingRect(contours[0])
            subregion_width = w // 3

            for j in range(3):
                subregion_mask = np.zeros_like(mask)
                x_start = x + j * subregion_width
                x_end = x + (j + 1) * subregion_width if j < 2 else x + w

                subregion_rect = np.zeros_like(mask)
                cv2.rectangle(subregion_rect, (x_start, y), (x_end, y + h), 255, -1)
                subregion_mask = cv2.bitwise_and(subregion_rect, region_mask)

                subregion_contours, _ = cv2.findContours(subregion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not subregion_contours:
                    continue

                sx, sy, sw, sh = cv2.boundingRect(subregion_contours[0])
                centroid = (sx + sw // 2, sy + sh // 2)
                roi = image[sy:sy + sh, sx:sx + sw]

                position_name = ""
                if region['level'] == 0:
                    position_name = "upper"
                elif region['level'] == 1:
                    position_name = "middle"
                else:
                    position_name = "lower"

                if j == 0:
                    position_name += "_left"
                elif j == 1:
                    position_name += "_center"
                else:
                    position_name += "_right"

                final_regions.append({
                    'name': f"{lung_name}_{position_name}",
                    'roi': roi,
                    'coords': (sx, sy, sx + sw, sy + sh),
                    'centroid': centroid,
                    'mask': subregion_mask
                })

        return image, final_regions

    def _put_text_with_outline(self, img, text, pos, font_scale=0.35, thickness=1):
        """带描边的文本显示函数"""
        x, y = pos
        # 绘制黑色描边（8个方向）
        offsets = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx != 0 or dy != 0]
        for dx, dy in offsets:
            cv2.putText(img, text, (x + dx, y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
        # 绘制白色文本
        cv2.putText(img, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def _visualize_results(self, vent_img, left_regions, right_regions, left_ratios, right_ratios):
        """优化后的可视化结果"""
        vis_img = vent_img.copy()
        all_ratios = left_ratios + right_ratios

        if not all_ratios:
            return vis_img

        # norm = Normalize(vmin=min(all_ratios), vmax=max(all_ratios))

        # 先把列表变成 ndarray，并显式转成 float（或你需要的其他类型）
        ratios_arr = np.asarray(all_ratios, dtype=float)

        # 再做归一化
        if ratios_arr.size == 0:
            raise ValueError("all_ratios 为空，无法归一化")

        vmin, vmax = ratios_arr.min(), ratios_arr.max()
        if vmin == vmax:                      # 防止除以 0
            norm = np.zeros_like(ratios_arr, dtype=float)
        else:
            norm = exposure.rescale_intensity(
                ratios_arr,
                in_range=(vmin, vmax),
                out_range=(0, 1)
            )
        # cmap = plt.get_cmap('jet')

        def build_jet_lut(n=256):
            """
            纯 NumPy 实现的经典 Jet colormap。
            返回形状 (n, 3) 的浮点 RGB 数组，数值范围 0-1。
            """
            x = np.linspace(0, 1, n)
            r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
            g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
            b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
            return np.stack((r, g, b), axis=1)

        cmap = build_jet_lut()  # (256, 3) array

        # 绘制左肺区域
        for region, ratio in zip(left_regions, left_ratios):
            contours, _ = cv2.findContours(region['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            color = (np.array(cmap(norm(ratio))[:3]) * 255).astype(np.uint8).tolist()
            cv2.drawContours(vis_img, contours, -1, color, 2)

            center_x, center_y = region['centroid']
            text = f"{ratio:.2f}" if ratio >= 0.1 else f"{ratio:.1e}"
            self._put_text_with_outline(vis_img, text, (center_x - 15, center_y))

        # 绘制右肺区域
        for region, ratio in zip(right_regions, right_ratios):
            contours, _ = cv2.findContours(region['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            color = (np.array(cmap(norm(ratio))[:3]) * 255).astype(np.uint8).tolist()
            cv2.drawContours(vis_img, contours, -1, color, 2)

            center_x, center_y = region['centroid']
            text = f"{ratio:.2f}" if ratio >= 0.1 else f"{ratio:.1e}"
            self._put_text_with_outline(vis_img, text, (center_x - 15, center_y))

        # 添加颜色条
        colorbar_height = vis_img.shape[0]
        colorbar = np.zeros((colorbar_height, 50, 3), dtype=np.uint8)
        for i in range(colorbar_height):
            val = norm((colorbar_height - 1 - i) / (colorbar_height - 1)) if all_ratios else 0
            color = (np.array(cmap(val)[:3]) * 255).astype(np.uint8).tolist()
            colorbar[i, :] = color

        # 添加颜色条标签
        if all_ratios:
            max_text = f"{max(all_ratios):.2f}" if max(all_ratios) >= 0.1 else f"{max(all_ratios):.1e}"
            min_text = f"{min(all_ratios):.2f}" if min(all_ratios) >= 0.1 else f"{min(all_ratios):.1e}"

            self._put_text_with_outline(colorbar, max_text, (5, 20), 0.35)
            self._put_text_with_outline(colorbar, min_text, (5, colorbar_height - 10), 0.35)
            self._put_text_with_outline(colorbar, "V/Q", (5, colorbar_height // 2), 0.5)

        return np.hstack([vis_img, colorbar])

    def analyze_and_visualize_dicts(self, vent_dict, flow_dict):
        """
        分析并可视化区域VQ比，对结果图片覆盖输出（同名key）。
        返回: Dict[filename, bytes] (key为原文件名，value为处理后可视化图片字节流)
        """
        common_keys = set(vent_dict.keys()) & set(flow_dict.keys())
        output_dict = {}

        for filename in common_keys:
            try:
                v_img_bytes = vent_dict[filename]
                p_img_bytes = flow_dict[filename]

                # 将字节数据转换为图像
                v_img = cv2.imdecode(np.frombuffer(v_img_bytes, np.uint8), cv2.IMREAD_COLOR)
                p_img = cv2.imdecode(np.frombuffer(p_img_bytes, np.uint8), cv2.IMREAD_COLOR)

                if v_img is None or p_img is None:
                    print(f"{filename} 图像加载失败")
                    continue

                # 验证图像配准情况
                self._validate_images(v_img, p_img)

                # 分割左右肺
                left_mask, right_mask = self._separate_lungs(v_img)

                # 处理左右肺区域
                v_img, left_regions = self._process_lung(v_img, left_mask, "Left")
                v_img, right_regions = self._process_lung(v_img, right_mask, "Right")

                left_ratios, right_ratios = [], []

                # 计算左肺区域比值
                for region in left_regions:
                    x1, y1, x2, y2 = region['coords']
                    vent_roi = v_img[y1:y2, x1:x2]
                    flow_roi = p_img[y1:y2, x1:x2]
                    ratio = self._calculate_vq_from_images(vent_roi, flow_roi)
                    left_ratios.append(ratio)

                # 计算右肺区域比值
                for region in right_regions:
                    x1, y1, x2, y2 = region['coords']
                    vent_roi = v_img[y1:y2, x1:x2]
                    flow_roi = p_img[y1:y2, x1:x2]
                    ratio = self._calculate_vq_from_images(vent_roi, flow_roi)
                    right_ratios.append(ratio)

                # 区域可视化
                result_img = self._visualize_results(v_img, left_regions, right_regions, left_ratios, right_ratios)

                # 编码为PNG字节流
                success, buffer = cv2.imencode('.png', result_img)
                if success:
                    output_dict[filename] = buffer.tobytes()
                    print(f"成功处理图像: {filename}")
                else:
                    print(f"{filename} 结果图片编码失败")

            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
                # 生成错误提示图像
                error_img = np.zeros((300, 500, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Error Processing: {filename}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(error_img, str(e), (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                _, buffer = cv2.imencode('.png', error_img)
                output_dict[filename] = buffer.tobytes()

        return output_dict

    def analyze_dicts(self, vent_dict, flow_dict):
        """
        计算所有成对图片的平均V/Q比值，仅返回均值（无区域分析/无图片输出）。
        """
        common_keys = set(vent_dict.keys()) & set(flow_dict.keys())
        ratios = []

        for filename in sorted(common_keys):
            try:
                vent_bytes = vent_dict[filename]
                flow_bytes = flow_dict[filename]

                # 将字节数据转换为图像
                vent_img = cv2.imdecode(np.frombuffer(vent_bytes, np.uint8), cv2.IMREAD_COLOR)
                flow_img = cv2.imdecode(np.frombuffer(flow_bytes, np.uint8), cv2.IMREAD_COLOR)

                self._validate_images(vent_img, flow_img)
                ratio = self._calculate_vq_ratio(vent_img, flow_img)
                if ratio != 0:
                    ratios.append(ratio)
                print(f"{filename}: V/Q ratio = {ratio:.4f}")
            except Exception as e:
                print(f"{filename} analysis failed: {str(e)}")

        if ratios:
            average_ratio = np.mean(ratios)
            print(f"\n平均V/Q比值: {average_ratio:.4f}")
            return average_ratio
        else:
            print("没有计算有效的V/Q比值")
            return None