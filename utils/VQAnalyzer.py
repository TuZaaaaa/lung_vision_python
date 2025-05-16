import cv2
import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim

class VQAnalyzer:
    def __init__(self, black_threshold=15, registration_threshold=2):
        self.black_threshold = black_threshold
        self.registration_threshold = registration_threshold

    def _validate_images(self, vent_img, flow_img):
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
        try:
            gray_vent = cv2.cvtColor(vent_img, cv2.COLOR_BGR2GRAY)
            mask_vent = gray_vent > self.black_threshold
            gray_flow = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)
            mask_flow = gray_flow > self.black_threshold
            valid_pixels = np.logical_and(mask_vent, mask_flow)
            if np.sum(valid_pixels) < 100:
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
        try:
            if vent_img is None or flow_img is None:
                raise ValueError("图像数据为空")
            mask_vent = cv2.inRange(vent_img, (1, 1, 1), (255, 255, 255))
            mask_flow = cv2.inRange(flow_img, (1, 1, 1), (255, 255, 255))
            combined_mask = cv2.bitwise_and(mask_vent, mask_flow)
            if np.count_nonzero(combined_mask) == 0:
                raise ValueError("无有效像素")
            vent_gray = cv2.cvtColor(vent_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            flow_gray = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            vent_values = vent_gray[combined_mask > 0]
            flow_values = flow_gray[combined_mask > 0]
            v_mean = np.mean(vent_values)
            q_mean = np.mean(flow_values)
            return float(v_mean / (q_mean + eps))
        except Exception as e:
            print(f"区域计算错误: {str(e)}")
            return 0.0

    def _separate_lungs(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        left_mask = np.zeros_like(gray)
        right_mask = np.zeros_like(gray)
        if len(contours) >= 1:
            cv2.drawContours(right_mask, [contours[0]], -1, 255, -1)
        if len(contours) >= 2:
            cv2.drawContours(left_mask, [contours[1]], -1, 255, -1)
        return left_mask, right_mask

    def _process_lung(self, image, mask, lung_name):
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        y_coords, x_coords = np.where(mask > 0)
        pixels = np.column_stack((y_coords, x_coords))
        pixels = pixels.astype(np.float32)
        regions = []
        if len(pixels) == 0:
            return None, regions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, 9, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        for i in range(9):
            region_mask = np.zeros_like(mask)
            cluster_pixels = pixels[labels.flatten() == i]
            if len(cluster_pixels) == 0:
                continue
            y_min, x_min = np.min(cluster_pixels, axis=0)
            y_max, x_max = np.max(cluster_pixels, axis=0)
            region_mask[int(y_min):int(y_max) + 1, int(x_min):int(x_max) + 1] = 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            x, y, w, h = cv2.boundingRect(contours[0])
            roi = masked_img[y:y + h, x:x + w]
            regions.append({
                'name': f"{lung_name}_region{i + 1}",
                'roi': roi,
                'coords': (x, y, x + w, y + h),
                'centroid': (int(centers[i][1]), int(centers[i][0]))
            })
        return masked_img, regions

    def _visualize_results(self, vent_img, left_regions, right_regions, left_ratios, right_ratios):
        vis_img = vent_img.copy()
        all_ratios = left_ratios + right_ratios
        if not all_ratios:
            return vis_img

        # 归一化所有V/Q比到0-255
        min_vq = float(np.min(all_ratios))
        max_vq = float(np.max(all_ratios))
        def ratio_to_color_val(r):
            if max_vq - min_vq < 1e-6:
                return 127  # 防止除0
            return int(255 * (r - min_vq) / (max_vq - min_vq))

        # 用 OpenCV Jet colormap 进行映射
        def get_color(r):
            val = ratio_to_color_val(r)
            color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            return tuple(int(c) for c in color.tolist())

        # 画区域与数字
        for region, ratio in zip(left_regions, left_ratios):
            x1, y1, x2, y2 = region['coords']
            color = get_color(ratio)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 1)
            center_x, center_y = region['centroid']
            cv2.putText(vis_img, f"{ratio:.2f}", (center_x - 15, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for region, ratio in zip(right_regions, right_ratios):
            x1, y1, x2, y2 = region['coords']
            color = get_color(ratio)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 1)
            center_x, center_y = region['centroid']
            cv2.putText(vis_img, f"{ratio:.2f}", (center_x - 15, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 生成 colorbar
        colorbar_height = vis_img.shape[0]
        colorbar = np.zeros((colorbar_height, 50, 3), dtype=np.uint8)
        for i in range(colorbar_height):
            r = min_vq + (max_vq - min_vq) * (colorbar_height - 1 - i) / (colorbar_height - 1)
            color = get_color(r)
            colorbar[i, :] = color

        # 加上文本
        if max_vq > min_vq:
            cv2.putText(colorbar, f"{max_vq:.2f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(colorbar, f"{min_vq:.2f}", (5, colorbar_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(colorbar, "V/Q", (5, colorbar_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        final_img = np.hstack([vis_img, colorbar])
        return final_img


    def analyze_and_visualize_dicts(self, vent_dict, flow_dict):
        """
        分析并可视化区域VQ比，对结果图片覆盖输出（同名key）。
        返回: Dict[filename, bytes] (key为原文件名，value为处理后可视化图片字节流)
        """
        common_keys = set(vent_dict.keys()) & set(flow_dict.keys())
        output_dict = {}

        for filename in common_keys:
            v_img_bytes = vent_dict[filename]
            p_img_bytes = flow_dict[filename]
            v_img = cv2.imdecode(np.frombuffer(v_img_bytes, np.uint8), cv2.IMREAD_COLOR)
            p_img = cv2.imdecode(np.frombuffer(p_img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if v_img is None or p_img is None:
                print(f"{filename} 图像加载失败")
                continue

            # 分割左右肺
            left_mask, right_mask = self._separate_lungs(v_img)
            left_lung_img, left_regions = self._process_lung(v_img, left_mask, "Left")
            right_lung_img, right_regions = self._process_lung(v_img, right_mask, "Right")

            left_ratios, right_ratios = [], []
            for region in left_regions:
                x1, y1, x2, y2 = region['coords']
                vent_roi = v_img[y1:y2, x1:x2]
                flow_roi = p_img[y1:y2, x1:x2]
                ratio = self._calculate_vq_from_images(vent_roi, flow_roi)
                left_ratios.append(ratio)
            for region in right_regions:
                x1, y1, x2, y2 = region['coords']
                vent_roi = v_img[y1:y2, x1:x2]
                flow_roi = p_img[y1:y2, x1:x2]
                ratio = self._calculate_vq_from_images(vent_roi, flow_roi)
                right_ratios.append(ratio)

            # 区域可视化
            result_img = self._visualize_results(v_img, left_regions, right_regions, left_ratios, right_ratios)
            success, buffer = cv2.imencode('.png', result_img)
            if success:
                output_dict[filename] = buffer.tobytes()
            else:
                print(f"{filename} 结果图片编码失败")

        return output_dict

    def analyze_dicts(self, vent_dict, flow_dict):
        """
        计算所有成对图片的平均V/Q比值，仅返回均值（无区域分析/无图片输出）。
        """
        common_keys = set(vent_dict.keys()) & set(flow_dict.keys())
        ratios = []
        for filename in sorted(common_keys):
            vent_bytes = vent_dict[filename]
            flow_bytes = flow_dict[filename]
            vent_img = cv2.imdecode(np.frombuffer(vent_bytes, np.uint8), cv2.IMREAD_COLOR)
            flow_img = cv2.imdecode(np.frombuffer(flow_bytes, np.uint8), cv2.IMREAD_COLOR)
            try:
                self._validate_images(vent_img, flow_img)
                ratio = self._calculate_vq_ratio(vent_img, flow_img)
                ratios.append(ratio)
                print(f"{filename}: V/Q ratio = {ratio:.4f}")
            except Exception as e:
                print(f"{filename} analysis failed: {str(e)}")
        if ratios:
            average_ratio = np.mean(ratios)
            print(f"\n平均V/Q比值: {average_ratio:.4f}")
            return average_ratio
        else:
            print("No valid ratios calculated.")
            return None
