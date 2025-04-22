import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import json
import onnxruntime as ort

# 导入human_detector和Unipose_demo的功能
from human_detector import setup_yolov5, load_model, detect_humans
from unipose_demo import (
    preprocess, build_session, inference, postprocess, 
    visualize, model_input_size, decode, get_simcc_maximum
)

# 设置输出目录
OUTPUT_DIR = Path('./results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置Unipose模型文件路径
ONNX_FILE = 'unipose_0314.onnx'

# 检查模型文件是否存在
def check_model_files():
    if not os.path.exists(ONNX_FILE):
        print(f"错误: 未找到Unipose模型文件 {ONNX_FILE}")
        return False
    return True

# 处理单个图像
def process_image(image_path, yolo_model, pose_sess):
    print(f"处理图像: {image_path}")
    
    # 1. 使用YOLOv5检测人体
    img, human_detections = detect_humans(yolo_model, image_path)
    if img is None or len(human_detections) == 0:
        print(f"未检测到人体: {image_path}")
        return None
    
    # 获取图像文件名
    image_name = Path(image_path).name
    output_image_path = str(OUTPUT_DIR / f"integrated_{image_name}")
    output_xml_path = str(OUTPUT_DIR / f"integrated_{Path(image_name).stem}.xml")
    
    # 创建原始图像的副本用于最终输出
    result_img = img.copy()
    
    # 2. 对每个检测到的人体进行关键点检测
    for _, detection in human_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        
        # 绘制边界框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加置信度标签
        label = f"Person: {conf:.2f}"
        cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 裁剪人体区域
        margin = 0  # 可以根据需要调整边距
        person_img = img[max(0, y1-margin):min(img.shape[0], y2+margin), 
                         max(0, x1-margin):min(img.shape[1], x2+margin)].copy()
        
        if person_img.size == 0:
            continue
        
        # 转换为RGB格式（Unipose需要RGB格式）
        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        try:
            # 3. 使用Unipose检测关键点
            # 预处理
            resized_img, center, scale = preprocess(person_img_rgb, model_input_size)
            
            # 推理
            outputs = inference(pose_sess, resized_img)
            
            # 后处理
            keypoints, scores = postprocess(outputs, model_input_size, center, scale)
            
            # 调整关键点坐标到原始图像坐标系
            for i in range(len(keypoints)):
                keypoints[i][:, 0] += x1
                keypoints[i][:, 1] += y1
            
            # 4. 在原始图像上可视化关键点
            visualize(result_img, keypoints, scores)
            
            # 保存关键点数据（可选）
            # keypoints_data = keypoints.tolist()
            # keypoints_file = str(OUTPUT_DIR / f"keypoints_{Path(image_name).stem}.json")
            # with open(keypoints_file, 'w') as f:
            #     json.dump(keypoints_data, f)
                
        except Exception as e:
            print(f"关键点检测失败: {e}")
    
    # 5. 保存结果图像
    cv2.imwrite(output_image_path, result_img)
    print(f"结果已保存到: {output_image_path}")
        
    return result_img


# 主函数
def main():
    # 1. 设置YOLOv5环境
    if not setup_yolov5():
        return
    
    # 2. 检查Unipose模型文件
    if not check_model_files():
        return
    
    # 3. 加载YOLOv5模型
    print("正在加载YOLOv5模型...")
    yolo_model = load_model()
    
    # 4. 加载Unipose模型
    print("正在加载Unipose模型...")
    device = 'cpu'
    pose_sess = build_session(ONNX_FILE, device)
    h, w = pose_sess.get_inputs()[0].shape[2:]
    global model_input_size
    model_input_size = (w, h)
    
    # 5. 设置测试数据集路径
    test_dir = Path('./dataset/test')
    
    # 6. 获取所有测试图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [str(f) for f in test_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    if not test_images:
        print(f"在 {test_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(test_images)} 个测试图像")
    
    # 7. 处理每个测试图像
    for image_path in test_images:
        result_img = process_image(image_path, yolo_model, pose_sess)
        
        # 显示结果（可选）
        # if result_img is not None:
        #     cv2.imshow('Integrated Detection', result_img)
        #     cv2.waitKey(1000)  # 显示1秒
    
    cv2.destroyAllWindows()
    print("处理完成!")

if __name__ == "__main__":
    main()