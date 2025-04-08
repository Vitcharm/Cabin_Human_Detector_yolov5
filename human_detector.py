import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# 设置YOLOv5的路径
def setup_yolov5():
    # 检查YOLOv5是否已经安装，如果没有则克隆并安装
    yolov5_dir = Path('./thirdParty/yolov5')
    if not yolov5_dir.exists():
        print(f"thirdParty目录下未找到yolov5，请查看README下载依赖")
        return False
    # 将YOLOv5目录添加到系统路径
    yolov5_path = str(Path('./thirdParty/yolov5').resolve())
    if yolov5_path not in sys.path:
        sys.path.append(yolov5_path)
    print(f"yolov5配置成功")
    return True

# 加载YOLOv5模型
def load_model():
    # 检查本地是否存在yolov5s.pt模型文件
    model_path = 'yolov5s.pt'
    if os.path.exists(model_path):
        print(f"从本地加载模型: {model_path}")
        # 从本地加载模型，使用当前目录下的yolov5文件夹作为仓库路径
        yolov5_repo_path = str(Path('./thirdParty/yolov5').resolve())
        model = torch.hub.load(yolov5_repo_path, 'custom', path=model_path, source='local')
    else:
        print("本地未找到模型文件，从PyTorch Hub下载模型...")
        # 从PyTorch Hub下载预训练的YOLOv5s模型
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # 只检测人类（类别索引为0）
    model.classes = [0]  # 只检测人类
    return model

# 处理图像并进行检测
def detect_humans(model, image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    # 转换BGR到RGB（YOLOv5需要RGB格式）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 使用模型进行检测
    results = model(img_rgb)
    
    # 获取检测结果
    detections = results.pandas().xyxy[0]
    
    # 过滤出人类检测结果（class=0）
    human_detections = detections[detections['class'] == 0]
    
    return img, human_detections

# 可视化检测结果
def visualize_detections(img, detections, output_path=None):
    # 在图像上绘制检测框
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加置信度标签
        label = f"Person: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存结果（如果指定了输出路径）
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
    
    # 显示结果
    cv2.imshow('Human Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主函数
def main():
    # 设置YOLOv5环境
    if not setup_yolov5():
        return
    
    # 加载模型
    print("正在加载YOLOv5s模型...")
    model = load_model()
    
    # 设置测试数据集路径
    test_dir = Path('./dataset/test')
    output_dir = Path('./results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有测试图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [str(f) for f in test_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    if not test_images:
        print(f"在 {test_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(test_images)} 个测试图像")
    
    # 处理每个测试图像
    for image_path in test_images:
        print(f"处理图像: {image_path}")
        
        # 检测人体
        img, detections = detect_humans(model, image_path)
        if img is None:
            continue
        
        # 获取图像文件名
        image_name = Path(image_path).name
        output_path = str(output_dir / f"detected_{image_name}")
        
        # 可视化并保存结果
        visualize_detections(img, detections, output_path)
        
        print(f"检测到 {len(detections)} 个人体，结果已保存到 {output_path}")

if __name__ == "__main__":
    main()