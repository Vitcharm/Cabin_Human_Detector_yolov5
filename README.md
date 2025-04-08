# 基于YOLOv5s的人体检测器

这个项目实现了一个简易的基于YOLOv5s的人体检测器，可以检测图像中的人体边界框。

## 项目结构

- `simple_human_detector.py`: 简化版人体检测脚本，使用PyTorch Hub直接加载YOLOv5s模型
- `human_detector.py`: 完整版人体检测脚本，包含YOLOv5环境设置
- `requirements.txt`: 项目依赖列表
- `dataset/test/`: 测试数据集目录
- `results/`: 检测结果输出目录（运行脚本后自动创建）

## 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```
在thirdParty目录下：
```bash
git clone https://github.com/ultralytics/yolov5.git
```
或手动下载，文件夹目录名为yolov5，然后安装yolov5依赖
```bash
pip install -r thirdParty/yolov5/requirements.txt
```

## 使用方法

### 完整版

运行完整版检测脚本：

```bash
python human_detector.py
```

## 功能说明

- 自动加载预训练的YOLOv5s模型
- 处理`dataset/test`目录下的所有图像文件
- 只检测人体（COCO数据集中的类别0）
- 在检测到的人体周围绘制边界框
- 将检测结果保存到`results`目录

## 注意事项

- 首次运行时会从PyTorch Hub下载YOLOv5s模型，需要联网
- 检测结果中会显示每个检测框的置信度
- 如果需要调整检测参数，可以修改脚本中的相关设置
