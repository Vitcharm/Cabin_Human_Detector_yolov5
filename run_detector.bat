@echo off
echo 正在运行YOLOv5s人体检测器...

echo 检查并安装依赖...
pip install -r requirements.txt

echo 运行人体检测器...
python simple_human_detector.py

echo 检测完成！结果保存在results目录中。
pause