# 2023 RAICOM 平安城市


# 简介
基于YOLOv5开发

# road_line_detect.py
## 运行需求
1. 主办方提供的数据集
## 运行方式
1. 修改road_line_detect.py并运行

# crowd_detect.py
## 运行需求
1. YOLOv5所需环境
2. 主办方提供的数据集
3. 人偶的针对性模型
## 运行方式
1. 从github获取YOLOv5源代码
2. 将数据集、crowd_detect.py放入YOLOv5根目录
3. 创建weights文件夹，将模型crowd_yolov5s.pt放入
4. 修改crowd_detect.py并运行

# building_segement.py
同上crowd_detect.py
