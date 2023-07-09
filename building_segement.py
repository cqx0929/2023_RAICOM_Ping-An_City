print('导入基本包')
import os
import sys
from pathlib import Path
import numpy as np
print('导入torch')
import torch
# 获取根目录，加入到python引用path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print('导入yolov5')
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, cv2,
                           increment_path, non_max_suppression, scale_boxes, scale_segments)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import smart_inference_mode
print('导入ocr')
from easyocr import Reader


@smart_inference_mode()
def run(
        weights=ROOT / 'weights/building_segement_best.pt',  # model.pt path(s)
        source=ROOT / 'datasets/yolov5_building_segement_dataset/valid/images/',  # file/dir
        data=ROOT / 'datasets/yolov5_building_segement_dataset/data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_conf=False,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=True,
):
    reader = Reader(
        lang_list=['ch_sim'],
        gpu=True,
        model_storage_directory='../easyocr_weights',
        download_enabled=False,
        detector=True,
        recognizer=True
    )
    # 加载模型
    print('加载模型')
    device = torch.device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # 文件路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    source = str(source)

    # 加载数据
    print('加载数据')
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # 进行推理
    print('开始推理')
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # 极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # 预测
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            imc = im0.copy()
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                segments = [
                    scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                    for x in reversed(masks2segments(masks))]

                # 结果处理
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # 分割信息
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format

                    # 加载原始图像并获取其宽度和高度
                    original_image = imc
                    original_height, original_width = original_image.shape[:2]

                    # 从文件中加载顶点坐标
                    points = line[1:]

                    # 缩放坐标点
                    scaled_points = []
                    for i in range(0, len(points), 2):
                        x = points[i] * original_width
                        y = points[i + 1] * original_height
                        scaled_points.append((x, y))

                    # 初始化最左上角和最右下角的点
                    top_left = scaled_points[0]
                    bottom_right = scaled_points[0]
                    top_right = scaled_points[0]
                    bottom_left = scaled_points[0]

                    # 遍历所有点，找到四个角
                    for point in scaled_points:
                        x, y = point
                        if x + y < top_left[0] + top_left[1]:  # 左上角
                            top_left = (x, y)
                        if x + (original_height - y) < bottom_left[0] + (original_height - bottom_left[1]):  # 左下角
                            bottom_left = (x, y)
                        if (original_width - x) + y < (original_width - top_right[0]) + top_right[1]:  # 右上角
                            top_right = (x, y)
                        if (original_width - x) + (original_height - y) < \
                                (original_width - bottom_right[0]) + (original_height - bottom_right[1]):  # 右下角
                            bottom_right = (x, y)

                    # 定义透视变换前的四个顶点坐标
                    src_pts = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

                    # 定义透视变换后的图像宽度和高度
                    dst_width = 2000
                    dst_height = int(dst_width/original_width*original_height)

                    # 定义透视变换后的四个顶点坐标
                    dst_pts = np.array([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]],
                                       dtype=np.float32)

                    # 计算透视变换矩阵
                    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                    # 进行透视变换
                    ppc_img = cv2.warpPerspective(original_image, perspective_matrix, (dst_width, dst_height))
                    if ppc_img.shape[1] > ppc_img.shape[0]:
                        ppc_img = cv2.resize(ppc_img, (ppc_img.shape[0], ppc_img.shape[1]))

                    # 裁切
                    ppc_w, ppc_h, _ = ppc_img.shape
                    ocr_src_img = ppc_img[int(dst_height/50):int(dst_height/5), :]
                    ocr_src_img = cv2.cvtColor(ocr_src_img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
                    _, ocr_src_img = cv2.threshold(ocr_src_img, 100, 255, cv2.THRESH_BINARY)

                    # ocr识别
                    result = reader.readtext(
                        image=ocr_src_img,
                        batch_size=1,
                        decoder='beamsearch',
                        beamWidth=1000
                    )
                    if len(result) > 0:
                        print(p.name+'的预测结果：', result[0][1])
                        cv2.imwrite(save_path, ppc_img)
                        ocr_save_path = str(save_dir / Path(p.name[:8]+result[0][1]+'.jpg'))
                        os.rename(save_path, ocr_save_path)
                    else:
                        print(p.name+'无识别结果！')

    print(f"结果保存于：{save_dir}")


if __name__ == '__main__':
    run()
