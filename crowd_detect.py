def pprint(_str):
    print('* '*20+_str+' *'*20+'\n\n')


pprint('导入包')
import os
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import Profile, scale_boxes, xyxy2xywh, cv2, increment_path, \
    non_max_suppression
from utils.plots import Annotator, save_one_box
from utils.torch_utils import smart_inference_mode


# from utils.torch_utils import select_device, smart_inference_mode
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, \
# colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

def pprint(__str):
    print('* '*20+__str+' *'*20+'\n\n')


@smart_inference_mode()
def run(
        weights=ROOT / 'weights/crowd_yolov5s.pt',  # 模型位置
        source=ROOT / 'crowd_dataset/val/images/P2_No17.jpg',  # 图片文件或者图片文件夹
        data=ROOT / 'data/crowd.yaml',  # 数据.yaml的位置
        imgsz=(640, 640),  # 推理尺寸
        conf_thres=0.25,  # 可信度阈值
        iou_thres=0.45,  # 非最大值抑制重合阈值
        max_det=10,  # 每张图片最大检测数
        view_img=True,  # 是否展示图片
        save_txt=False,  # 是否将结果保存到文本文档*.txt
        save_conf=False,  # 是否将可信度保存到 --save-txt labels
        save_crop=True,  # 是否保存根据预测框裁剪的图片
        nosave=False,  # 是否不保存图片
        classes=None,  # 根据类别过滤: '0' or '0 2 3'
        agnostic_nms=False,  # 类无关NMS
        augment=False,  # 增广推理
        visualize=False,  # 可视化特征
        project=ROOT / 'runs/detect',  # 结果保存文件夹为project/name
        name='crowd_recognition',  # 结果保存文件夹为project/name
        exist_ok=True,  # 现有项目/名称可以，不递增
        line_thickness=3,  # 边界框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏可信度
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧速率步幅
        ):
    # 加载模型
    pprint('加载模型')
    device = torch.device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt

    # 仅图片
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 是否保存图片

    # 设置文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 递增运行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建文件夹

    # 数据加载
    pprint('加载数据')
    bs = 1  # 批数
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # 进行推理
    pprint('进行推理')
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 热身
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 精度转换
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # 批量调光扩展

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # 极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 进行预测
        pprint('进行预测')
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                color_res = str()  # 各个颜色结果输出
                c2n = {'blue': 0, 'red': 0, 'gray': 0}

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    c2n[names[int(c)]] = n
                out_res = f'图片“{p.name}”中人群总数{len(det)}人：红色系{c2n["red"]}人，蓝色系{c2n["blue"]}人，灰黑色系{c2n["gray"]}人。'
                print(out_res)

                # 结果绘制
                c2h = {0: (255, 0, 0), 1: (0, 0, 255), 2: (70, 70, 70)}
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=c2h[c])
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 图片展示
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)

            # 保存画有预测框的图片
            if save_img and dataset.mode == 'image':
                cv2.imwrite(save_path, im0)

    # 输出结果去向
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"结果保存于{save_dir}{s}")


if __name__ == '__main__':
    with torch.no_grad():
        run()
