print('导入基本包')
import os
import sys
from pathlib import Path
print('导入torch')
import torch
print('导入yolov5')
from utils.dataloaders import LoadImages
from utils.general import scale_boxes, xyxy2xywh, cv2, increment_path, non_max_suppression
from utils.plots import Annotator
from utils.torch_utils import smart_inference_mode
from models.common import DetectMultiBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将根目录加入系统环境路径变量
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
def run(
        weights=ROOT / 'weights/crowd_yolov5s.pt',  # 模型位置
        source=ROOT / 'crowd_dataset/val/images',  # 图片文件或者图片文件夹
        data=ROOT / 'data/crowd.yaml',  # 数据.yaml的位置
        imgsz=(640, 640),  # 推理尺寸
        conf_thres=0.25,  # 可信度阈值
        iou_thres=0.45,  # 非最大值抑制重合阈值
        max_det=10,  # 每张图片最大检测数
        view_img=0,  # 是否展示图片
        save_txt=True,  # 是否将结果保存到文本文档*.txt
        save_conf=False,  # 是否将可信度保存到 --save-txt labels
        nosave=False,  # 是否不保存图片
        classes=None,  # 根据类别过滤: '0' or '0 2 3'
        agnostic_nms=False,  # 类无关NMS
        augment=False,  # 增广推理
        visualize=False,  # 可视化特征
        project=ROOT / 'runs',  # 结果保存文件夹为project/name
        name='crowd_recognition',  # 结果保存文件夹为project/name
        exist_ok=True,  # 现有项目/名称可以，不递增
        line_thickness=3,  # 边界框厚度（像素）
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
):
    # 加载模型
    print('加载模型')
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt

    # 仅图片及图片文件夹
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 是否保存图片

    # 设置文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 递增运行
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # 创建文件夹

    # 数据加载
    print('加载数据')
    bs = 30  # 批数
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # 模型热身
    print('模型热身')
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 热身

    # 对每张图片进行预测
    print('进行预测')
    c2n = {'blue': 0, 'red': 0, 'gray': 0}
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)  # 将图像转到目标设备
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 精度转换
        im /= 255  # 归一化，所有色值除以255
        if len(im.shape) == 3:  # 批量调光扩展
            im = im[None]

        # 推理
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # 极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 结果处理
        for i, det in enumerate(pred):  # 对当前图片的预测结果进行处理，det为一张图中所有预测结果
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # 转换成路径对象
            save_path = str(save_dir / p.name)  # 图片保存路径
            txt_path = str(save_dir / 'labels' / p.stem)  # 每张图的预测框文本保存路径
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 将原图whwh转为tensor
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 绘图工具
            if len(det):
                # 将预测框从格式化图片缩放至原始图像大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                # 对各个分类进行数量统计
                for c in det[:, 5].unique():
                    c_num = (det[:, 5] == c).sum()
                    c2n[names[int(c)]] = c_num

                # 文本输出
                out_str = f'图片“{p.name}”中人群总数{len(det)}人：红色系{c2n["red"]}人，蓝色系{c2n["blue"]}人，灰黑色系{c2n["gray"]}人。'
                print(out_str)

                # 结果绘制
                c2c = {0: (255, 0, 0), 1: (0, 0, 255), 2: (70, 70, 70)}
                for *xyxy, conf, cls in reversed(det):

                    # 保存预测框信息
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 正常的xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 预测框标注信息格式
                    s = ('%g ' * len(line)).rstrip() % line + '\n'

                    with open(f'{txt_path}.txt', 'a+') as fp:
                        fp.write(s)

                    # 绘制格子和标签
                    c = int(cls)  # 整数化类别
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=c2c[c])

            # 图片展示
            boxed_res = annotator.result()
            if view_img:
                cv2.imshow('res', boxed_res)
                cv2.waitKey(1)

            # 保存画有预测框的图片
            cv2.imwrite(save_path, boxed_res)

    cv2.destroyAllWindows()

    # 输出结果去向
    if save_txt or save_img:
        print(f"结果保存于{save_dir}")


if __name__ == '__main__':
    with torch.no_grad():
        run()
