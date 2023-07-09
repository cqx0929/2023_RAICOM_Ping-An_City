import cv2
import os
import numpy as np
import time
import concurrent.futures


def img_imread(img_path):

    # 以灰度图读取图片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return img


def process_image(img):

    # 调整大小，加快处理
    resize_rate = 30
    img = cv2.resize(img, (int(img.shape[1] / resize_rate), int(img.shape[0] / resize_rate)))  # h, w
    h, w = img.shape

    # 定义腐蚀和膨胀的内核
    kernel = np.ones((3, 3), np.uint8)

    # 膨胀操作
    dilated_image = cv2.dilate(img, kernel, iterations=1)

    # 腐蚀操作
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # 二值化
    _, binary_image = cv2.threshold(eroded_image, 90, 255, cv2.THRESH_BINARY)

    # 循环处理图片每行，获取黑色区域左、右边缘
    left_edge = float('+inf')
    right_edge = float('-inf')
    for row in binary_image:
        black_x = np.where(row == 0)[0]
        if len(black_x):
            black_x_min = min(black_x)
            black_x_max = max(black_x)
            if black_x_min < left_edge:
                left_edge = black_x_min
            if black_x_max > right_edge:
                right_edge = black_x_max

    # 计算信标值代表左右转向情况
    turn_signal = - (left_edge + right_edge - w) * resize_rate
    return turn_signal


def run(
        img_load_dir='src',
        res_save_dir='res',
        res_dir='res',
        num_threads=61,
):
    try:
        # 图片处理计时起点
        process_start = time.perf_counter()

        # 清空 res_dir 中的文件
        res_list = os.listdir(res_dir)
        for file in res_list:
            # 如果文件存在，则删除
            if file:
                os.remove(os.path.join(res_dir, file))
        if res_save_dir not in os.listdir():
            os.mkdir(res_save_dir)

        # 转向信息保存路径
        txt_save_path = os.path.join(res_save_dir, 'turn_info.txt')

        # 所有图片名称
        p_names = os.listdir(img_load_dir)

        # 获取所有图片的路径
        img_paths = [os.path.join(img_load_dir, p_name) for p_name in p_names]

        # 多线读取图片
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            # 并行读取图片
            img_list = executor.map(img_imread, img_paths)

        # 多线处理图片
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            # 并行处理图片
            turn_signals = executor.map(process_image, img_list)

        # 转向信号处理
        s = ''
        for p_name, turn_signal in zip(p_names, turn_signals):
            turn_standard = 200
            if turn_signal > turn_standard:
                turn = '左转'
            elif turn_signal < -turn_standard:
                turn = '右转'
            else:
                turn = '直行'

            # 结果输出
            print(f'图片“{p_name}”所示的情况需要{turn}')

            # 记录信标值，方便分析
            s += f'{p_name} {turn_signal} {turn}\n'

        # 性能检测
        process_time = time.perf_counter() - process_start
        rate = len(img_paths) / process_time
        print(f'process_time: {process_time:.2f} s')
        print(f'rate: {rate:.2f} /s')

        # 保存所有转向信号信息
        with open(txt_save_path, 'w+', encoding='utf-8') as fp:
            fp.write(s)

    except Exception as e:
        print("发生错误：", str(e))


if __name__ == '__main__':
    run()
