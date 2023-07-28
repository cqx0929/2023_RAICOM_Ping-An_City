#!/home/cqx/miniconda3/envs/py39/bin/python

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import numpy as np


class PACar(object):
    def __init__(self):
        # 初始化节点
        self.node = rospy.init_node('move_turtle_to_target', anonymous=False)

        # 定义当前姿态
        self.current_pose = Pose()

        # 创建一个运动速度发布者
        self.pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)

        # 设置ROS节点的循环频率10Hz
        self.rate = rospy.Rate(10)  # 设置ROS节点的循环频率为10Hz

        # 路线
        self.road_line = [
            [],
            []
        ]

    def static(self):
        pass

    # 依次调用目标点运动函数
    def move_to_targets(self):
        points_x, points_y = self.road_line[0], self.road_line[1]
        if len(self.road_line[0]) == 0:
            pass
            # 调用运动函数
        else:
            for i, j in zip(points_x, points_y):
                self.move_to_target(i, j)

    # 订阅到的turtle1位置信息回调函数
    def pose_callback(self, data):
        self.current_pose = data

    def move_to_target(self, target_x, target_y):

        while not rospy.is_shutdown():

            # 获取turtle1的位置信息
            rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)

            current_x, current_y = self.current_pose.x, self.current_pose.y

            # 计算到目标点的距离
            distance = ((target_x - current_x) ** 2 + (target_y - current_y) ** 2) ** 0.5
            twist_msg = Twist()
            if distance < 0.1:
                # 到达目标点，停止运动
                self.pub.publish(twist_msg)
                rospy.loginfo(f"Turtle reached the target point({target_x, target_y})!")
                break

            # 计算目标点方向角
            target_angle = math.atan2(target_y - current_y, target_x - current_x)

            # 计算角速度，线速度
            angular_speed = 10.0
            liner_x_speed = 5
            angle_diff = target_angle - self.current_pose.theta
            while angle_diff > 3.14159:
                angle_diff -= 2 * 3.14159
            while angle_diff < -3.14159:
                angle_diff += 2 * 3.14159
            twist_msg.angular.z = angular_speed * angle_diff
            if abs(angle_diff) < 0.00001:
                twist_msg.linear.x = liner_x_speed * distance
            else:
                pass

            # 发布控制指令
            self.pub.publish(twist_msg)
            self.rate.sleep()

    def g_circle(self, center_x, center_y, radius, num_points, start, stop):
        self.static()
        # 生成一组均匀分布的角度值，从0到2*pi
        angles = np.linspace(start * 2 * np.pi, stop * 2 * np.pi, num_points)
        # 计算圆上每个点的x和y坐标
        points_x = center_x + radius * np.cos(angles)
        points_y = center_y + radius * np.sin(angles)
        self.road_line_merge(points_x, points_y)

    def g_rect(self, left_btn_x, left_btn_y, right_top_x, right_top_y):
        self.static()
        points_x = [left_btn_x, right_top_x, right_top_x, left_btn_x, left_btn_x]
        points_y = [left_btn_y, left_btn_y, right_top_y, right_top_y, left_btn_y]
        self.road_line_merge(points_x, points_y)

    def g_point(self, point_x, point_y):
        i, j = point_x, point_y
        self.road_line[0].append(i)
        self.road_line[1].append(j)

    def road_line_merge(self, points_x, points_y):
        for i, j in zip(points_x, points_y):
            self.road_line[0].append(i)
            self.road_line[1].append(j)


def run():
    car = PACar()
    # PA City space
    # car.g_point(0, 2)
    # car.g_point(0, 1.5)
    # car.g_circle(1.5, 1.5, 1.5, 5, 1/2, 3/4)
    # car.g_point(3.5, 0)
    # car.g_circle(3.5, 1.5, 1.5, 5, -1/4, 0)
    # car.g_point(5, 4)
    # car.g_circle(3.75, 4, 1.25, 10, 0, 1/2)
    # car.g_circle(1.25, 4, 1.25, 10, 0, -1/2)
    # car.g_circle(0.2, 5.8, 0.2, 5, -1/2, -3/4)
    # car.g_point(3.75, 6)
    # car.g_circle(3.75, 7.25, 1.25, 10, -1/4, 1/4)
    # car.g_circle(0.2, 8.3, 0.2, 5, 1/4, 1/2)
    # car.g_point(0, 2)
    # for i in range(22):
    # car.g_rect(5.5, 5.5, 0.5*i, 0.5*i)
    angles = np.linspace(0 * 2 * np.pi, 1 * 2 * np.pi, 7)
    points_x = 5.5 + 5 * np.cos(angles)
    points_y = 5.5 + 5 * np.sin(angles)
    car.g_point(5.5, 5.5)
    for i, j in zip(points_x, points_y):
        car.g_point(i, j)
        car.g_point(5.5, 5.5)
    for i in range(10):
        car.g_circle(5.5, 5.5, i * 0.5, 7, 0, 1)
    car.move_to_targets()


if __name__ == '__main__':
    run()
