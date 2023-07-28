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
            liner_x_speed = 10.0
            angle_diff = target_angle - self.current_pose.theta
            while angle_diff > 3.14159:
                angle_diff -= 2 * 3.14159
            while angle_diff < -3.14159:
                angle_diff += 2 * 3.14159
            twist_msg.angular.z = angular_speed * angle_diff
            if abs(angle_diff) < 0.0001:
                twist_msg.linear.x = liner_x_speed * distance
            else:
                pass

            # 发布控制指令
            self.pub.publish(twist_msg)
            self.rate.sleep()

    def g_circle(self, center_x, center_y, radius, num_points, start, stop) -> np.array:
        self.static()
        # 生成一组均匀分布的角度值，从0到2*pi
        angles = np.linspace(start * np.pi, stop * np.pi, num_points)
        # 计算圆上每个点的x和y坐标
        points_x = center_x + radius * np.cos(angles)
        points_y = center_y + radius * np.sin(angles)
        self.road_line_merge(points_x, points_y)

    # def g_square(self, left_btn_x, left_btn_y, side_length) -> np.array:
    #     self.static()
    #     right_side = left_btn_x + side_length
    #     top_side = left_btn_y + side_length
    #     points_x = [left_btn_x, right_side, right_side, left_btn_x, left_btn_x]
    #     points_y = [left_btn_y, left_btn_y, top_side, top_side, left_btn_y]
    #     self.road_line_merge(points_x, points_y)

    def g_point(self, point_x, point_y):
        i, j = point_x, point_y
        self.road_line[0].append(i)
        self.road_line[1].append(j)

    def road_line_merge(self, points_x, points_y):
        for i, j in zip(points_x, points_y):
            self.road_line[0].append(i)
            self.road_line[1].append(j)


if __name__ == '__main__':
    car = PACar()
    car.g_circle(5.5, 5.5, 3, 30, 0, 2)
    car.g_circle(5.5, 5.5, 4, 30, 0, 2)
    car.g_circle(5.5, 5.5, 5, 30, 0, 2)
    car.move_to_targets()

