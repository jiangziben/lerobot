import time
import datetime
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import pyaubo_sdk
from typing import List
from collections import deque

from fontTools.merge.util import first

# 机器人通讯相关配置
robot_ip = "192.168.32.33"
rtde_port = 30010
rpc_port = 30004
# 创建RTDE-数据获取和RPC-运动控制客户端
robot_rtde_client = pyaubo_sdk.RtdeClient()
robot_rpc_client = pyaubo_sdk.RpcClient()

def wait_arrival(robot_interface):
    max_retry_count = 5
    cnt = 0

    # 接口调用: 获取当前的运动指令 ID
    exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂开始运动
    while exec_id == -1:
        if cnt > max_retry_count:
            return -1
        time.sleep(0.05)
        cnt += 1
        exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂运动完成
    while robot_interface.getMotionControl().getExecId() != -1:
        time.sleep(0.05)

    return 0


def main():
    # 接口调用: 设置 RPC 超时
    robot_rpc_client.setRequestTimeout(1000)
    # 接口调用: 连接 RPC 服务
    robot_rpc_client.connect(robot_ip, rpc_port)
    # 接口调用: 连接 RTDE 服务
    robot_rtde_client.connect(robot_ip, rtde_port)
    if robot_rpc_client.hasConnected() and robot_rpc_client.hasConnected():
        print("Robot RPC and RTDE connected successfully.")
        # 接口调用: 登录 RPC 服务
        robot_rpc_client.login("aubo", "123456")
        # 接口调用: 登陆 RTDE 服务
        robot_rtde_client.login("aubo", "123456")
        if robot_rpc_client.hasLogined() and robot_rtde_client.hasLogined():
            print("Robot RPC and RTDE logined successfully.")
            robot_name = robot_rpc_client.getRobotNames()[0]
            robot_interface = robot_rpc_client.getRobotInterface(robot_name)

            robot_interface.getRobotConfig().selectTcpForceSensor("Xinjingcheng")

            robot_interface.getRobotConfig().setPayload(0, [0]*3, [0] * 3, [0] * 9)
            robot_interface.getRobotConfig().setTcpForceOffset([0]*6)

            sensor_pose=[0.0, 0.0, 0.028, 0, 0, 0]
            robot_interface.getRobotConfig().setTcpForceSensorPose(sensor_pose)

            calib_joint1=[0,-15.0/180*np.pi,100.0/180*np.pi,115.0/180*np.pi,90.0/180*np.pi,-90.0/180*np.pi]
            calib_joint2=[0,-15.0/180*np.pi,100.0/180*np.pi,115.0/180*np.pi,90.0/180*np.pi,-180.0/180*np.pi]
            calib_joint3=[0,-15.0/180*np.pi,100.0/180*np.pi,25.0/180*np.pi,90.0/180*np.pi,-45.0/180*np.pi]

            robot_interface.getMotionControl().moveJoint(calib_joint1, 10 * (np.pi / 180), 10 * (np.pi / 180), 0, 0)
            wait_arrival(robot_interface)
            time.sleep(1)
            q1=robot_interface.getRobotState().getJointPositions()
            tcp_force1=robot_interface.getRobotState().getTcpForceSensors()
            print('Arrived frist point')

            robot_interface.getMotionControl().moveJoint(calib_joint2, 10 * (np.pi / 180), 10 * (np.pi / 180), 0, 0)
            wait_arrival(robot_interface)
            time.sleep(1)
            q2=robot_interface.getRobotState().getJointPositions()
            tcp_force2=robot_interface.getRobotState().getTcpForceSensors()
            print('Arrived second point')

            robot_interface.getMotionControl().moveJoint(calib_joint3, 10 * (np.pi / 180), 5 * (np.pi / 180), 0, 0)
            wait_arrival(robot_interface)
            time.sleep(1)
            q3=robot_interface.getRobotState().getJointPositions()
            tcp_force3=robot_interface.getRobotState().getTcpForceSensors()
            print('Arrived third point')

            pose1=robot_interface.getRobotAlgorithm().forwardKinematics(q1)
            pose2=robot_interface.getRobotAlgorithm().forwardKinematics(q2)
            pose3=robot_interface.getRobotAlgorithm().forwardKinematics(q3)

            qs=[q1,q2,q3]
            forces=[tcp_force1,tcp_force2,tcp_force3]
            poses=[pose1[0],pose2[0],pose3[0]]

            calib_result1=robot_interface.getRobotAlgorithm().calibrateTcpForceSensor(forces,poses)
            calib_result2=robot_interface.getRobotAlgorithm().calibrateTcpForceSensor2(forces,poses)

            robot_interface.getRobotConfig().setPayload(calib_result2[2], calib_result2[1], [0]*3, [0]*9)

            robot_interface.getRobotConfig().setTcpForceOffset(calib_result2[0])

            force_test1 = []
            force_test2 = []

            for i in range(10):
                TCP_force = robot_interface.getRobotState().getTcpForce()
                Sensor_force = robot_interface.getRobotState().getTcpForceSensors()
                force_test1.append(np.array(TCP_force,np.float32))
                force_test2.append(np.array(Sensor_force,np.float32))
                time.sleep(0.5)
            avg_force_test1 = sum(force_test1)/len(force_test1)
            avg_force_test2 = sum(force_test2)/len(force_test2)
            if all(-1 <= x <= 1 for x in avg_force_test1) :
                print('Calibration successful')
                print('TCP_force:',avg_force_test1)
            else:
                print('Calibration failed')
                robot_interface.getRobotConfig().setPayload(0, [0] * 3, [0] * 3, [0] * 9)
                robot_interface.getRobotConfig().setTcpForceOffset([0] * 6)
            print('Calibrtion Result1:',calib_result1)
            print('Calibrtion Result2:',calib_result2)

if __name__ == "__main__":
    main()