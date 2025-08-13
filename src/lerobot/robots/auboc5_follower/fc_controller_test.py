#! /usr/bin/env python
# coding=utf-8

"""
servoCartesian 运动

步骤:
第一步: 连接到 RPC 服务、机械臂登录、设置RPC请求超时时间
第二步: 读取 .txt 轨迹文件
第三步: 关节运动到轨迹中的第一个点
第四步: 开启 servo 模式
第五步: 做 servoCartesian 运动
第六步: 关闭 servo 模式
第七步: 断开 RPC 连接
"""
import pyaubo_sdk
import time
import numpy as np

robot_ip = "192.168.31.35"  # 服务器 IP 地址
robot_port = 30004  # 端口号
M_PI = 3.14159265358979323846
robot_rpc_client = pyaubo_sdk.RpcClient()


# 阻塞
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


def force_control():
    robot_name = robot_rpc_client.getRobotNames()[0]  # 接口调用: 获取机器人的名字
    robot_interface = robot_rpc_client.getRobotInterface(robot_name)
    #设置传感器类型
    robot_interface.getRobotConfig().selectTcpForceSensor("xinjingcheng")

    #设置负载参数
    weight = 0.847
    robot_interface.getRobotConfig().setPayload(weight,[0,0,0], [0,0,0], [0,0,0,0,0,0,0,0,0])

    #设置传感器安装位姿
    sensor_pose = [ 0, 0, -0.132, 0, 0, -0.785 ]
    robot_interface.getRobotConfig().setTcpForceSensorPose(sensor_pose)

    # #设置tcp偏置
    # tcp_offset = [0, 0, 0.0, 0, 0, 0]
    # robot_interface.getRobotConfig().setTcpOffset(tcp_offset)
    
    #设置力控参数
    admittance_m=[30.0,30.0,30.0,1.0,1.0,1.0]
    admittance_d=[1000.0,1000.0,2000.0,10.0,10.0,10.0]
    admittance_k= [0.0,0.0,0.0,0.0,0.0,0.0]
    robot_interface.getForceControl().setDynamicModel(admittance_m, admittance_d, admittance_k)
    #设置目标
    compliance = [True] * 6
    target_wrench = [0.0] * 6
    speed_limits = [2.0] * 6
    feature = [0.0] * 6
    robot_interface.getForceControl().setTargetForce(feature,compliance, target_wrench, speed_limits, pyaubo_sdk.TaskFrameType.TOOL_FORCE)
    #设置机械臂的速度比率
    mc = robot_interface.getMotionControl()
    mc.setSpeedFraction(0.3)
    # 获取机械臂当前位姿
    pose = robot_interface.getRobotState().getTcpPose()
    # 设置目标点
    movel_distance = 0.1  # 移动距离
    target_pose = robot_rpc_client.getMath().poseTrans(pose,[0.0,0.0,movel_distance,0.0,0.0,0.0])
    
    robot_interface.getForceControl().fcEnable()
    print("开启力控成功！")

    # 移动到目标点
    mc.moveLine(target_pose, 1.2, 0.1, 0.025, 0.0)
    wait_arrival(robot_interface)

    input_num = robot_interface.getIoControl().getConfigurableDigitalInputNum()
    print("IO输入数量：", input_num)

    # 打印所有的标准数字输入值
    input_value = []
    for i in range(input_num):
        value = robot_interface.getIoControl().getConfigurableDigitalInput(i)
        input_value.append(value)
    print("输入值:", input_value)
    if input_value[0] == True:
        print("success!")
    else:
        print("fail!")

    # time.sleep(100)

    # #关闭力控
    robot_interface.getForceControl().fcDisable()
    print("关闭力控成功！")
    
    return 0


if __name__ == '__main__':
    robot_rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 请求超时时间
    robot_rpc_client.connect(robot_ip, robot_port)  # 接口调用: 连接到 RPC 服务
    if robot_rpc_client.hasConnected():
        print("RPC连接成功！")
        robot_rpc_client.login("aubo", "123456")  # 接口调用: 机械臂登录
        force_control()  
        robot_rpc_client.disconnect()  # 接口调用: 断开RPC连接
    else:
        print("RPC连接失败！")
