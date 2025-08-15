# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from . import AUBOC5Follower
from .config_auboc5_follower import AUBOC5FollowerEndEffectorConfig

logger = logging.getLogger(__name__)
import pyaubo_sdk
import math

class AUBOC5FollowerEndEffector(AUBOC5Follower):
    """
    AUBOC5Follower robot with end-effector space control.

    This robot inherits from AUBOC5Follower but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = AUBOC5FollowerEndEffectorConfig
    name = "auboc5_follower_end_effector"

    def __init__(self, config: AUBOC5FollowerEndEffectorConfig):
        super().__init__(config)
        norm_mode_body = MotorNormMode.DEGREES
        self.motors = {
                "1": Motor(1, "", norm_mode_body),
                "2": Motor(2, "", norm_mode_body),
                "3": Motor(3, "", norm_mode_body),
                "4": Motor(4, "", norm_mode_body),
                "5": Motor(5, "", norm_mode_body),
                "6": Motor(6, "", norm_mode_body),
                "7": Motor(7, "", MotorNormMode.RANGE_0_100),
            },

        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config

        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        self.robot_ip = "192.168.3.12"  # 服务器 IP 地址
        self.robot_port = 30004  # 端口号
        self.dt = 1.0 / config.fps  # 控制周期，单位为秒
        
    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }
    
    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        robot_name = self.robot_rpc_client.getRobotNames()[0]  # 接口调用: 获取机器人的名字
        self.robot_interface = self.robot_rpc_client.getRobotInterface(robot_name)
        #设置传感器类型
        self.robot_interface.getRobotConfig().selectTcpForceSensor("xinjingcheng")

        #设置负载参数
        weight = 0.847
        self.robot_interface.getRobotConfig().setPayload(weight,[0,0,0], [0,0,0], [0,0,0,0,0,0,0,0,0])

        #设置传感器安装位姿
        sensor_pose = [ 0, 0, -0.132, 0, 0, -0.785 ]
        self.robot_interface.getRobotConfig().setTcpForceSensorPose(sensor_pose)

        # #设置tcp偏置
        # tcp_offset = [0, 0, 0.0, 0, 0, 0]
        # robot_interface.getRobotConfig().setTcpOffset(tcp_offset)
        
        #设置力控参数
        admittance_m=[30.0,30.0,30.0,1.0,1.0,1.0]
        admittance_d=[1000.0,1000.0,2000.0,50.0,50.0,50.0]
        admittance_k= [0.0,0.0,0.0,0.0,0.0,0.0]
        self.robot_interface.getForceControl().setDynamicModel(admittance_m, admittance_d, admittance_k)
        #设置目标
        compliance = [True] * 6
        target_wrench = [0.0] * 6
        speed_limits = [2.0] * 6
        feature = [0.0] * 6
        self.robot_interface.getForceControl().setTargetForce(feature,compliance, target_wrench, speed_limits, pyaubo_sdk.TaskFrameType.TOOL_FORCE)
        #设置机械臂的速度比率
        self.mc = self.robot_interface.getMotionControl()
        # self.mc.setSpeedFraction(1.0)
        self.robot_interface.getForceControl().fcEnable()
        print("开启力控成功！")
        # # 开启 servo 模式
        # self.robot_interface.getMotionControl().setServoMode(True)
        # i = 0
        # while not self.mc.isServoModeEnabled():
        #     i = i + 1
        #     if i > 5:
        #         print("开启Servo模式失败！当前的Servo模式是： ", self.mc.isServoModeEnabled())
        #         return -1
        #     time.sleep(0.005)
        # print("开启Servo模式成功！当前的Servo模式是： ", self.mc.isServoModeEnabled())
        # self.csv_file = open("ee_velocity.csv", "w")
        # self.csv_file.write("speed_x,speed_y,speed_z,current_ee_vel_x,current_ee_vel_y,current_ee_vel_z\n")
        # self.force_csv_file = open("force.csv", "w")
        # self.force_csv_file.write("force_x,force_y,force_z,torque_x,torque_y,torque_z,force_x_tcp,force_y_tcp,force_z_tcp,torque_x_tcp,torque_y_tcp,torque_z_tcp\n")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'delta_x', 'delta_y', 'delta_z' for end-effector control
                   or a numpy array with [delta_x, delta_y, delta_z]

        Returns:
            The joint-space action that was sent to the motors
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Convert action to numpy array if not already
        if isinstance(action, dict):
            if all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                delta_ee = np.array(
                    [
                        action["delta_x"] * self.config.end_effector_step_sizes["x"],
                        action["delta_y"] * self.config.end_effector_step_sizes["y"],
                        action["delta_z"] * self.config.end_effector_step_sizes["z"],
                    ],
                    dtype=np.float32,
                )
                if "gripper" not in action:
                    action["gripper"] = [1.0]
                action = np.append(delta_ee, action["gripper"])
            else:
                logger.warning(
                    f"Expected action keys 'delta_x', 'delta_y', 'delta_z', got {list(action.keys())}"
                )
                action = np.zeros(4, dtype=np.float32)

        # Calculate current end-effector position using forward kinematics
        current_ee_pos = self.robot_interface.getRobotState().getTcpPose()

        # Set desired end-effector position by adding delta
        desired_ee_pos = current_ee_pos.copy()  # Keep orientation

        # frame transform
        frame = current_ee_pos.copy()
        frame[0:3] = [0,0,0]
        delta_ee_base = np.zeros(6, dtype=np.float32)
        delta_ee_base[:3] = action[:3]  # Only position change
        delta_ee_base = self.robot_rpc_client.getMath().poseTrans(frame,delta_ee_base)

        # Add delta to position and clip to bounds
        desired_ee_pos[:3] = np.array(current_ee_pos[:3]) + np.array(delta_ee_base[:3])
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3] = np.clip(
                desired_ee_pos[:3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )
        delta_ee_base[:3] = np.array(desired_ee_pos[:3]) - np.array(current_ee_pos[:3])
        speed = np.zeros(6, dtype=np.float32)
        current_ee_vel = self.robot_interface.getRobotState().getTcpSpeed()
        speed[:3] = np.array(delta_ee_base[:3]) / self.dt # Convert to speed for motion control
        speed[:3] = np.clip(speed[:3], -0.25, 0.25) 
        
        # Move the robot to the desired end-effector position
        # import time
        # start = time.time()        
        ret = self.mc.speedLine(speed,1.2, self.dt)
        # used_time = time.time() - start
        # self.mc.servoCartesian(desired_ee_pos,0.0,0.0,self.dt,0.0,0.0)
        # print("ret: ",ret)
        # print("used_time: ",used_time)
        print("desired_ee_pos:", desired_ee_pos)
        print("current_ee_pos:", current_ee_pos)
        print("action:", action)
        print("speed:", speed)
        print("current_ee_vel:", current_ee_vel)
        tcp_force = self.robot_interface.getRobotState().getTcpForce()
        print("tcp_force:", tcp_force)
        tcp_force_sensors = np.array(self.robot_interface.getRobotState().getTcpForceSensors())
        #伴随矩阵
        adjoint_matrix = np.zeros((6, 6), dtype=np.float32)
        R = np.array([[math.cos(np.pi/4),math.sin(np.pi/4),0],[-math.sin(np.pi/4),math.cos(np.pi/4),0],[0,0,1]], dtype=np.float32)
        adjoint_matrix[:3,:3] = R
        adjoint_matrix[3:6,3:6] = R
        tcp_force_sensors_tcp = adjoint_matrix @ tcp_force_sensors
        print("tcp_force_sensors: ",tcp_force_sensors)
        print("tcp_force_sensors_tcp: ",tcp_force_sensors_tcp)
        # # 将速度保存到csv中
        # self.csv_file.write(f"{speed[0]},{speed[1]},{speed[2]},{current_ee_vel[0]},{current_ee_vel[1]},{current_ee_vel[2]}\n")
        # # 将tcp_force保存到csv中
        # self.force_csv_file.write(f"{tcp_force_sensors[0]},{tcp_force_sensors[1]},{tcp_force_sensors[2]},{tcp_force_sensors[3]},{tcp_force_sensors[4]},{tcp_force_sensors[5]}, \
        #                           {tcp_force_sensors_tcp[0]},{tcp_force_sensors_tcp[1]},{tcp_force_sensors_tcp[2]},{tcp_force_sensors_tcp[3]},{tcp_force_sensors_tcp[4]},{tcp_force_sensors_tcp[5]}\n")
        # self.force_csv_file.flush()

        return desired_ee_pos

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        joint_pos = self.robot_interface.getRobotState().getJointPositions()
        obs_dict = {f"{i+1}.pos":joint_pos[i] for i in range(len(joint_pos))}
        obs_dict["7.pos"] = 0.0
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        # Success io
        semi_success = self.robot_interface.getIoControl().getConfigurableDigitalInput(1)
        success = self.robot_interface.getIoControl().getConfigurableDigitalInput(0)
        obs_dict["io"] = [semi_success,success]

        return obs_dict

    def reset(self):
        # Calculate current end-effector position using forward kinematics
        current_ee_pos = self.robot_interface.getRobotState().getTcpPose()
        delta_ee_base = np.zeros(6, dtype=np.float32)
        delta_ee_base[:3] = [0,0,-0.1]  # Only position change
        frame = current_ee_pos.copy()
        desired_ee_base = self.robot_rpc_client.getMath().poseTrans(frame,delta_ee_base)
        self.mc.moveLine(desired_ee_base,1.2, 0.25, 0, 0)
        self.wait_arrival(self.robot_interface)
        joint_angles = np.array([2.75,
            -40.34,
            126.55,
            165.40,
            84.76,
            -45.59],dtype=np.float32) * (math.pi / 180)
        super().send_action(joint_angles.tolist())

    def disconnect(self):
        self.robot_interface.getForceControl().fcDisable()
        print("关闭力控成功！")
        # # 关闭 servo 模式
        # self.mc.setServoMode(False)
        # i = 0
        # while self.mc.isServoModeEnabled():
        #     i = i + 1
        #     if i > 5:
        #         print("关闭Servo模式失败！当前的Servo模式是： ", self.mc.isServoModeEnabled())
        #         return -1
        #     time.sleep(0.005)
        # print("关闭Servo模式成功！当前的Servo模式是： ", self.mc.isServoModeEnabled())
        super().disconnect()