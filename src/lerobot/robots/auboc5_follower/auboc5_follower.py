#!/usr/bin/env python

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
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode, MotorsBus
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_auboc5_follower import AUBOC5FollowerConfig
import pyaubo_sdk
import math

logger = logging.getLogger(__name__)


class AUBOC5Follower(Robot):
    """
    AUBOC5 Follower Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = AUBOC5FollowerConfig
    name = "auboc5_follower"

    def __init__(self, config: AUBOC5FollowerConfig):
        super().__init__(config)
        self.config = config
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
        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        self.robot_ip = "192.168.31.35"  # 服务器 IP 地址
        self.robot_port = 30004  # 端口号
        self.robot_interface = None
        self.mc = None
        self.dt = 0.1

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.robot_rpc_client.hasConnected() and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot_rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 请求超时时间
        self.robot_rpc_client.connect(self.robot_ip, self.robot_port)  # 接口调用: 连接到 RPC 服务
        if self.robot_rpc_client.hasConnected():
            print("RPC连接成功！")
            self.robot_rpc_client.login("aubo", "123456")  # 接口调用: 机械臂登录

            #连接相机
            for cam in self.cameras.values():
                cam.connect()

            logger.info(f"{self} connected.")
        else:
            raise DeviceNotConnectedError(f"{self} could not connect to RPC service at {self.robot_ip}:{self.robot_port}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        joint_pos = self.robot_interface.getRobotState().getJointPositions()
        obs_dict = {f"{i+1}":joint_pos[i] for i in range(len(joint_pos))}
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        obs_dict["7.pos"] = 0.0
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {f"{i+1}": val for i,val in enumerate(action)}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.robot_interface.getRobotState().getJointPositions()
            goal_present_pos = {goal_pos[str(i)]:val for i,val in enumerate(present_pos)}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
        
        # Send goal position to the arm
        q1 = [goal_pos[str(i+1)] for i in range(0, 6)]
        self.mc.moveJoint(q1, 80 * (math.pi / 180), 60 * (math.pi / 180), 0, 0)

        # 阻塞
        self.wait_arrival(self.robot_interface)
        
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.robot_rpc_client.disconnect()  # 接口调用: 断开RPC连接
        # for cam in self.cameras.values():
        #     cam.disconnect()

        logger.info(f"{self} disconnected.")

    def wait_arrival(self,robot_interface):
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