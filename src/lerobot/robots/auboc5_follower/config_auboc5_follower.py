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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig  # noqa: F401

@RobotConfig.register_subclass("auboc5_follower")
@dataclass
class AUBOC5FollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    # When the robot disconnects, disable the torque to prevent it from moving
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

@RobotConfig.register_subclass("auboc5_follower_end_effector")
@dataclass
class AUBOC5FollowerEndEffectorConfig(AUBOC5FollowerConfig):
    """Configuration for the AUBOC5FollowerEndEffector robot."""

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )