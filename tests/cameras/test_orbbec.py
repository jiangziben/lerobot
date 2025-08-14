#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_opencv.py::test_connect
# ```

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

pytest.importorskip("pyorbbecsdk")

from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig

TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"

def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH")
    _ = OrbbecCamera(config)


def test_connect():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH")
    camera = OrbbecCamera(config)

    camera.connect(warmup=False)
    assert camera.is_connected


def test_connect_already_connected():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH")
    camera = OrbbecCamera(config)
    camera.connect(warmup=False)

    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_read():
    config = OrbbecCameraConfig(serial_number_or_name="1", width=640, height=360, fps=30)
    camera = OrbbecCamera(config)
    camera.connect(warmup=True)

    img = camera.read()
    assert isinstance(img, np.ndarray)


def test_read_depth():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH", width=640, height=480, fps=30, use_depth=True)
    camera = OrbbecCamera(config)
    camera.connect(warmup=True)

    img = camera.read_depth(timeout_ms=2000)  # NOTE(Steven): Reading depth takes longer in CI environments.
    assert isinstance(img, np.ndarray)


def test_read_before_connect():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH")
    camera = OrbbecCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH")
    camera = OrbbecCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH")
    camera = OrbbecCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_async_read():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH", width=640, height=480, fps=30)
    camera = OrbbecCamera(config)
    camera.connect(warmup=True)

    try:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)
    finally:
        if camera.is_connected:
            camera.disconnect()  # To stop/join the thread. Otherwise get warnings when the test ends


def test_async_read_timeout():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH", width=640, height=480, fps=30)
    camera = OrbbecCamera(config)
    camera.connect(warmup=False)

    try:
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=0)
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_async_read_before_connect():
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH")
    camera = OrbbecCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


# @pytest.mark.parametrize(
#     "rotation",
#     [
#         Cv2Rotation.NO_ROTATION,
#         Cv2Rotation.ROTATE_90,
#         Cv2Rotation.ROTATE_180,
#         Cv2Rotation.ROTATE_270,
#     ],
#     ids=["no_rot", "rot90", "rot180", "rot270"],
# )
def test_rotation():
    rotation = Cv2Rotation.ROTATE_90
    config = OrbbecCameraConfig(serial_number_or_name="CP1Z842000SH", rotation=rotation, width=640, height=480, fps=30)
    camera = OrbbecCamera(config)
    camera.connect(warmup=True)

    img = camera.read()
    assert isinstance(img, np.ndarray)

    if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
        assert camera.width == 480
        assert camera.height == 640
        assert img.shape[:2] == (640, 480)
    else:
        assert camera.width == 640
        assert camera.height == 480
        assert img.shape[:2] == (480, 640)
