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

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)
from pathlib import Path
import time
from lerobot.utils.robot_utils import busy_wait

logging.basicConfig(level=logging.INFO)


def eval_policy(env, policy, n_episodes,control_fps=5):
    sum_reward_episode = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        while True:
            start_loop_s = time.perf_counter()
            action = policy.select_action(obs)
            action[:,3:6] = 0.0
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                print(f"Episode reward: {reward}")
                if reward > 0:
                    print("Success!")
                else:
                    print("Failure!")
                break
            dt_s = time.perf_counter() - start_loop_s
            # print("dt_s: ",dt_s)
            busy_wait(1 / control_fps - dt_s)
        sum_reward_episode.append(episode_reward)
    env.reset()
    env.close()
    logging.info(f"Success after 100 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode) * 100}%")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    config_path = parser.parse_arg("config_path")
    policy_path = Path(config_path).parent
    cfg.policy.pretrained_path = policy_path
    env_cfg = cfg.env
    env = make_robot_env(env_cfg,use_gamepad=True,render_mode="human",random_actor=True)
    policy =  make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env
    )
    policy = policy.eval()

    eval_policy(env, policy=policy, n_episodes=10,control_fps = cfg.env.fps)


if __name__ == "__main__":
    main()
