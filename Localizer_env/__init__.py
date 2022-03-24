# -*- coding: utf-8 -*-

import logging
import gym
from gym.envs.registration import register

logger = logging.getLogger(__name__)

gym.envs.register(
     id='Local-v0',
     entry_point='Localizer_env.envs:localizer_env',
)
