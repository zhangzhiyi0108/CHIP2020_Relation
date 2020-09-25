#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_load_config
# @Author   : Xiaoming Liu
# @Time     : 2020/8/25 15:55
from abc import abstractmethod, ABC

import dynamic_yaml


class CommonConfig(ABC):

    def __init__(self):
        super(CommonConfig, self).__init__()
        self._config_file = "common_config.yml"
        self._load_config()
        pass

    def _load_config(self):
        with open(self._config_file, mode='r', encoding='UTF-8') as f:
            config = dynamic_yaml.load(f)
        return config
        pass
