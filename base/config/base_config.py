#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_load_config
# @Author   : Xiaoming Liu
# @Time     : 2020/8/25 15:55
from abc import abstractmethod, ABC


class BaseConfig(ABC):

    @abstractmethod
    def __init__(self):
        super(BaseConfig, self).__init__()

    @abstractmethod
    def _load_config(self, dict_paths: dict):
        """
        Add the config you need.
        :param dict_paths: *.yml path
        :return: config(YamlDict)
        """
        pass
