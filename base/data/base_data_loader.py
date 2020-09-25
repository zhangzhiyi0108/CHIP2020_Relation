#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_data_loader
# @Author   : Xiaoming Liu
# @Time     : 2020/8/23 18:34
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):

    @abstractmethod
    def _load_data(self):
        """
            load raw data according to data config
        Returns:

        """
        pass

    @abstractmethod
    def load_train(self):
        pass

    @abstractmethod
    def load_valid(self):
        pass

    @abstractmethod
    def load_test(self):
        pass

    pass
