#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_model
# @Author   : LiuYan
# @Time     : 2020/8/23 18:34

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, dict_inputs: dict) -> dict:
        pass
