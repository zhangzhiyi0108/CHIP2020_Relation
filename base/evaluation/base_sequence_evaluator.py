# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/16 16:20

from abc import ABC, abstractmethod


class BaseSeqEvaluator(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _change_type(self, pred, target):
        pass

    @abstractmethod
    def evaluate(self, pred, target):
        # Send batch data
        pass

    @abstractmethod
    def _get_eval_result(self):
        # get processed eval result
        pass

    @abstractmethod
    def get_eval_output(self):
        # get the result(processed) of valid data or test data and return it to external interface
        # it also can to display eval result or not
        pass

    @abstractmethod
    def _print_table(self, List):
        # display eval result
        pass

    @abstractmethod
    def _write_csv(self, table):
        # write the eval result to csv_file
        pass
