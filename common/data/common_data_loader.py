from __future__ import unicode_literals, print_function, division

from abc import ABC

from base.data.base_data_loader import BaseDataLoader


class CommonDataLoader(BaseDataLoader, ABC):

    def __init__(self, data_config):
        self._config = data_config
        pass

    pass
