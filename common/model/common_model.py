from abc import ABC

from base.model.base_model import BaseModel


class CommonModel(BaseModel, ABC):
    def __init__(self, model_config):
        super(CommonModel, self).__init__()
        self._config = model_config
        self._device = self._config.device
        pass

    pass
