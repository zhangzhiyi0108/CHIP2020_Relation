from typing import Any

import torch

from common.loss.common_loss import CommonLoss


class SequenceCRFLoss(CommonLoss):
    def __init__(self, config):
        super(SequenceCRFLoss, self).__init__(config)

        # self._num_tag = self._config.data.num_tag
        # self.crf = CRF(self._num_tag).to(self._config.device)

    def forward(self, dict_outputs: dict) -> dict:
        """
        :param dict_outputs: {outputs, targets, sentence}
        :return: loss_dict: example {crf_loss, dae_loss, dice_loss, refactor_loss}
        """
        # emissions, target_sequence = dict_outputs
        emissions = dict_outputs['emissions']
        target_sequence = dict_outputs['target_sequence']

        loss_dict = dict()
        loss_crf = dict_outputs['loss_crf']
        loss_batch = loss_crf
        loss_dict['loss_batch'] = loss_batch

        return loss_dict
