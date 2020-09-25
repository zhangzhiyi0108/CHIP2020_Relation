import math
import torch
from torch import nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

import numpy as np
from common.model.common_model import CommonModel

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiRNNCRF(CommonModel):
    def __init__(self, model_config):
        super(BiRNNCRF, self).__init__(model_config)

        self._dim_embedding = self._config.model.dim_embedding
        self._dim_hidden = self._config.model.dim_hidden

        self._num_vocab = self._config.data.num_vocab
        self._num_tag = self._config.data.num_tag

        self.embedding = nn.Embedding(self._num_vocab, self._dim_embedding)

        # Maps the output of the RNN into tagger space.
        self.hidden2tagging = nn.Linear(self._dim_hidden, self._num_tag)

        self.rnn = nn.RNN(
            input_size=self._dim_embedding, hidden_size=self._dim_hidden,
            num_layers=self._config.model.num_layer, bidirectional=self._config.model.bidirectional,
            batch_first=self._config.model.batch_first, dropout=self._config.learn.dropout_rate
        )

        pass

    def _init_hidden(self, batch_size):
        num_layer = self._config.model.num_layer
        dim_hidden = self._dim_hidden // 2
        hidden = (
            torch.zeros(num_layer, batch_size, dim_hidden),
            torch.zeros(num_layer, batch_size, dim_hidden)
        )
        return hidden

    def forward(self, dict_inputs: dict) -> dict:
        dict_outputs = dict()
        # dict_outputs.update(dict_inputs)

        input_sequence, input_length = dict_inputs.text
        tag_sequence = dict_inputs.tag
        dict_outputs['tag_sequence'] = tag_sequence

        embed_sequence = self.embedding(input_sequence)

        # batch_size = len(input_length)
        # hidden_init = self._init_hidden(batch_size)

        padded_sequence = pack_padded_sequence(input=embed_sequence, lengths=input_length)
        output_sequence, hidden_sequence = self.rnn(padded_sequence)

        tagging_sequence = self.hidden2tagging(hidden_sequence)

        dict_outputs['input_sequence'] = input_sequence
        dict_outputs['input_length'] = input_length
        dict_outputs['output_sequence'] = output_sequence
        dict_outputs['hidden_sequence'] = hidden_sequence
        dict_outputs['tagging_sequence'] = tagging_sequence

        return dict_outputs

    pass


class BiLSTMCRF(CommonModel):

    def __init__(self, config):
        super(BiLSTMCRF, self).__init__(config)

        self._dim_embedding = self._config.model.dim_embedding
        self._dim_hidden = self._config.model.dim_hidden
        self._num_layer = self._config.model.num_layer

        self._num_vocab = self._config.data.num_vocab
        self._num_tag = self._config.data.num_tag

        self._dropout = self._config.learn.dropout_rate

        self._embedding = nn.Embedding(self._num_vocab, self._dim_embedding)
        self._rnn = nn.LSTM(
            input_size=self._dim_embedding, hidden_size=self._dim_hidden // 2,
            bidirectional=True, num_layers=self._num_layer, dropout=self._dropout
        )
        self._hidden2label = nn.Linear(self._dim_hidden, self._num_tag)
        self._crf = CRF(self._num_tag)

    def __init_hidden(self, batch_size=None):
        h0 = torch.zeros(self._num_layer * 2, batch_size, self._dim_hidden // 2).to(self._device)
        c0 = torch.zeros(self._num_layer * 2, batch_size, self._dim_hidden // 2).to(self._device)
        return h0, c0

    def forward(self, dict_inputs: dict) -> dict:
        """
        :param batch_data: {sentence, sent_lengths}
        :return:
        """
        dict_outputs = dict()
        dict_outputs['batch_data'] = dict_inputs

        input_sequence, input_length = dict_inputs.text
        if hasattr(dict_inputs, 'tag'):
            target_sequence = dict_inputs.tag
            dict_outputs['target_sequence'] = target_sequence
        # inputs, target_sequence = batch_data
        # input_seq, input_pos, input_chuck = inputs
        # input_sequence, input_length = input_seq

        dict_outputs['input_sequence'] = input_sequence

        # mask_crf = torch.ne(input_sequence, 1)
        input_embed = self._embedding(input_sequence)
        input_padded = pack_padded_sequence(input=input_embed, lengths=input_length)

        hidden_init = self.__init_hidden(batch_size=len(input_length))
        rnn_out, rnn_hidden = self._rnn(input_padded, hidden_init)
        rnn_hidden_padded, new_batch_size = pad_packed_sequence(sequence=rnn_out)

        assert torch.equal(input_length, new_batch_size.to(input_sequence.device))

        emissions = self._hidden2label(rnn_hidden_padded)
        dict_outputs['emissions'] = emissions
        mask_crf = torch.ne(input_sequence, 1).to(input_sequence.device)
        dict_outputs['mask'] = mask_crf
        if hasattr(dict_inputs, 'tag'):
            loss_crf = -self._crf(emissions, target_sequence, mask=mask_crf) / target_sequence.size(1)
            dict_outputs['loss_crf'] = loss_crf

        outputs = self._crf.decode(emissions=emissions, mask=mask_crf)
        dict_outputs['outputs'] = outputs

        return dict_outputs

    def loss(self, dict_outputs):
        emissions = dict_outputs['emissions']
        target_sequence = dict_outputs['target_sequence']
        mask = dict_outputs['mask']
        loss_crf = -self._crf(
            emissions, target_sequence, mask
        ) / target_sequence.size(1)
        dict_outputs['loss_crf'] = loss_crf
        pass

    def decode(self, batch_data):
        dict_outputs = self.forward(batch_data)
        emissions = dict_outputs['emissions']
        mask = dict_outputs['mask']
        outputs = self._crf.decode(
            emissions=emissions, mask=mask
        )
        dict_outputs['outputs'] = outputs

        pass


class TransformerEncoderModel(CommonModel):

    def __init__(self, config):
        super(TransformerEncoderModel, self).__init__(config)
        self.src_mask = None
        self._dim_embedding = self._config.model.dim_embedding
        self._dim_hidden = self._config.model.dim_hidden
        self._num_layer = self._config.model.nlayer

        self._num_vocab = self._config.data.num_vocab
        self._num_tag = self._config.data.num_tag

        self._dropout = self._config.learn.dropout_rate
        self.pos_encoder = PositionalEncoding(self._dim_embedding, self._dropout)
        self._embedding = nn.Embedding(self._num_vocab, self._dim_embedding)
        encoder_layers = TransformerEncoderLayer(self._dim_hidden, self._config.model.nhead, self._config.model.nhid,
                                                 self._config.learn.dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self._config.model.nlayer)
        self.lstm = nn.LSTM(
            input_size=self._dim_embedding, hidden_size=self._dim_hidden // 2,
            bidirectional=True, num_layers=self._num_layer, dropout=self._dropout
        )
        self._hidden2label = nn.Linear(self._dim_hidden, self._num_tag)
        self._crf = CRF(self._num_tag)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _get_src_key_padding_mask(self, text_len, seq_len):
        batchszie = text_len.size(0)
        list1 = []
        for i in range(batchszie):
            list2 = []
            list2.append([False for i in range(text_len[i])] + [True for i in range(seq_len - text_len[i])])
            list1.append(list2)
        src_key_padding_mask = torch.tensor(np.array(list1)).squeeze(1)
        return src_key_padding_mask

    def init_hidden_lstm(self):
        return (torch.randn(2, self._config.data.batch_size, self._dim_hidden // 2).to(self._device),
                torch.randn(2, self._config.data.batch_size, self._dim_hidden // 2).to(self._device))

    def init_weights(self):
        initrange = 0.1
        self.linner.bias.data.zero_()
        self.linner.weight.data.uniform_(-initrange, initrange)

    def forward(self, dict_inputs: dict) -> dict:
        self.hidden = self.init_hidden_lstm()

        dict_outputs = dict()
        dict_outputs['batch_data'] = dict_inputs

        input_sequence, input_length = dict_inputs.text
        if hasattr(dict_inputs, 'tag'):
            target_sequence = dict_inputs.tag
            dict_outputs['target_sequence'] = target_sequence

        dict_outputs['input_sequence'] = input_sequence

        # mask_crf = torch.ne(input_sequence, 1)
        input_embed = self._embedding(input_sequence)

        mask_crf = torch.ne(input_sequence, 1).to(input_sequence.device)
        dict_outputs['mask'] = mask_crf
        transformer_out = self.transformer_forward(input_embed, input_sequence, input_length)
        lstm_out, _ = self.lstm(transformer_out)
        emissions = self._hidden2label(lstm_out)
        dict_outputs['emissions'] = emissions
        if hasattr(dict_inputs, 'tag'):
            loss_crf = -self._crf(emissions, target_sequence, mask=mask_crf) / target_sequence.size(1)
            dict_outputs['loss_crf'] = loss_crf

        outputs = self._crf.decode(emissions=emissions, mask=mask_crf)
        dict_outputs['outputs'] = outputs
        return dict_outputs

    def transformer_forward(self, input_embed, input_sequence, input_length):
        src_key_padding_mask = self._get_src_key_padding_mask(input_length, input_sequence.size(0))
        if self.src_mask is None or self.src_mask.size(0) != len(input_sequence):
            mask = self._generate_square_subsequent_mask(len(input_sequence))
            self.src_mask = mask
        src = input_embed * math.sqrt(self._dim_embedding)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask.to(input_sequence.device),
                                          )
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
