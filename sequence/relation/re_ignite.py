from typing import Sequence, Optional, Union

import torch
from ignite.contrib.handlers import BasicTimeProfiler
from ignite.engine import create_supervised_trainer
from ignite.utils import setup_logger

from sequence.tagger.seq_data import SequenceDataLoader
from sequence.tagger.seq_loss import SequenceCRFLoss
from sequence.tagger.seq_model import BiLSTMCRF

if __name__ == '__main__':
    # Device
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file = 'seq_config.yml'
    import dynamic_yaml

    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)
    config.device = _device

    dataloader = SequenceDataLoader(config)
    train_loader = dataloader.train()


    def prepare_batch(
            batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
    ):
        dict_inputs = dict()
        inputs, target_sequence = dict_inputs
        input_seq, input_pos, input_chuck = dict_inputs
        input_sequence, input_length = input_seq

        (x_tokens, x_length), x_pos, x_chunk = x_inputs


        return


    model = BiLSTMCRF(config).to(config.device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.learn.learning_rate,
        weight_decay=config.learn.weight_decay
    )
    criterion = SequenceCRFLoss(config).to(config.device)

    trainer = create_supervised_trainer(
        model, optimizer, criterion,
        prepare_batch=prepare_batch,
        device=config.device
    )

    # Create an object of the profiler and attach an engine to it
    profiler = BasicTimeProfiler()
    profiler.attach(trainer)

    trainer.logger = setup_logger("trainer")

    trainer.run(train_loader, max_epochs=100)
    pass
