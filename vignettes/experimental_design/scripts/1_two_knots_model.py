
import os
from argparse import ArgumentParser
import torch
import pyro
import pytorch_lightning
import MPLearn.embedding_notebook
from MPLearn.experimental_design import hit_rate_model
from MPLearn.experimental_design import hill_model
from MPLearn.experimental_design import toy_model


root_dir = os.path.dirname(
    os.path.realpath("~/opt/MPLearn/vignettes/dose_response/intermediate_data"))


def run_toy_model():
    SEED = 2334
    torch.manual_seed(SEED)
    #np.random.seed(SEED)
    pyro.clear_param_store()


    logger = pytorch_lightning.loggers.TestTubeLogger(
        save_dir=root_dir,
        name="test_top_model")
    logger.experiment.tag({'design_size': 51, 'optimizer': 'ace'}) 

    parent_parser = ArgumentParser(add_help=False)

    parser = toy_model.ToyModel.add_model_specific_args(
        parent_parser, root_dir)
    hparams = parser.parse_args(args=[
        '--device', 'cpu:0',
        '--start_lr', '.01',
        '--end_lr', '.01',
        '--design_size', '51'])
    model = toy_model.ToyModel(hparams)

    trainer = pytorch_lightning.Trainer(
        nb_sanity_val_steps=0,
        max_nb_epochs=2000,
        logger=logger)
    trainer.fit(model)
run_toy_model()
