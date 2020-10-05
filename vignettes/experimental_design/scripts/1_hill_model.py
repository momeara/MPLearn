


import os
from argparse import ArgumentParser
import torch
import pyro
import pytorch_lightning
import MPLearn.embedding_notebook
from MPLearn.experimental_design import hit_rate_model
from MPLearn.experimental_design import hill_model
from MPLearn.experimental_design import toy_model


root_dir = os.path.dirname("intermediate_data")


def check_hill_model():
    SEED = 2334
    torch.manual_seed(SEED)
    #np.random.seed(SEED)
    pyro.clear_param_store()


    logger = pytorch_lightning.loggers.TestTubeLogger(
        save_dir=root_dir,
        name="test_hill_model")
    logger.experiment.tag({'design_size': 10, 'optimizer': 'ace'}) 

    parent_parser = ArgumentParser(add_help=False)

    parser = hill_model.HillModel.add_model_specific_args(
        parent_parser, root_dir)
    hparams = parser.parse_args(args=[
        '--device', 'cpu:0',
        '--start_lr', '.000001',
        '--end_lr', '.000001'])
    model = hill_model.HillModel(hparams)

    print(f"Prior Entryp: {model.prior_entropy()}")
check_hill_model()



def run_hill_model():
    SEED = 2334
    torch.manual_seed(SEED)
    #np.random.seed(SEED)
    pyro.clear_param_store()


    logger = pytorch_lightning.loggers.TestTubeLogger(
        save_dir=root_dir,
        name="test_hill_model")
    logger.experiment.tag({'design_size': 10, 'optimizer': 'ace'}) 

    parent_parser = ArgumentParser(add_help=False)

    parser = hill_model.HillModel.add_model_specific_args(
        parent_parser, root_dir)
    hparams = parser.parse_args(args=[
        '--device', 'cpu:0',
        '--start_lr', '.000001',
        '--end_lr', '.000001'])
    model = hill_model.HillModel(hparams)

    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=0,
        max_epochs=2000,
        logger=logger,
        gradient_clip_val=0.5,
        track_grad_norm=2)
    trainer.fit(model)
run_hill_model()
