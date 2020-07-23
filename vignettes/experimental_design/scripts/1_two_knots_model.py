
import os
from argparse import ArgumentParser
import torch
import pyro
import pytorch_lightning
import MPLearn.embedding_notebook
from MPLearn.experimental_design import toy_model


root_dir = os.path.dirname("intermediate_data")


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

    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu:0"

    hparams = parser.parse_args(args=[
        '--device', device,
        '--optimizer_name', 'exponential',
        '--exponential_lr_start', '.001',
        '--exponential_lr_end', '.001',
        '--num_samples', '20',
        '--design_size', '51'])
    model = toy_model.ToyModel(hparams)

    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=0,
        max_epochs=2000,
        gpus=torch.cuda.device_count(),
        logger=logger)

    print("Training model...")
    trainer.fit(model)
run_toy_model()
