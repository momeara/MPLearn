
import math
from argparse import ArgumentParser
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

class Linear2p5(pl.LightningModule):
    """
    Virtual staining 3D -> 2D
    linear 3D convolution + maximum projection

    to use

    root_dir = 'runs'
    parser = ArgumentParser(add_help=False)
    perser = Linear2p5.add_model_specific_args(parser, root_dir)
    hparams = parser.parse_args(
      args=['--learning-rate', '0.02'])
    model = Linear2p5(hparams)

    

    trainer = pl.Trainer(
        gpus = 1,
        logger = mlf_loger)
    """

    def __init__(self, hparams):
        super(Linear2p5, self).__init__()
        self.save_hyperparameters(hparams)
        self._check_hparams()

        padding = int((hparams.kernel_size - 1) / 2)
        self.conv = torch.nn.Conv3d(
            in_channels = 1,
            out_channels = 1,
            kernel_size = (self.hparams.kernel_size,)*3,
            stride = (1,1,1),
            padding = (padding,)*3)
        self.loss = torch.nn.MSELoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help = False)
        parser.add_argument("--kernel_size", type = int, default = 15)
        parser.add_argument("--learning_rate", type = float, default = 1e-3)
        return parser

    def _check_hparams(self):
        # want d_in = d_out
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        # d_out = (d_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        # d_out = (d_in + 2 * padding - (kernel_size - 1) - 1)/1 + 1
        # d_out = d_in + 2 * padding - (kernel_size - 1)
        # padding = (kernel_size - 1) / 2 
        # padding and kernel_size must be an integers, so kernel_size must be odd
        if self.hparams.kernel_size % 2 == 0:
            raise Exception("Kernel size must be odd to have the output the same size as the input")

        
    def forward(self, x):
        return torch.max(self.conv(x), dim = 2)[0]
        
    def training_step(self, batch, batch_idx):
        dpc_subject, stain_subject = batch
        dpc = dpc_subject['dpc']['data'].float()
        virtual_stain = self(dpc)
        stain = torch.max(stain_subject['stain']['data'].float(), dim = 2)[0]
        loss = self.loss(virtual_stain, stain)
        self.log('train_loss', loss)
        return loss
            

    def validation_step(self, batch, batch_idx):
        dpc_subject, stain_subject = batch
        dpc = dpc_subject['dpc']['data'].float()
        virtual_stain = self(dpc)
        stain = torch.max(stain_subject['stain']['data'].float(), dim = 2)[0]
        self.logger.experiment.add_image(
            tag = "virtual_stain",
            img_tensor = torchvision.utils.make_grid(
                tensor = virtual_stain),
            global_step = batch_idx)
        self.logger.experiment.add_image(
            tag = "stain",
            img_tensor = torchvision.utils.make_grid(
                tensor = stain),
            global_step = batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)

