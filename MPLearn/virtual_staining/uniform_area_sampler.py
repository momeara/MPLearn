
from typing import Optional, Generator, Tuple
import numpy as np
import torch
import torchio as tio



class UniformAreaSampler(tio.data.sampler.RandomSampler):
    """Randomly extract patches from area with uniform probability
       and full depth

    Args:
        patch_size: See :class:`~torchio.data.PatchSampler`.
    """

    def get_probability_map(self, subject: tio.Subject) -> torch.Tensor:
        return torch.ones(1, *subject.spatial_shape)

    def __call__(
            self,
            dpc_subject: tio.Subject,
            stain_subject: tio.Subject,            
            num_patches: int = None,
            ) -> Generator[Tuple[tio.Subject, tio.Subject], None, None]:

        spatial_shapes = [
            image.spatial_shape[-2:]
            for image in dpc_subject.get_images(intensity_only = False)]
        if np.any(self.patch_size > spatial_shapes):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than the dpc image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)

        spatial_shapes = [
            image.spatial_shape[-2:]
            for image in stain_subject.get_images(intensity_only = False)]
        if np.any(self.patch_size > spatial_shapes):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than stain image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)

        valid_range = spatial_shapes[0] - self.patch_size
        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            index_ini = [ torch.randint(x + 1, (1,)).item() for x in valid_range ]
            # just sample the x and y coordinates but take the rest of the dimensions
            # time and z coordinates
            dpc_patch = self.extract_patch(dpc_subject, np.asarray(index_ini))
            stain_patch = self.extract_patch(stain_subject, np.asarray(index_ini))
            yield (dpc_patch, stain_patch)
            if num_patches is not None:
                patches_left -= 1

    def extract_patch(
            self,
            subject: tio.Subject,
            index_ini: Tuple[int, int]
            ) -> tio.Subject:
        cropped_subject = self.crop(
            subject = subject,
            index_ini = np.append([0], index_ini),
            patch_size = np.append(subject.shape[1], self.patch_size))
        cropped_subject['index_ini'] = np.array(index_ini).astype(int)
        return cropped_subject
