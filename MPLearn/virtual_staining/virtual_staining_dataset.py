

from typing import *
import copy

import torch
import torchio as tio

class VirtualStainingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dpc_subjects: Sequence[tio.Subject],
            stain_subjects: Sequence[tio.Subject],
            transform: Optional[Callable] = None,            
            load_getitem: bool = True
            ):
        
        self._parse_subjects_list(dpc_subjects)
        self._dpc_subjects = dpc_subjects
        
        self._parse_subjects_list(stain_subjects)
        self._stain_subjects = stain_subjects

        self._transform: Optional[Callable]
        self.set_transform(transform)

        self.load_getitem = load_getitem

        assert len(dpc_subjects) == len(stain_subjects)

    def __len__(self):
        assert len(self._dpc_subjects) == len(self._stain_subjects)        
        return len(self._dpc_subjects)

    def __getitem__(self, index: int) -> Tuple[tio.Subject, tio.Subject]:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')

        dpc_subject = self._dpc_subjects[index]
        dpc_subject = copy.deepcopy(dpc_subject)  # cheap since images not loaded yet
        if self.load_getitem:
            dpc_subject.load()

        stain_subject = self._stain_subjects[index]
        stain_subject = copy.deepcopy(stain_subject)  # cheap since images not loaded yet
        if self.load_getitem:
            stain_subject.load()
            
        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            dpc_subject, stain_subject = self._transform(dpc_subject, stain_subject)
            
        return (dpc_subject, stain_subject)

    def dry_iter(self):
        """Return the internal list of subjects.

        This can be used to iterate over the subjects without loading the data
        and applying any transforms::

        >>> names = [
             (dpc_subject.name, stain_subject.name)
             for dpc_subject, stain_subject in dataset.dry_iter()]
        """
        return zip(self._dpc_subjects, self._stain_subjects)

    def set_transform(self, transform: Optional[Callable]) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: Callable object, typically an subclass of
                :class:`VirtualStainingTransform`.
        """
        if transform is not None and not callable(transform):
            message = (
                'The transform must be a callable object,'
                f' but it has type {type(transform)}'
            )
            raise ValueError(message)
        self._transform = transform
        
    @staticmethod
    def _parse_subjects_list(subjects_list: Iterable[tio.Subject]) -> None:
        # Check that it's an iterable
        try:
            iter(subjects_list)
        except TypeError as e:
            message = (
                f'Subject list must be an iterable, not {type(subjects_list)}'
            )
            raise TypeError(message) from e

        # Check that it's not empty
        if not subjects_list:
            raise ValueError('Subjects list is empty')

        # Check each element
        for subject in subjects_list:
            if not isinstance(subject, tio.Subject) :
                message = (
                    'Subjects list must contain instances of torchio.Subject,'
                    f' not "{type(dpc_subject)}"'
                )
                raise TypeError(message)
