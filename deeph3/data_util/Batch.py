from abc import ABCMeta, abstractmethod


class Batch(metaclass=ABCMeta):
    """A super class for H5 data batches

    Defines functions that batch classes must implement
    """

    def data(self):
        return self.features(), self.labels()

    def id(self):
        return self.id_

    @abstractmethod
    def features(self):
        pass

    @abstractmethod
    def labels(self):
        pass

