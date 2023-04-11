from abc import ABCMeta, abstractmethod
from reporters.no_reporter import NoReporter

class Writer(metaclass=ABCMeta):
    def __init__(self, reporter=NoReporter()) -> None:
        self.reporter = reporter

    @abstractmethod
    def write(self, dones, infos, rewards):
        raise NotImplementedError('Implement me')
    

class NoWriter(Writer):
    def __init__(self, reporter=NoReporter()) -> None:
        super().__init__(reporter)

    def write(self, dones, infos, rewards):
        pass