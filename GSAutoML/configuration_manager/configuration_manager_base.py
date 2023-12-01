from abc import ABC, abstractmethod

from GSAutoML.configuration_manager.singleton import SingletonABC


class ConfigurationManagerBase(ABC, metaclass=SingletonABC):
    @property
    @abstractmethod
    def automl_engine_path(self) -> str:
        """
        Name of the Project.
        """
        pass
    @property
    @abstractmethod
    def engine_host(self) -> str:
        """
        Name of the Project.
        """
        pass

    @property
    @abstractmethod
    def engine_port(self) -> int:
        """
        Name of the microservice.
        """
        pass

    @property
    @abstractmethod
    def algorithms_url(self) -> str:
        """
        Random Seed to be used throughout the application.
        """
        pass

    @property
    @abstractmethod
    def datasets_url(self) -> str:
        """
        Name of the microservice.
        """
        pass

    @property
    @abstractmethod
    def uni_meta_features_url(self) -> str:
        """
        Name of the microservice.
        """
        pass

    @property
    @abstractmethod
    def multi_meta_features_url(self) -> str:
        """
        Name of the microservice.
        """
        pass
