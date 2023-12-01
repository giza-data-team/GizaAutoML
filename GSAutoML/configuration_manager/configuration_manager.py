import os
from pathlib import Path
from decouple import Config, RepositoryEnv

from GSAutoML.configuration_manager.configuration_manager_base import ConfigurationManagerBase


class ConfigurationManager(ConfigurationManagerBase):
    _BASE_DIR = os.getenv("CONF_PATH", os.path.join(Path(__file__).resolve().parent.parent,
                                                    'resources'))
    _FILE_NAME = f'config-dev.env'
    _FILE_PATH = os.path.join(_BASE_DIR, _FILE_NAME)
    config = Config(RepositoryEnv(_FILE_PATH))
    @property
    def automl_engine_path(self) -> str:
        return self.config('AUTOML_ENGINE_PATH')
    @property
    def engine_host(self) -> str:
        return self.config('ENGINE_HOST')

    @property
    def engine_port(self) -> str:
        return self.config('ENGINE_PORT')

    @property
    def algorithms_url(self) -> int:
        return self.config('ALGORITHMS_URL')

    @property
    def datasets_url(self) -> str:
        return self.config('DATASETS_URL')

    @property
    def uni_meta_features_url(self) -> str:
        return self.config('UNI_META_FEATURES_URL')

    @property
    def multi_meta_features_url(self) -> str:
        return self.config('MULTI_META_FEATURES_URL')