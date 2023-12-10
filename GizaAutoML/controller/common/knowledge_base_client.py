import requests
from requests.adapters import HTTPAdapter
from GizaAutoML.configuration_manager.configuration_manager import ConfigurationManager


class KnowledgeBaseClient:

    def __init__(self):
        self.config_manager = ConfigurationManager()
        engine_host = self.config_manager.engine_host
        engine_port = self.config_manager.engine_port
        automl_engine_path = self.config_manager.automl_engine_path
        self.base_url = f'http://{engine_host}:{engine_port}{automl_engine_path}'
        self.algorithms_url = self.config_manager.algorithms_url
        self.datasets_url = self.config_manager.datasets_url
        self.uni_meta_features_url = self.config_manager.uni_meta_features_url
        self.multi_meta_features_url= self.config_manager.multi_meta_features_url
        self.headers = {'Content-Type': 'application/json'}

    def get_uni_variate_meta_features(self):
        try:
            response = requests.get(self.base_url + self.uni_meta_features_url)

            if response.status_code == 200:
                data = response.json()['result']
                return data

            else:
                print(f"Request failed with status code {response.status_code}")
                print(response.text)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")

    def add_uni_variate_meta_features(self, data):
        with requests.Session() as session:
            adapter = HTTPAdapter(max_retries=5)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            response = session.request(method="POST",
                                       url=self.base_url+self.uni_meta_features_url,
                                       headers=self.headers,
                                       json=data,
                                       timeout=300)

            return response

    def add_multi_variate_meta_features(self, data):
        with requests.Session() as session:
            adapter = HTTPAdapter(max_retries=5)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            # Replace NaN values with None if exists
            data = {key: None if value != value else value for key, value in data.items()}
            response = session.request(method="POST",
                                       url=self.base_url+self.multi_meta_features_url,
                                       headers=self.headers,
                                       json=data,
                                       timeout=300)

            return response

    def get_datasets(self, dataset_name):
        try:
            response = requests.get(self.base_url + self.datasets_url, {'dataset_name': dataset_name})

            if response.status_code == 200:
                data = response.json()['result']
                return data

            else:
                print(f"Request failed with status code {response.status_code}")
                print(response.text)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")

    def add_dataset(self, data):
        with requests.Session() as session:
            adapter = HTTPAdapter(max_retries=5)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            response = session.request(method="POST",
                                       url=self.base_url+self.datasets_url,
                                       headers=self.headers,
                                       json=data,
                                       timeout=300)

            return response

    def get_algorithms_performance(self):
        try:
            response = requests.get(self.base_url + self.algorithms_url)

            if response.status_code == 200:
                data = response.json()['result']
                return data

            else:
                print(f"Request failed with status code {response.status_code}")
                print(response.text)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")

    def add_algorithms_performance(self, data):
        with requests.Session() as session:
            adapter = HTTPAdapter(max_retries=5)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            response = session.request(method="POST",
                                       url=self.base_url+self.algorithms_url,
                                       headers=self.headers,
                                       json=data,
                                       timeout=300)
            return response
            return response
