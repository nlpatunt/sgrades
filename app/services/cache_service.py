import pickle
from typing import Dict, Any

class DatasetCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    def get_dataset(self, dataset_name: str):
        return self.cache.get(dataset_name)
    
    def set_dataset(self, dataset_name: str, data: Any):
        self.cache[dataset_name] = data