import yaml
import os


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    def __init__(self, cfg_path: str):
        with open(cfg_path, 'r') as f:
            try:
                self.params = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                self.params = None
        
        # make path absolute
        LIB_DIR = os.path.abspath(__file__).split('text_xai')[0]
        self.params["base_path"] = LIB_DIR + 'data' + self.params["base_path"].split('data')[-1]
