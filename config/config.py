import json

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    # if you have config.json file, can use load function
    @classmethod
    def load(clscls, file):
        with open(file, 'r', encoding = 'utf-8') as f:
            config = json.loads(f.read())
            return Config(config)
