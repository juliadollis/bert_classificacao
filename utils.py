import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                node = getattr(self, key, None)
                if node is None or not isinstance(node, CfgNode):
                    node = CfgNode()
                    setattr(self, key, node)
                node.merge_from_dict(value)
            else:
                setattr(self, key, value)

    def merge_from_args(self, args):
        for arg in args:
            keyval = arg.split('=')
            assert len(keyval) == 2, "esperando cada argumento de substituição no formato --arg=value, recebeu %s" % arg
            key, val = keyval
            try:
                val = literal_eval(val)
            except:
                pass
            assert key[:2] == '--', f"o argumento {arg} não começa com '--'"
            key = key[2:]
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                if not hasattr(obj, k):
                    setattr(obj, k, CfgNode())
                obj = getattr(obj, k)
            leaf_key = keys[-1]
            if hasattr(obj, leaf_key):
                print(f"sobrescrevendo atributo de configuração {key} com {val}")
                setattr(obj, leaf_key, val)
            else:
                raise AttributeError(f"{key} não é um atributo existente na configuração")
