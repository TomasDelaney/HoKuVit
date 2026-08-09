import yaml


class Tee:
    """ Store console output for validation purposes"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def load_yaml_config(path):
    """Load a (possibly nested) YAML config and flatten it into a single
    dict of {arg_name: value}, suitable for parser.set_defaults(**cfg)."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    flat = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)   # nested section -> merge its keys up
        else:
            flat[key] = value    # already flat
    return flat
