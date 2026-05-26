import os
import yaml

libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
)
config_file = os.path.join(libero_config_path, "config.yaml")


def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = os.path.dirname(os.path.abspath(__file__))
    else:
        benchmark_root_path = custom_location
    return {
        "benchmark_root": benchmark_root_path,
        "assets": os.path.join(benchmark_root_path, "./assets"),
    }


def get_libero_path(query_key):
    with open(config_file, "r") as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))
    for key in config:
        if not os.path.exists(config[key]):
            print(f"[Warning]: {key} path {config[key]} does not exist!")
    assert query_key in config, (
        f"Key {query_key} not found in config file {config_file}."
        f" Available keys: {list(config.keys())}"
    )
    return config[query_key]


def set_libero_default_path(
    custom_location=os.path.dirname(os.path.abspath(__file__)),
):
    new_config = get_default_path_dict(custom_location)
    with open(config_file, "w") as f:
        yaml.dump(new_config, f)


os.makedirs(libero_config_path, exist_ok=True)
if not os.path.exists(config_file):
    with open(config_file, "w") as f:
        yaml.dump(get_default_path_dict(), f)
