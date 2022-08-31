import yaml

def read_params(config_path):
    '''Helper function to read the config file'''
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config