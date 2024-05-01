import argparse
import ruamel.yaml
from pathlib import Path


def config(yaml_file):
    yaml_file = Path(yaml_file)
    if not yaml_file.is_file():
        raise FileNotFoundError('The file {} does not exist'.
                                format(str(yaml_file)))
    with open(str(yaml_file), 'r') as yfile:
        yaml_dict = ruamel.yaml.safe_load(yfile)
    print(yaml_dict)
    return yaml_dict


def main():
    parser = argparse.ArgumentParser(
        description='Configuring the parameters of a dataset')

    parser.add_argument('--yaml-file',
                        type=Path,
                        default='HOUDINI/Yaml/PORTEC.yaml',
                        metavar='DIR',
                        help='path to the yaml file')

    args = parser.parse_args()
    config(args.yaml_file)


if __name__ == '__main__':
    main()
