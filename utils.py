# 读取Yaml文件
def read_yaml(path):
    import yaml
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return f'错误：{path}不存在!'
    except yaml.YAMLError as e:
        return f'{path}文件解析错误！{e}'

# 读取JSON文件
def read_json(path):
    import json
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return f'错误：{path}不存在!'
    except json.JSONDecodeError as e:
        return f'{path}文件解析错误！{e}'