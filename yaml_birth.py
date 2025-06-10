import yaml

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip()]
    
yaml_dict = {'dependencies': requirements}

with open('environment.yaml', 'w') as f:
    yaml.dump(yaml_dict, f)