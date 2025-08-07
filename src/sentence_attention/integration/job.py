

def accelerate_config_by_instance_type(instance_type: str, workdir_prefix: str) -> str:

    if instance_type == 'a100.8gpu':
        return f'{workdir_prefix}/configs/accelerate/8gpu.yaml'
    elif instance_type == 'a100.6gpu':
        return f'{workdir_prefix}/configs/accelerate/6gpu.yaml'
    elif instance_type == 'a100.4gpu':
        return f'{workdir_prefix}/configs/accelerate/4gpu.yaml'
    elif instance_type == 'a100.2gpu':
        return f'{workdir_prefix}/configs/accelerate/2gpu.yaml'
    elif instance_type == 'a100.1gpu':
        return f'{workdir_prefix}/configs/accelerate/1gpu.yaml'
    else:
        raise ValueError(f"Unknown instance type: {instance_type}")
