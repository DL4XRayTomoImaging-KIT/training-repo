import builtins
from timeit import default_timer as timer
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

def cfg_ut(group_name, logline, force=False):
    def cfg_unroll(f):
        def wrapper(*args, **kwargs):
            print('>>>>>>>>>>>>>>>>', logline)
            s = timer()

            is_configured = False
            if (group_name is not None) and ('cfg' in kwargs.keys()):
                cfg = kwargs.pop('cfg')
                if (group_name in cfg.keys()) and (cfg[group_name] is not None):
                    if isinstance(cfg[group_name], ListConfig):
                        args += tuple(OmegaConf.to_container(cfg[group_name]))
                    elif isinstance(cfg[group_name], DictConfig):
                        kwargs.update(OmegaConf.to_container(cfg[group_name]))
                    else:
                        raise ValueError(f'Unknown type of group config: {type(cfg[group_name])}')
                    is_configured = True
            else:
                is_configured = True # nothing to config

            if is_configured or force:
                r = f(*args, **kwargs)
            else:
                r = None
            d = timer() - s
            if d < 10:
                print(f'<<<<<<<<<<<<<<<< done in {d:.2} sec.')
            else:
                print(f'<<<<<<<<<<<<<<<< done in {int(d)} sec.')

            return r
        return wrapper
    return cfg_unroll

def check_exclusion(cfg):
    to_exclude = False
    if 'exclude_combinations' not in cfg.keys():
        return False
    if cfg['exclude_combinations'] is None:
        return False
    for combo in cfg['exclude_combinations']:
        local_rule_in_force = True
        for k, v in combo.items():
            current_section = cfg
            for k_ in k.split('.'):
                current_section = current_section[k_]
            if current_section != v:
                local_rule_in_force = False
        to_exclude = to_exclude or local_rule_in_force
    return to_exclude
