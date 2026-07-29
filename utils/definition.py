import itertools


def expand(e, quick_kw, grid_kw):
    e = expand_quick_def(e, quick_kw)
    e = expand_grid_def(e, grid_kw)
    return e


def expand_grid_def(e, grid_kw):
    if isinstance(e, dict):
        expanded_instances = []
        if grid_kw in e:
            grid_combinations = [dict(zip(e[grid_kw].keys(), combination)) for combination in itertools.product(*e[grid_kw].values())]
            del e[grid_kw]
            expanded_instances.extend([{**e, **comb} for comb in grid_combinations])
        else:
            expanded_instances.append(e)
        return expanded_instances
    elif isinstance(e, (list, tuple)):
        return list(itertools.chain(*[expand_grid_def(i, grid_kw) for i in e]))
    else:
        raise TypeError('invalid type of e')


def expand_quick_def(e, quick_kw):
    if isinstance(e, dict):
        expanded_instances = []
        if quick_kw in e:
            keys = list(e[quick_kw].keys())
            vals = list(e[quick_kw].values())

            if len(set(map(len, vals))) != 1:
                raise ValueError('length mismatch')

            combinations = [dict(zip(keys, v)) for v in zip(*vals)]
            del e[quick_kw]
            expanded_instances.extend([{**e, **comb} for comb in combinations])
        else:
            expanded_instances.append(e)
        return expanded_instances
    elif isinstance(e, (list, tuple)):
        return list(itertools.chain(*[expand_quick_def(i, quick_kw) for i in e]))
    else:
        raise TypeError('invalid type of e')
