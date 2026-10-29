"""Source-isolated dropout propagation: no Trainer/torch/runtime imports."""

def test_feature_dropout_reload_source_contract():
    import ast
    import json
    from pathlib import Path
    from types import SimpleNamespace as NS
    S = Path(__file__).parents[1]
    p = S / 'chess_anti_engine/train/trainer.py'
    tree = ast.parse(p.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    kwargs_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trainer_kwargs_from_config')
    kw_dict = next(n.value for n in kwargs_fn.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and (n.target.id == 'kw'))
    assert isinstance(kw_dict, ast.Dict)
    startup_expr = next((v for k, v in zip(kw_dict.keys, kw_dict.values) if isinstance(k, ast.Constant) and k.value == 'feature_dropout_p'))
    assert isinstance(startup_expr, ast.Call)
    startup_default = startup_expr.args[1]
    assert isinstance(startup_default, ast.Constant)
    assert startup_default.value == 0.3
    trial_tree = ast.parse((S / 'chess_anti_engine/tune/trial_config.py').read_text())
    trial_cls = next(n for n in trial_tree.body if isinstance(n, ast.ClassDef) and n.name == 'TrialConfig')
    trial_default = next(n.value for n in trial_cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and (n.target.id == 'feature_dropout_p'))
    assert trial_default is not None
    assert ast.literal_eval(trial_default) == 0.0
    start = next((i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_fdp' for t in n.targets)))
    body = init.body[start:start + 3]
    ns: dict[str, object] = {'self': NS(), 'feature_dropout_p': 0.3, 'fdp_king_safety': -1.0, 'fdp_pins': 0.0, 'fdp_pawns': None, 'fdp_mobility': 0.8, 'fdp_outposts': -2.0}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(p), 'exec'), ns)
    trainer = ns['self']
    assert isinstance(trainer, NS)
    trainer.feature_dropout_p = 0.3
    trainer.rebuild_sf_targets = False
    trainer.sync_search_width = lambda _x: None
    trainer.set_sf_target_rebuild = lambda **_kw: False
    p = S / 'chess_anti_engine/tune/trainable_config_ops.py'
    tree = ast.parse(p.read_text())
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sync_trainer_weights')
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sync_feature_dropout')
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), helper, f], type_ignores=[])
    ns2: dict[str, object] = {'_apply_lr_gamma_weights': lambda *_a, **_k: None, 'resolve_sf_target_params': lambda _c: None, '_dynamic_sf_wdl_weight': lambda **_k: None}
    exec(compile(ast.fix_missing_locations(mod), str(p), 'exec'), ns2)
    sync_weights = ns2['_sync_trainer_weights']
    sync_dropout = ns2['_sync_feature_dropout']
    assert callable(sync_weights)
    assert callable(sync_dropout)
    tc = NS(gumbel_topk=1, sf_wdl_frac=0.0, sf_wdl_frac_floor=0.0, sf_wdl_floor_at_regret=0.0, sf_pid_wdl_regret_max=0.0, feature_dropout_p=0.6, fdp_king_safety=-1.0, fdp_pins=0.0, fdp_pawns=-1.0, fdp_mobility=0.8, fdp_outposts=-2.0)
    layout = [(o, n) for o, n, _ in trainer._feature_group_dropout]
    for global_p, overrides, expected in [(0.6, (-1.0, 0.0, -1.0, 0.8, -2.0), [0.6, 0.0, 0.6, 0.8, 0.6, 0.6]), (0.0, (-1.0, 0.0, -1.0, 0.8, -2.0), [0.0, 0.0, 0.0, 0.8, 0.0, 0.0]), (0.3, (-1.0, 0.0, None, 0.8, -2.0), [0.3, 0.0, 0.3, 0.8, 0.3, 0.3])]:
        tc.feature_dropout_p = global_p
        for key, value in zip(('fdp_king_safety', 'fdp_pins', 'fdp_pawns', 'fdp_mobility', 'fdp_outposts'), overrides):
            setattr(tc, key, value)
        sync_weights(trainer, {key: getattr(tc, key) for key in ('feature_dropout_p', 'fdp_king_safety', 'fdp_pins', 'fdp_pawns', 'fdp_mobility', 'fdp_outposts')}, tc, NS(wdl_regret=0.0))
        assert [p for _, _, p in trainer._feature_group_dropout] == expected
        assert [(o, n) for o, n, _ in trainer._feature_group_dropout] == layout
        assert trainer.feature_dropout_p == global_p
        constructed = NS()
        ctor_ns: dict[str, object] = {'self': constructed, 'feature_dropout_p': global_p, **{key: getattr(tc, key) for key in ('fdp_king_safety', 'fdp_pins', 'fdp_pawns', 'fdp_mobility', 'fdp_outposts')}}
        exec(compile(ast.Module(body=body, type_ignores=[]), 'actual-constructor', 'exec'), ctor_ns)
        assert trainer._feature_group_dropout == constructed._feature_group_dropout
        consumed = [(off, length, p) for off, length, p in trainer._feature_group_dropout if p > 0.0]
        assert consumed == [(off, length, p) for (off, length), p in zip(layout, expected) if p > 0.0]
        before = list(trainer._feature_group_dropout)
        sync_weights(trainer, {key: getattr(tc, key) for key in ('feature_dropout_p', 'fdp_king_safety', 'fdp_pins', 'fdp_pawns', 'fdp_mobility', 'fdp_outposts')}, tc, NS(wdl_regret=0.0))
        assert trainer._feature_group_dropout == before
    reload_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_reload_yaml_into_config')
    import builtins
    safe_builtins = dict(vars(builtins))
    real_import = builtins.__import__

    def inert_import(name: str, globals: dict[str, object] | None=None, locals: dict[str, object] | None=None, fromlist: tuple[str, ...]=(), level: int=0) -> object:
        if name == 'chess_anti_engine.utils':
            return NS(flatten_run_config_defaults=lambda x: x, load_yaml_file=lambda _p: {})
        return real_import(name, globals, locals, fromlist, level)
    safe_builtins['__import__'] = inert_import
    reload_ns: dict[str, object] = {'__builtins__': safe_builtins, '_warn_on_keys_dropped_from_yaml': lambda **_kw: None, '_LAST_RELOAD_YAML_KEYS': {}, 'log': NS(warning=lambda *_a: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), reload_fn], type_ignores=[])), 'actual-reload', 'exec'), reload_ns)
    config = {'feature_dropout_p': 0.3, 'fdp_mobility': 0.8}
    reload_config = reload_ns['_reload_yaml_into_config']
    assert callable(reload_config)
    reload_config(config, 'inert.yaml', live_reload=True)
    assert config == {'feature_dropout_p': 0.3, 'fdp_mobility': 0.8}
    trainer.feature_dropout_p = 0.3
    trainer._feature_group_dropout = [(o, n, p) for (o, n), p in zip(layout, [0.3, 0.2, 0.3, 0.8, 0.3, 0.3])]
    startup = list(trainer._feature_group_dropout)
    sync_weights(trainer, {}, NS(**{**vars(tc), 'feature_dropout_p': 0.0}), NS(wdl_regret=0.0))
    assert trainer.feature_dropout_p == 0.3
    assert trainer._feature_group_dropout == startup
    sync_dropout(trainer, {'fdp_pins': 0.4})
    assert [p for _, _, p in trainer._feature_group_dropout] == [0.3, 0.4, 0.3, 0.8, 0.3, 0.3]
    sync_dropout(trainer, {'feature_dropout_p': 0.7, 'fdp_pins': 0.4, 'fdp_mobility': 0.8})
    assert [p for _, _, p in trainer._feature_group_dropout] == [0.7, 0.4, 0.7, 0.8, 0.7, 0.7]
    before = (trainer.feature_dropout_p, list(trainer._feature_group_dropout))
    tc.fdp_mobility = 'invalid'
    try:
        sync_dropout(trainer, {'fdp_mobility': tc.fdp_mobility})
    except TypeError:
        pass
    else:
        raise AssertionError('invalid override accepted')
    assert (trainer.feature_dropout_p, trainer._feature_group_dropout) == before
    print(json.dumps({'result': 'passed', 'cases': ['startup omission preserves Trainer .3', 'saved/resume metadata retained', 'per-group-only edit', 'global-only edit retains explicit overrides', 'global reload', 'negative sentinel fallback', 'zero override', 'disabled global with explicit override', 'None fallback', 'unchanged control', 'invalid input atomicity'], 'imports': 'stdlib only'}))
