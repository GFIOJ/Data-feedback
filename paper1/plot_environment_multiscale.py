"""Fixed-count 500/1000/1500/2000/2500-container yard presets."""
from .common import Struct, options, round_half_away
from .plot_environment import ENV_DEFAULTS, generate_scene


def plot_environment_multiscale(*args, **kwargs):
    opts = options(dict(ENV_DEFAULTS, SceneIndex=3, TagDistributionMode='all'), args, kwargs)
    idx = min(5, max(1, int(round_half_away(opts.SceneIndex)))) - 1
    nx, ny, tag = [(3, 4, 'S'), (4, 6, 'M'), (6, 6, 'ML'), (6, 8, 'L'), (6, 10, 'XL')][idx]
    target = (idx + 1) * 500
    opts.SceneName = f'port_scale_{target}'
    containers, tags, scene, layout = generate_scene(opts, nx, ny, target)
    scene.update(scenario_tag=tag, scale_level=tag)
    preset = Struct(name=opts.SceneName, n_blocks_x=nx, n_blocks_y=ny, scale_tag=tag,
                    target_containers=target)
    if opts.Visualize:
        from .visualization import draw_scene
        draw_scene(containers, tags, scene)
    return containers, tags, scene, layout, preset
