"""Port container yard and RFID tag generation."""
from __future__ import annotations
import numpy as np
from .common import Struct, options, round_half_away, positive

CONTAINER_LENGTH = 12.192
CONTAINER_WIDTH = 2.438
CONTAINER_HEIGHT = 2.896

ENV_DEFAULTS = dict(NumTags=None, TagSeed=1, TagDistributionMode='subset_uniform',
    TagDensityPerBox=None, TagClusterLevel=None, AisleWidth=None, AisleWidthY=None,
    AisleWidthX=None, FixedSceneLayout=True, LayoutSeed=20260323,
    SceneWidth=None, SceneLength=None, NumBlocksX=4, NumBlocksY=6, BaysPerBlock=None, NumContainers=None,
    AggregateYard=False, GroundSlots=60000, StackTiers=5, StripsPerRegion=4, BoundaryMargin=0.,
    StackIrregularityLevel=0, StackPositionJitter=0, StackHeightJitter=0,
    StackFillJitter=0, StackSparseBayProb=0, SceneName='port_default', Visualize=True)


def generate_aggregate_scene(opts):
    """Generate stack-strip obstacles while retaining physical yard capacity metadata."""
    width = positive(opts.SceneWidth, 'SceneWidth')
    length = positive(opts.SceneLength, 'SceneLength')
    nx = positive(opts.NumBlocksX, 'NumBlocksX', integer=True)
    ny = positive(opts.NumBlocksY, 'NumBlocksY', integer=True)
    strips = positive(opts.StripsPerRegion, 'StripsPerRegion', integer=True)
    slots = positive(opts.GroundSlots, 'GroundSlots', integer=True)
    tiers = positive(opts.StackTiers, 'StackTiers', integer=True)
    margin = positive(opts.BoundaryMargin, 'BoundaryMargin', allow_zero=True)
    tag_count = positive(opts.NumTags, 'NumTags', integer=True)
    block_count = nx * ny * strips
    if slots < block_count or tag_count < block_count:
        raise ValueError('GroundSlots and NumTags must be at least the aggregate stack count')
    if 2 * margin >= min(width, length):
        raise ValueError('BoundaryMargin leaves no usable yard area')
    usable_width, usable_length = width - 2 * margin, length - 2 * margin
    cell_x, cell_y = usable_width / nx, usable_length / ny
    strip_x = cell_x * .75
    slot_counts = np.full(block_count, slots // block_count, int)
    slot_counts[:slots % block_count] += 1
    strip_y_max = slot_counts.max() * CONTAINER_LENGTH * CONTAINER_WIDTH / strip_x
    gap_y = (cell_y - strips * strip_y_max) / (strips + 1)
    if gap_y < 8:
        raise ValueError('Regions are too dense to preserve RFID scan aisles')
    containers, faces = [], []
    index = 0
    for by in range(ny):
        for bx in range(nx):
            for row in range(strips):
                strip_y = slot_counts[index] * CONTAINER_LENGTH * CONTAINER_WIDTH / strip_x
                x = margin + bx * cell_x + (cell_x - strip_x) / 2
                y = margin + by * cell_y + gap_y * (row + 1) + strip_y_max * row + (strip_y_max - strip_y) / 2
                sign = -1 if row % 2 == 0 else 1
                containers.append(Struct(x=x, y=y, z=0., width=strip_x, length=strip_y,
                    height=tiers * CONTAINER_HEIGHT, aggregate=True,
                    ground_slots=int(slot_counts[index]), stack_tiers=tiers))
                faces.append((index, sign))
                index += 1
    rng = np.random.RandomState(int(opts.TagSeed))
    tags_per_face = np.full(block_count, tag_count // block_count, int)
    tags_per_face[rng.permutation(block_count)[:tag_count % block_count]] += 1
    positions, normals, owners = [], [], []
    for block_id, sign in faces:
        block, count = containers[block_id], tags_per_face[block_id]
        order = rng.permutation(count)
        for k in range(count):
            x = block.x + (order[k] + .5) / count * block.width
            y = block.y if sign < 0 else block.y + block.length
            z = (k % tiers + .5) * CONTAINER_HEIGHT
            positions.append([x, y, z]); normals.append([0., float(sign), 0.]); owners.append(block_id)
    xyz = np.asarray(positions)
    tags = Struct(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], normal=np.asarray(normals),
                  owner_idx=np.asarray(owners), id=np.arange(tag_count))
    utilization = slots * CONTAINER_LENGTH * CONTAINER_WIDTH / (width * length)
    scene = Struct(x_range=np.array([0., width]), y_range=np.array([0., length]),
        z_range=np.array([0., max(20., tiers * CONTAINER_HEIGHT + 5)]), name=opts.SceneName,
        scenario_tag=opts.SceneName, aisle_width_y=gap_y, aisle_width_x=cell_x - strip_x,
        aggregate_yard=True, num_regions=nx * ny, aggregate_stack_count=block_count,
        boundary_margin=margin, usable_yard_area=usable_width * usable_length,
        ground_slots=slots, stack_tiers=tiers, physical_container_count=slots * tiers,
        ground_utilization=float(utilization),
        physical_container_dimensions=Struct(length=CONTAINER_LENGTH,width=CONTAINER_WIDTH,height=CONTAINER_HEIGHT))
    for key in ('irregularity_level', 'position_jitter', 'height_jitter', 'fill_jitter', 'sparse_bay_prob'):
        scene['stack_' + key] = opts['Stack' + ''.join(x.title() for x in key.split('_'))]
    layout = Struct(n_blocks_x=nx, n_blocks_y=ny, rows_per_block=strips,
                    bays_per_block=None, max_tier=tiers, aggregate=True)
    return containers, tags, scene, layout


def select_tags(full, opts):
    n = len(full.x)
    count = opts.NumTags
    if count is None:
        return full
    count = positive(count, 'NumTags', integer=True)
    if count >= n:
        return full
    rng = np.random.RandomState(int(opts.TagSeed))
    mode = opts.TagDistributionMode.lower().strip()
    level = opts.get('TagClusterLevel')
    if level is None:
        level = dict(cluster_weak=.25, cluster_mid=.5, cluster_strong=.75, cluster_high=1).get(mode, 0)
    if not 0 <= level <= 1:
        raise ValueError('TagClusterLevel must be in [0, 1]')
    if level > 0:
        pos = np.column_stack((full.x, full.y, full.z))
        centers = pos[rng.choice(n, min(3, n), replace=False)]
        d2 = ((pos[:, None] - centers) ** 2).sum(2).min(1)
        sigma = max(1, max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), 1) * (.35 - .25 * level))
        weights = np.exp(-d2 / (2 * sigma**2)) + 1e-9
        clustered = int(round_half_away(count * level))
        ids = rng.choice(n, clustered, replace=False, p=weights / weights.sum())
        rest = np.setdiff1d(np.arange(n), ids)
        ids = np.r_[ids, rng.choice(rest, count - clustered, replace=False)]
        rng.shuffle(ids)
    else:
        # The source generates exactly one tag per box; density_per_box therefore
        # reduces to a uniform subset as well.
        ids = rng.choice(n, count, replace=False)
    return Struct({k: v[ids] for k, v in full.items()})


def generate_scene(opts, nx=None, ny=None, target=None):
    nx = positive(opts.NumBlocksX if nx is None else nx, 'NumBlocksX', integer=True)
    ny = positive(opts.NumBlocksY if ny is None else ny, 'NumBlocksY', integer=True)
    target = opts.NumContainers if target is None else target
    if target is not None:
        target = positive(target, 'NumContainers', integer=True)
    bays = 6 if opts.BaysPerBlock is None else positive(opts.BaysPerBlock, 'BaysPerBlock', integer=True)
    if target is not None and opts.BaysPerBlock is None:
        bays = max(bays, int(np.ceil(target / (nx * ny * 2 * 5))))
    for key in ('StackIrregularityLevel', 'StackSparseBayProb'):
        if not 0 <= opts[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    for key in ('StackPositionJitter', 'StackHeightJitter', 'StackFillJitter'):
        positive(opts[key], key, allow_zero=True)
    ay = opts.AisleWidthY if opts.AisleWidthY is not None else opts.AisleWidth
    ay = 5. if ay is None else positive(ay, 'AisleWidthY')
    ax = 15. if opts.AisleWidthX is None else positive(opts.AisleWidthX, 'AisleWidthX')
    layout = Struct(n_blocks_x=nx, n_blocks_y=ny, rows_per_block=2, bays_per_block=bays,
                    max_tier=5, aisle_width_y=ay, aisle_width_x=ax, top_fill_prob=.8,
                    margin_x=22, margin_y=28)
    # Preserve the source arithmetic order: replacing 6*(2.4+.4) by 16.8
    # changes some coordinates by 1e-14, enough to change floor(x/GridRes).
    block_length = bays * (CONTAINER_WIDTH + .4)
    block_width = 2 * (CONTAINER_LENGTH + .5)
    total_x = nx * block_length + (nx - 1) * ax
    total_y = ny * block_width + (ny - 1) * ay
    xmax = (150. if target is None else total_x + 44) if opts.SceneWidth is None else positive(opts.SceneWidth, 'SceneWidth')
    ymax = (250. if target is None else total_y + 56) if opts.SceneLength is None else positive(opts.SceneLength, 'SceneLength')
    if xmax < total_x or ymax < total_y:
        raise ValueError('Scene dimensions are too small for the requested block layout')
    scene = Struct(x_range=np.array([0., xmax]), y_range=np.array([0., ymax]),
                   z_range=np.array([0., 20.]), name=opts.SceneName, scenario_tag=opts.SceneName,
                   aisle_width_y=ay, aisle_width_x=ax)
    for key in ('irregularity_level', 'position_jitter', 'height_jitter', 'fill_jitter', 'sparse_bay_prob'):
        scene['stack_' + key] = opts['Stack' + ''.join(x.title() for x in key.split('_'))]
    rng = np.random.RandomState(int(opts.LayoutSeed) if opts.FixedSceneLayout else None)
    tiers = None
    if target is not None:
        slots = nx * ny * bays * 2
        if target > slots * layout.max_tier:
            raise ValueError(f'NumContainers cannot exceed {slots * layout.max_tier} for this block layout')
        tiers = np.full(slots, target // slots)
        tiers[rng.choice(slots, target % slots, replace=False)] += 1
        scene.update(target_containers=target, n_blocks_x=nx, n_blocks_y=ny)
    containers, positions, normals = [], [], []
    x_origins = np.linspace(22, xmax - 22 - block_length, nx) if opts.SceneWidth is not None else [(xmax - total_x) / 2 + bx * (block_length + ax) for bx in range(nx)]
    y_origins = np.linspace(28, ymax - 28 - block_width, ny) if opts.SceneLength is not None else [(ymax - total_y) / 2 + by * (block_width + ay) for by in range(ny)]
    slot = 0
    for by in range(ny):
        for bx in range(nx):
            xs, ys = x_origins[bx], y_origins[by]
            for bay in range(bays):
                for row in range(2):
                    if tiers is None:
                        additional = int(rng.rand() < .8) + int(rng.rand() < .8)
                        if opts.StackIrregularityLevel > 0:
                            if rng.rand() < opts.StackSparseBayProb:
                                additional = max(0, additional - 1)
                            if opts.StackFillJitter > 0:
                                additional = int(np.clip(additional + round_half_away(opts.StackFillJitter * rng.randn()), 0, 3))
                        nt = min(5, 2 + additional)
                    else:
                        nt = int(tiers[slot])
                    slot += 1
                    for tier in range(nt):
                        x = xs + bay * (CONTAINER_WIDTH + .5)
                        y = ys + row * (CONTAINER_LENGTH + .4)
                        z, height = tier * CONTAINER_HEIGHT, CONTAINER_HEIGHT
                        if opts.StackIrregularityLevel > 0:
                            if opts.StackPositionJitter > 0:
                                dx, dy = opts.StackPositionJitter * (2 * rng.rand(2) - 1)
                                x, y = x + dx, y + dy
                            if opts.StackHeightJitter > 0:
                                height *= np.clip(1 + opts.StackHeightJitter * rng.randn(), .75, 1.25)
                        sign = -1 if row == 0 else 1
                        pos = [x + CONTAINER_WIDTH / 2, y if sign < 0 else y + CONTAINER_LENGTH, z + height / 2]
                        normal = [0., float(sign), 0.]
                        containers.append(Struct(x=x, y=y, z=z, width=CONTAINER_WIDTH, length=CONTAINER_LENGTH,
                            height=height, tag_pos=np.array(pos), tag_normal=np.array(normal)))
                        positions.append(pos)
                        normals.append(normal)
    xyz = np.asarray(positions)
    full = Struct(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], normal=np.asarray(normals),
                   owner_idx=np.arange(len(xyz)), id=np.arange(len(xyz)))
    tags = select_tags(full, opts)
    return containers, tags, scene, layout


def plot_environment(*args, **kwargs):
    opts = options(ENV_DEFAULTS, args, kwargs)
    containers, tags, scene, _ = generate_aggregate_scene(opts) if opts.AggregateYard else generate_scene(opts)
    if opts.Visualize:
        from .visualization import draw_scene
        draw_scene(containers, tags, scene)
    return containers, tags, scene


if __name__ == '__main__':
    plot_environment()
    import matplotlib.pyplot as plt
    plt.show()
