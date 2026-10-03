"""Matplotlib figures shared by the two stages; importing has no GUI effects."""
import os
from pathlib import Path
_cache = Path(__file__).resolve().parent / '.runtime' / 'matplotlib'
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_cache))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_scene(containers, tags, scene, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8), layout='constrained')
        ax = fig.add_subplot(111, projection='3d')
    faces = []
    for c in containers:
        x, y, z, w, length, h = (c[k] for k in ('x', 'y', 'z', 'width', 'length', 'height'))
        v = np.array([[x,y,z], [x+w,y,z], [x+w,y+length,z], [x,y+length,z],
                      [x,y,z+h], [x+w,y,z+h], [x+w,y+length,z+h], [x,y+length,z+h]])
        for ids in ([0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]):
            faces.append(v[ids])
    if faces:
        ax.add_collection3d(Poly3DCollection(faces, facecolors='#717ba7', edgecolors='#454b65', linewidths=.15, alpha=.18))
    ax.scatter(tags.x, tags.y, tags.z, c='#b82c38', s=3, label='RFID tags')
    ax.set(xlim=scene.x_range, ylim=scene.y_range, zlim=scene.z_range,
           xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=scene.name)
    ax.set_box_aspect([np.ptp(scene.x_range), np.ptp(scene.y_range), max(30, np.ptp(scene.z_range))])
    ax.view_init(25, 35)
    return ax.figure, ax


def visualize_stage1(s):
    fig, ax = draw_scene(s.Containers, s.Tags, s.scene)
    hp = s.hoverPoints
    label = 'Scan tasks' if s.get('flyByRFID', False) else 'Hover points'
    ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], marker='*', c='#d89b18', s=50, label=label)
    for segment in s.get('scanSegments', []):
        ax.plot(*np.vstack((segment[:3], segment[3:6])).T, color='#d89b18', linewidth=2)
    ax.set_title(f'{label}: K={s.Khover}, coverage={s.coverageRate:.1%}')
    ax.legend(loc='upper right')
    fig.set_label('stage1_hover_points')
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(s.history.K_values, np.array(s.history.coverage) * 100, 'o-')
    axes[0].set(xlabel=f'Number of {label.lower()}', ylabel='Expected coverage (%)', title='Minimum set cover')
    axes[0].grid(alpha=.3)
    axes[1].hist(s.cov_prob, bins=np.linspace(0, 1, 21), color='#45759d')
    axes[1].set(xlabel='Tag read probability', ylabel='Number of tags', title='Joint coverage probability')
    fig2.set_label('stage1_coverage')
    return [fig, fig2]


def visualize_stage2(s):
    first, sol = s.stage1, s.solution
    fig, ax = draw_scene(first.Containers, first.Tags, first.scene)
    colors = plt.get_cmap('tab10')(np.arange(s.numUAVs) % 10)
    s.uavColors = colors[:, :3]
    for u, path in enumerate(sol.paths):
        if len(path):
            ax.plot(*path[:, :3].T, color=colors[u], linewidth=1.5, label=f'UAV {u + 1} ({sol.uavCosts[u]:.1f}s)')
    ax.scatter(*s.depotPos, marker='s', c='red', s=45)
    ax.set_title(f'UAV paths — makespan {s.makespan:.1f} s')
    ax.legend(fontsize=8)
    fig.set_label('stage2_paths_3d')
    fig2, ax2 = plt.subplots(figsize=(7, 9), layout='constrained')
    for c in first.Containers:
        if c.z == 0:
            ax2.add_patch(plt.Rectangle((c.x, c.y), c.width, c.length, facecolor='#c4c8d7', edgecolor='#777d95', linewidth=.3))
    for u, path in enumerate(sol.paths):
        if len(path):
            ax2.plot(path[:, 0], path[:, 1], color=colors[u], label=f'UAV {u + 1}')
    ax2.scatter(*s.depotPos[:2], marker='s', c='red', s=45)
    ax2.set(xlim=first.scene.x_range, ylim=first.scene.y_range, xlabel='X (m)', ylabel='Y (m)', title='Top view')
    ax2.set_aspect('equal')
    ax2.legend()
    fig2.set_label('stage2_paths_top')
    fig3, ax3 = plt.subplots(figsize=(8, 4), layout='constrained')
    ax3.bar(np.arange(s.numUAVs) + 1, sol.uavCosts, color=colors)
    ax3.axhline(s.makespan, color='red', linestyle='--', label='Makespan')
    ax3.axhline(sol.uavCosts.mean(), color='navy', linestyle=':', label='Mean')
    ax3.set(xlabel='UAV', ylabel='Completion time (s)', title=f'Load distribution; SOC={sol.soc:.1f} s', xticks=np.arange(s.numUAVs) + 1)
    ax3.legend()
    fig3.set_label('stage2_load')
    return [fig, fig2, fig3]


def visualize_game(containers, tags, scene, candidates, hover_points, hover_idx, P, history, coverage_rate, *args):
    from .common import Struct
    from .game_hover_solver import coverage_probability
    return visualize_stage1(Struct(Containers=containers, Tags=tags, scene=scene,
        hoverPoints=hover_points, Khover=len(hover_idx), history=history,
        coverageRate=coverage_rate, cov_prob=coverage_probability(P, hover_idx)))
