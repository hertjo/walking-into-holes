"""Faint, depth-tested display shell of the removed housing bars.

This restores context above the display section only. Saved PDE fields,
physical channel geometries, and experimental measurements are not modified.
"""
import json
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, maximum_filter
from render_professional import a, ASSETS, METHODS, CUT, centers, channel, geometry, scene_config, mi


def main():
    size = 1500
    base = trimesh.load(a.b.MESH, process=False)
    window = trimesh.creation.cylinder(radius=.86, height=2., sections=192)
    window.apply_translation([0, 0, CUT+1.])
    records = []
    for method in METHODS:
        physical = trimesh.boolean.difference([base, *[channel(c) for c in centers(method)]], engine='manifold')
        removed = trimesh.boolean.intersection([physical, window], engine='manifold')
        q = removed.triangles_center
        # Exclude synthetic faces introduced by the display window itself.
        artificial = (np.abs(q[:, 2]-CUT)<1e-5) | (np.abs(np.linalg.norm(q[:, :2], axis=1)-.86)<.0002)
        shell = removed.submesh([np.flatnonzero(~artificial)], append=True, repair=False)
        shell_path = ASSETS/f'{method}-removed-bars.ply'
        shell.export(shell_path)
        cfg, _, _ = scene_config(geometry(method), size, 1, False)
        cfg.pop('key'); cfg.pop('environment'); cfg.pop('floor')
        scene = mi.load_dict(cfg)
        ghost = mi.load_dict({'type':'scene', 'shell':{'type':'ply', 'filename':str(shell_path)}})
        yy, xx = np.mgrid[:size, :size]
        sample = mi.Point2f(np.array([(xx.ravel()+.5)/size, (yy.ravel()+.5)/size], dtype=np.float32))
        ray, _ = scene.sensors()[0].sample_ray(0, 0, sample, mi.Point2f(.5))
        front = scene.ray_intersect(ray); hit = ghost.ray_intersect(ray)
        visible = (np.asarray(hit.is_valid()) & (np.asarray(hit.t) < np.asarray(front.t)-1e-4)).reshape(size, size)
        normal = np.nan_to_num(np.asarray(hit.n).T).reshape(size, size, 3)
        depth = np.asarray(hit.t).reshape(size, size)
        edge = visible & ~binary_erosion(visible, iterations=1)
        for axis in [0, 1]:
            adjacent = np.roll(visible, 1, axis=axis) & visible
            normal_jump = np.sum(normal*np.roll(normal, 1, axis=axis), axis=2) < .8
            with np.errstate(invalid='ignore'):
                depth_jump = np.abs(depth-np.roll(depth, 1, axis=axis)) > .045
            edge |= adjacent & (normal_jump | depth_jump)
        edge = maximum_filter(edge, size=2) & visible
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[:, :, :3] = [178, 187, 199]
        rgba[:, :, 3] = np.where(edge, 62, np.where(visible, 12, 0))
        overlay = Image.fromarray(rgba)
        overlay.save(ASSETS/f'{method}-ghost-overlay.png')
        original = Image.open(ASSETS/f'{method}.png').convert('RGBA')
        Image.alpha_composite(original, overlay).convert('RGB').save(ASSETS/f'{method}-with-ghost.png')
        records.append({'method':method, 'triangles':len(shell.faces), 'visible_pixels':int(visible.sum()),
                        'edge_opacity':62/255, 'surface_opacity':12/255})
        print('ghost shell', method, flush=True)
    (ASSETS/'ghost-manifest.json').write_text(json.dumps({'display_only':True,
        'geometry':'Physical housing minus saved channels, intersected with the display window; synthetic window faces excluded.',
        'visibility':'First shell intersection must be in front of the surviving opaque geometry.',
        'temperature_fields_unchanged':True, 'detail_views':'Unobstructed; no ghost shell.', 'records':records}, indent=2)+'\n')


if __name__=='__main__': main()
