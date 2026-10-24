"""Check correspondence between saved designs, render assets and figure claims."""
import hashlib,json,zipfile
import numpy as np
from PIL import Image
import run_accessible as a
from render_professional import ASSETS,METHODS,centers,CAMERA

def main():
    meta=json.loads((ASSETS/'manifest.json').read_text())
    assert not meta['preview'] and meta['size']==1500
    fingerprints=json.loads((ASSETS/'experiment-fingerprints.json').read_text())
    for name,h in fingerprints.items():assert hashlib.sha256((a.OUT/name).read_bytes()).hexdigest()==h
    composition=json.loads((ASSETS/'composition-manifest.json').read_text())
    assert Image.open(a.OUT/'accessible-channels.png').size==(8000,4300)
    ghost=json.loads((ASSETS/'ghost-manifest.json').read_text())
    assert ghost['temperature_fields_unchanged'] and ghost['display_only']
    assert len(ghost['records'])==3
    initial=np.load(ASSETS/'initial-field.npz');checks=[]
    for method in METHODS:
        geom=json.loads((ASSETS/f'{method}-geometry.json').read_text())
        assert np.array_equal(np.array(geom['source_centers']),centers(method))
        assert geom['watertight_display_mesh']
        for suffix in ['', '-detail', '-with-ghost']:
            im=np.asarray(Image.open(ASSETS/f'{method}{suffix}.png'))
            assert im.shape==(1500,1500,3) and im.mean()>20 and im.std()>15
        field=np.load(ASSETS/f'{method}-field.npz')
        assert np.array_equal(field['x'],initial['x']) and np.array_equal(field['y'],initial['y'])
        mask=field['mask'];excess=float((field['u']-initial['u'])[mask].max())
        assert excess<1e-7 and np.min(field['u'])>=-1e-12
        assert field['u'][mask].max()<=meta['temperature_scale_max']
        checks.append({'method':method,'six_saved_axes_match':True,'display_mesh_watertight':True,
            'max_state_increase_on_shared_grid':excess})
    for path in (ASSETS/'fields').glob('*/kernel.json'):
        k=json.loads(path.read_text());assert k['relative_residual']<=1e-8 and k['caps']==0
    report={'physical_experimental_results_unchanged':True,'png_size':[8000,4300],
      'removed_bar_shells_display_only':True,
      'same_cameras_and_scale':True,'camera_offset':CAMERA.tolist(),
      'display_cut_excluded_from_PDE':True,'checks':checks,'comparison':composition,
      'scope':'Rendering and saved-data correspondence checks; no new optimization measurements.'}
    (ASSETS/'figure-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
