"""Apply deterministic SVG material shading to Penrose-produced geometry.
No geometry, labels, samples, or equations are moved by this pass.
"""
from pathlib import Path
from lxml import etree
import json
R=Path(__file__).resolve().parents[1];NS='http://www.w3.org/2000/svg'
defs='''<defs xmlns="http://www.w3.org/2000/svg">
<linearGradient id="domain" x1="20%" y1="0%" x2="64%" y2="100%"><stop offset="0" stop-color="#fffefe"/><stop offset=".28" stop-color="#f3eef7"/><stop offset=".70" stop-color="#e5ddec"/><stop offset="1" stop-color="#b9a9ca"/></linearGradient>
<radialGradient id="sphere" cx="31%" cy="24%" r="76%"><stop offset="0" stop-color="#ffffff" stop-opacity=".30"/><stop offset=".44" stop-color="#c5e1eb" stop-opacity=".12"/><stop offset=".82" stop-color="#79a8bc" stop-opacity=".18"/><stop offset="1" stop-color="#456b84" stop-opacity=".40"/></radialGradient>
<radialGradient id="cavity" cx="70%" cy="81%" r="100%"><stop offset="0" stop-color="#c6b1d5"/><stop offset=".46" stop-color="#826198"/><stop offset="1" stop-color="#30253d"/></radialGradient>
<filter id="shadow" x="-20%" y="-25%" width="140%" height="160%"><feGaussianBlur stdDeviation="5"/></filter>
</defs>'''
for stem,roles in json.loads((R/'video/assets/penrose-depth-paint.json').read_text()).items():
 p=R/'figures'/f'{stem}.svg';tree=etree.parse(str(p));root=tree.getroot();shapes=[c for c in root if etree.QName(c).localname not in ['penrose','defs']]
 root.insert(0,etree.fromstring(defs.encode()))
 for name,role in roles.items():
  target=shapes[int(name[1:])-1]
  if role=='shadow':target.set('fill','#77648b');target.set('fill-opacity','.15');target.set('filter','url(#shadow)');target.set('stroke','none')
  else:target.set('fill',f'url(#{role})');target.set('fill-opacity','1')
 p.write_bytes(etree.tostring(tree))
 print('Shaded',stem)
