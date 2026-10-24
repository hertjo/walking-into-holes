"""Check rendered text extents before saving scientific plates."""
import json
from matplotlib.text import Text


def check_text(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    suppressed={id(t) for ax in fig.axes if not ax.axison
                for axis in (ax.xaxis,ax.yaxis) for t in axis.findobj(Text)}
    # Locators also create ticks beyond the view limits. Axis.draw omits
    # those ticks; inspecting every Text object would falsely flag them.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            tolerance = 1e-10 * max(1., abs(lo), abs(hi))
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not lo-tolerance <= tick.get_loc() <= hi+tolerance:
                    suppressed.update([id(tick.label1), id(tick.label2)])
    labels=[]
    for obj in fig.findobj(Text):
        if id(obj) in suppressed: continue
        if not obj.get_visible() or not obj.get_text().strip(): continue
        # Annotation.get_window_extent includes its leader arrow; test the text
        # itself, while arrow placement is inspected visually.
        box=Text.get_window_extent(obj,renderer)
        if box.width<.1 or box.height<.1: continue
        labels.append((obj.get_text(), box))
    clashes=[]
    for i,(text,box) in enumerate(labels):
        for other,bb in labels[i+1:]:
            dx=min(box.x1,bb.x1)-max(box.x0,bb.x0)
            dy=min(box.y1,bb.y1)-max(box.y0,bb.y0)
            if dx>1.0 and dy>1.0:
                clashes.append({'text':text,'other':other,'overlap_pixels':[dx,dy]})
    bounds=fig.bbox
    outside=[t for t,b in labels if b.x0<bounds.x0-.5 or b.y0<bounds.y0-.5 or b.x1>bounds.x1+.5 or b.y1>bounds.y1+.5]
    report={'text_count':len(labels),'overlaps':clashes,'outside_canvas':outside,
            'scope':'Rendered Matplotlib text bounding boxes, excluding annotation leader arrows; arrows and Penrose schematics checked visually.'}
    path.write_text(json.dumps(report,indent=2)+'\n')
    if clashes or outside: raise RuntimeError(f'Text layout check failed: {path}')
