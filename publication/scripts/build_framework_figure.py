"""Generic implementation diagram. Run with conda run -n vtsim python."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'publication/figures/framework'
OUT.mkdir(parents=True,exist_ok=True)
from process_grid import process_grid
fig=process_grid([
    ('Prepare inputs','3D mesh\nUnits / scale\nSensor preset\nSampling settings'),
    ('Set contact','XYZ rotation\nXY offset\nIndentation\nFixed or sampled'),
    ('Contact\ngeometry','Raw depth / mask\nDepth smoothing\nSurface normals'),
    ('Optical\nrendering','Pad colour; gradients\nLighting\nColour controls\nClean RGB'),
    ('Image\nprocessing','Texture; vignette\nBlur; noise\nContrast; gamma\nFinal RGB'),
    ('Export samples','Visuo-tactile images\nDepth / masks\nMesh; metadata'),
], 'Sensor configuration: direct or reference-calibrated.\nRepeat steps 2–6 for batch generation.')
for ext in ('png','pdf','svg'):fig.savefig(OUT/f'fig_framework.{ext}',dpi=600,facecolor='white')
plt.close(fig)
sources=['mesh2tact/geometric.py','mesh2tact/gather.py','mesh2tact/calibration.py','mesh2tact/render/gel.py','mesh2tact/render/optical.py','mesh2tact/render/effects.py']
(OUT/'provenance.json').write_text(json.dumps({'figure_size_inches':[3.5,2.65],'width_mm':88.9,'diagram':'Generic static mesh-to-visuo-tactile generation in a two-row, three-column grid; numbered flow and batch repetition.','source_code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},'scope':'Conceptual implementation diagram; no example object or photograph. Prior example arrays in this directory are legacy artifacts, not figure inputs.'},indent=2),encoding='utf-8')
print(OUT/'fig_framework.png')
