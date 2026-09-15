"""Build the manually selected Mesh2Tact reading list from saved Crossref records."""
import json, re, html
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'references'
candidates={r['index']:r for r in json.loads((OUT/'crossref_candidates.json').read_text(encoding='utf-8'))}
# Query number, candidate index, manuscript use. Selection is explicit, not top-hit automation.
groups=[('Direct tactile simulation and generation',[
(1,0,'Closest geometric generation baseline: compare depth construction, smoothing, optical rendering and real-data evaluation.'),
(2,0,'Essential calibrated optical baseline; distinguish our explicit lighting fit from example-based optical mapping.'),
(3,0,'Essential rendering and scalable synthetic-data baseline; prevents claiming automated generation itself is new.'),
(4,0,'Modern fast optical-simulation baseline; inspect its optical mapping, shadows and transfer evaluation.'),
(5,0,'Deformation-simulation alternative; clarify what our geometric approximation omits.'),
(6,0,'GPU simulation and learning framework; relevant to throughput and system-level comparisons.'),
(7,0,'Deformed-mesh to tactile-output framework; important overlap with geometry-to-image generation.'),
(8,0,'Rendering efficiency versus fidelity; relevant to the computational-cost discussion.'),
(8,1,'Physically based optical-rendering alternative to our approximate illumination model.'),
(9,0,'Sensor optics and illumination modeling; use when discussing physical fidelity and sensor-specific appearance.'),
(5,1,'Recent dynamic-contact generator: consider marker simulation and synthetic-data validation beyond static contacts.'),
(38,0,'Recent simulator for biomorphic sensor structures; helps state the limits of a planar heightmap renderer.')]),
('Visuo-tactile sensor foundations',[
(14,0,'Core GelSight reference for geometry, force and optical tactile sensing.'),
(15,0,'Compact vision-based tactile sensor and hardware context for simulation.'),
(16,0,'Calibrated tactile imaging and compact sensing-finger design.'),
(17,0,'Shape, force and slip measurement; distinguish actual sensor capabilities from simulated outputs.'),
(18,0,'GelSight optical design and geometric/slip sensing; relevant to illumination and reference images.'),
(19,0,'Nonplanar tactile sensor morphology; useful for describing generalization limits.'),
(20,0,'Marker-based soft optical sensor family; contrasts with photometric GelSight-style generation.'),
(21,0,'Multi-directional tactile imaging; contrasts with a single planar observation.'),
(22,0,'Round sensor and cross-instance learning context; do not infer that our fit transfers across units.'),
(23,0,'Alternative intensity-to-geometry sensing principle; useful for defining sensor-specific rendering assumptions.'),
(24,0,'Comparison of tactile sensor designs; motivates controlled cross-sensor evaluation.')]),
('Synthetic-to-real learning and tactile data use',[
(10,0,'Simulation benchmark and transfer evaluation design; useful for experiments beyond visual similarity.'),
(11,0,'Tactile sim-to-real task evaluation and calibration context.'),
(12,0,'Optical tactile domain transfer; use when discussing the gap between rendered and real images.'),
(13,0,'Learning tactile perception in simulation; relevant to the value of synthetic datasets.'),
(28,0,'Downstream use of vision and touch; motivation for tactile data, not a geometric generator baseline.'),
(31,0,'Cross-modal tactile synthesis alternative; distinguish image-conditioned prediction from mesh-conditioned generation.')]),
('Reviews for introduction and related work',[
(25,0,'Most directly aligned review: tactile data generation, datasets and applications; use its taxonomy to structure related work.'),
(26,0,'Broad tactile perception context and object-property recognition motivation.')]),
('Rendering, randomization and evaluation methods',[
(32,0,'Appearance/domain randomization foundation; relevant only to the randomization actually implemented and evaluated.'),
(33,0,'Dynamics-randomization context; optional because the current generator does not model contact dynamics.'),
(34,0,'Original SSIM reference if SSIM is used for real-versus-synthetic evaluation.'),
(35,0,'Original LPIPS/perceptual-metric reference if that metric is added; not proof of tactile task fidelity.'),
(36,1,'Original 1975 Phong illumination paper, selected instead of the 1998 reprint; theoretical rendering foundation.')])]

records=[]
for group,entries in groups:
    for query,index,use in entries:
        c=candidates[query]['candidates'][index]
        title=html.unescape(c['title'][0]); title=re.sub('<[^>]*>','',title)
        authors=[(a.get('given','')+' '+a.get('family','')).strip() or a.get('name','') for a in c.get('author',[])]
        year=c.get('published',c['issued'])['date-parts'][0][0]
        key=f'Mesh2TactRef{len(records)+1:02d}'
        records.append(dict(number=len(records)+1,key=key,title=title,authors=authors,year=year,
             venue=c['container-title'][0],doi=c['DOI'],url='https://doi.org/'+c['DOI'],
             publisher=c.get('publisher'),type=c['type'],volume=c.get('volume'),issue=c.get('issue'),
             pages=c.get('page'),article_number=c.get('article-number'),group=group,use=use,
             verification='Title, DOI, venue and publication year matched to publisher-deposited Crossref metadata; not a full-text critical appraisal.',
             metadata_source='https://api.crossref.org/works/'+c['DOI'],source_query=query,candidate_index=index))
assert len(records)==36
assert len({r['doi'] for r in records})==36
assert all(r['type'] in ('journal-article','proceedings-article') for r in records)
(OUT/'Mesh2Tact_references.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')

def esc(s):
    return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
bib=[]
for r in records:
    c=candidates[r['source_query']]['candidates'][r['candidate_index']]
    names=['{'+a['name']+'}' if 'name' in a else a.get('family','')+', '+a.get('given','') for a in c['author']]
    fields={'title':'{'+esc(r['title'])+'}','author':' and '.join(esc(n) for n in names),'year':r['year'],
            'journal' if r['type']=='journal-article' else 'booktitle':esc(r['venue']),
            'doi':r['doi'],'url':r['url']}
    for src,dest in [('volume','volume'),('issue','number'),('pages','pages')]:
        if r[src]: fields[dest]=str(r[src]).replace('-', '--') if src=='pages' else r[src]
    kind='article' if r['type']=='journal-article' else 'inproceedings'
    bib.append('@'+kind+'{'+r['key']+',\n'+',\n'.join('  '+k+' = {'+str(v)+'}' for k,v in fields.items())+'\n}\n')
(OUT/'Mesh2Tact_references.bib').write_text('\n'.join(bib),encoding='utf-8')

intro='''# Mesh2Tact: annotated reference library

36 selected published papers. Checked 2026-09-14. Scope: geometric tactile rendering, synthetic dataset generation, visuo-tactile sensor foundations and evaluation.

IEEE is a publisher; Scopus is an indexing service. The list uses IEEE publications and papers from non-IEEE journals with Scopus coverage. Publisher-deposited Crossref metadata was matched for all 36 titles, venues, publication years and DOIs. No arXiv-only entries are included. This is a curated reading list, not a claim to have critically reviewed every full text. Individual Scopus EIDs were not retrieved; venue coverage is distinct from verification of a particular article's Scopus record.

Years follow the published journal issue/conference metadata, not the arXiv year or the year embedded in a DOI. For example, TACTO and Taxim are cited as RA-L 2022; AllSight is cited as RA-L 2024 despite a 2023 DOI.

## Read first

Start with refs 1-5 (closest baselines), 6-7 (recent simulator frameworks), 30 (data-generation review), and 13 (GelSight foundations). Then read 8-12 for additional simulator coverage. Retain only references that support statements in the final paper; the bibliography is a candidate library, not a requirement to cite all 36.

'''
lines=[intro]
for group,_ in groups:
    lines.append('## '+group+'\n')
    for r in records:
        if r['group']!=group: continue
        authors=', '.join(r['authors'])
        lines.append(f"### [{r['number']}] {r['title']}\n\n{authors}. **{r['venue']}**, {r['year']}.\n\n[DOI / publisher link]({r['url']})\n\n**Use in Mesh2Tact:** {r['use']}\n")
lines.append('''## Indexing and source notes

- [Sensors: publisher indexing page](https://www.mdpi.com/journal/sensors/indexing).
- [Communications Engineering: publisher journal information](https://www.nature.com/commseng/journal-information).
- [Cyborg and Bionic Systems: publisher indexing page](https://spj.science.org/page/cbsystems/abstracting-indexing).
- [Sensors and Actuators A: institutional Scopus metrics](https://library.wur.nl/WebQuery/utbrowser?issn=1873-3069).
- Other non-IEEE venues in this library: Soft Robotics, Information Fusion, Mechatronics, Communications of the ACM. Check institutional Scopus access if article-level indexing evidence is required for an assessment.
- Crossref source responses are retained in `crossref_candidates.json`; only explicitly selected records are exported to the curated JSON and BibTeX. Rejected candidates and API errors are retained for audit, not included as references.
- DOI links lead to the publication record; access to full text may require an institutional subscription.

## Suggested manuscript mapping

- Introduction: 13-23, 30-31, with a small selection of sensor examples.
- Related work / novelty: 1-12 and 24-29.
- Methods: compare the geometric/optical choices against 1-2, 8-10 and 36; use 32 only where domain randomization applies.
- Results: use 34-35 only if SSIM/LPIPS are actually computed, and benchmark against implemented baselines rather than comparing reported numbers from different hardware/datasets.
- Discussion: 5-12 and 24-29 support a discussion of missing deformation, dynamic contact, sensor transfer and downstream validation.
''')
(OUT/'Mesh2Tact_related_references.md').write_text('\n'.join(lines),encoding='utf-8')
for r in records: print(f"{r['number']:02d}. {r['title']} | {r['year']} | {r['url']}")
