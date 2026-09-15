"""Verify source availability, figure page sizes and manuscript image links."""
from pathlib import Path
import hashlib, json, re, subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'publication/figures/methodology_revision'
TMP = ROOT/'tmp/pdfs/methodology_revision'
TMP.mkdir(parents=True, exist_ok=True)
inventory = json.loads((ROOT/'publication/references/reading_notes/inventory.json').read_text())
sources=[]
for item in inventory:
    source=ROOT/'publication/references'/item['pdf'].replace('\\','/').split('/')[-1]
    if not source.exists():
        sources.append({'file':str(source),'status':'missing'});continue
    reader=PdfReader(source)
    counts=[len(page.extract_text() or '') for page in reader.pages]
    sources.append({'file':str(source),'pages':len(reader.pages),
                    'hash_matches_reading_record':hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256'],
                    'low_text_pages':[i+1 for i,n in enumerate(counts) if n<100]})
figures=[]
for path in sorted(OUT.glob('*.pdf')):
    reader=PdfReader(path)
    page=reader.pages[0]
    width=float(page.mediabox.width)/72*25.4
    assert len(reader.pages)==1 and abs(width-88.9)<.01
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1400','-png',str(path),str(TMP/path.stem)],check=True)
    figures.append({'file':path.name,'width_mm':width,'height_mm':float(page.mediabox.height)/72*25.4})
md=(ROOT/'publication/manuscript/04_methods.md').read_text(encoding='utf-8')
links=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md)
assert all(Path(link).exists() for link in links),links
result={'reference_pdf_accessibility':sources,'figures':figures,'manuscript_image_links_valid':True,
        'reading_scope':'Availability/hash audit of all 36 library PDFs against existing full-library reading record; this audit is not a new semantic reading of every paper.'}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'library_pdfs':len(sources),'unchanged_sources':sum(s.get('hash_matches_reading_record',False) for s in sources),'low_text_pages':[(s['file'],s.get('low_text_pages')) for s in sources if s.get('low_text_pages')],'figures':figures,'image_links':len(links)},indent=2))
