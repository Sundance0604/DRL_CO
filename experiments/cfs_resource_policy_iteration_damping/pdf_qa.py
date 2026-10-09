"""Bundled-runtime PDF inspection and rendering; manual visual pass is explicit."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from pypdf import PdfReader
from PIL import Image,ImageDraw,ImageChops

def embedded(font):
    f=font.get_object()
    descendants=f.get('/DescendantFonts')
    if descendants:return all(embedded(x) for x in descendants)
    descriptor=f.get('/FontDescriptor')
    if descriptor:
        d=descriptor.get_object()
        return any(d.get(k) is not None for k in ['/FontFile','/FontFile2','/FontFile3'])
    return False

def image_count(resources):
    resources=resources.get_object();objects=resources.get('/XObject')
    if not objects:return 0
    total=0
    for ref in objects.get_object().values():
        obj=ref.get_object()
        if obj.get('/Subtype')=='/Image':total+=1
        elif obj.get('/Subtype')=='/Form' and obj.get('/Resources'):total+=image_count(obj['/Resources'])
    return total

def inspect(directory,reviewed):
    out=directory/'qa/pdf_render';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((directory/'figures/figure_manifest.json').read_text(encoding='utf-8'))
    rows=[];renders=[]
    for name in manifest['names']+[Path(manifest['atlas']).stem]:
        path=directory/'figures'/(name+'.pdf');reader=PdfReader(path)
        for i,page in enumerate(reader.pages):
            assert page.extract_text().strip()
            resources=page['/Resources'].get_object()
            fonts=resources.get('/Font',{}).get_object() if resources.get('/Font') else {}
            assert fonts and all(embedded(f) for f in fonts.values())
            assert image_count(resources)==0,'Final PDF must remain vector, not rasterized chart'
            prefix=out/f'{name}-page{i+1}'
            subprocess.run(['pdftoppm','-f',str(i+1),'-l',str(i+1),'-singlefile','-r','150','-png',str(path),str(prefix)],check=True,capture_output=True)
            rendered=prefix.with_suffix('.png');assert rendered.exists()
            rows.append(dict(file=path.name,page=i+1,embedded_fonts=True,raster_images=0,
                pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),render=rendered.name))
            renders.append(rendered)
        if name in manifest['names']:
            png=directory/'figures'/(name+'.png')
            with Image.open(png) as im:assert min(im.info.get('dpi',(0,0)))>=590
    assert len(rows)==2*len(manifest['names'])
    pairchecks=[]
    for i,name in enumerate(manifest['names']):
        a=out/f'{name}-page1.png';b=out/f'{Path(manifest["atlas"]).stem}-page{i+1}.png'
        with Image.open(a) as x,Image.open(b) as y:
            same=x.size==y.size and ImageChops.difference(x.convert('RGB'),y.convert('RGB')).getbbox() is None
            pairchecks.append(dict(name=name,render_pixels_identical=same))
    # All pages are retained separately; contact sheets are navigation aids only.
    sheets=[]
    for start in range(0,len(renders),6):
        batch=renders[start:start+6];canvas=Image.new('RGB',(1800,1950),'white');draw=ImageDraw.Draw(canvas)
        for j,p in enumerate(batch):
            with Image.open(p) as im:
                thumb=im.convert('RGB');thumb.thumbnail((860,590))
                x=(j%2)*900+(900-thumb.width)//2;y=(j//2)*650+35
                canvas.paste(thumb,(x,y));draw.text(((j%2)*900+10,(j//2)*650+10),p.stem,fill='black')
        cp=out/f'contact-{start//6+1}.png';canvas.save(cp);sheets.append(str(cp.resolve()))
    result=dict(status='PASSED' if reviewed else 'TECHNICAL_PASS_VISUAL_PENDING',
        pdf_pages=len(rows),all_embedded_fonts=True,all_vector=True,all_PNG_600dpi=True,
        manual_visual_review=reviewed,rows=rows,atlas_individual_pair_checks=pairchecks,contact_sheets=sheets,
        visual_scope='all individual and all atlas pages; identical pairs may share a visual review')
    (directory/'qa/pdf_visual_audit.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--visual-reviewed',action='store_true');args=p.parse_args()
    inspect(args.directory,args.visual_reviewed)
