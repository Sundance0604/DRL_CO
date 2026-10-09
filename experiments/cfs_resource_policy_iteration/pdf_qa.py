"""Bundled document runtime: structural QA and all-latest-render contact sheet."""
import argparse,csv,json,hashlib
from pathlib import Path
from pypdf import PdfReader
from PIL import Image,ImageDraw,ImageFont

def embedded(font):
    font=font.get_object()
    if '/FontDescriptor' in font:
        d=font['/FontDescriptor'].get_object();return any(k in d for k in ['/FontFile','/FontFile2','/FontFile3'])
    return bool(font.get('/DescendantFonts')) and all(embedded(f) for f in font['/DescendantFonts'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--visual-reviewed',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'figures/plotting_config.json').read_text(encoding='utf-8'))
    names=sorted(cfg['figures'],key=lambda n:cfg['figures'][n]['atlas_page']);assert len(names)==16;records=[]
    assert [cfg['figures'][n]['atlas_page'] for n in names]==list(range(1,17))
    for name in names:
        p=PdfReader(root/f'figures/{name}.pdf');assert len(p.pages)==1;page=p.pages[0]
        fonts=page['/Resources']['/Font'].get_object();assert fonts and all(embedded(f) for f in fonts.values())
        assert len(page.images)==0 and len(page.extract_text().strip())>30
        assert (root/f'figures/{name}.svg').exists() and (root/f'figures/plot_config/{name}.json').exists()
        with Image.open(root/f'figures/{name}.png') as png:
            size=png.size;dpi=png.info.get('dpi',(0,0));assert min(dpi)>590 and size[0]>1700 and size[1]>1000
        with (root/cfg['figures'][name]['source']).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        render=root/f'qa/pdf_render/{name}.png';assert rows and render.exists()
        assert render.stat().st_mtime+2>=(root/f'figures/{name}.pdf').stat().st_mtime,'Stale figure render'
        records.append(dict(name=name,atlas_page=cfg['figures'][name]['atlas_page'],all_fonts_embedded=True,raster_images=0,
            PNG_size=size,PNG_dpi=dpi,source_rows=len(rows),pdf_sha256=hashlib.sha256((root/f'figures/{name}.pdf').read_bytes()).hexdigest(),
            latest_render_sha256=hashlib.sha256(render.read_bytes()).hexdigest()))
    atlas=PdfReader(root/'figures/figure_atlas.pdf');assert len(atlas.pages)==16
    assert all(len(p.images)==0 and all(embedded(f) for f in p['/Resources']['/Font'].get_object().values()) for p in atlas.pages)
    atlasrenders={int(p.stem.rsplit('-',1)[1]):p for p in (root/'qa/pdf_render').glob('atlas-*.png') if p.stem.rsplit('-',1)[1].isdigit()}
    assert set(atlasrenders)==set(range(1,17)),'Render all 16 atlas pages as atlas-<page>.png'
    assert all(p.stat().st_mtime+2>=(root/'figures/figure_atlas.pdf').stat().st_mtime for p in atlasrenders.values()),'Stale atlas render'
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    except OSError:font=ImageFont.load_default()
    for is_atlas in [False,True]:
        sheet=Image.new('RGB',(2880,2160),'white');draw=ImageDraw.Draw(sheet)
        for i,name in enumerate(names):
            path=atlasrenders[i+1] if is_atlas else root/f'qa/pdf_render/{name}.png'
            with Image.open(path) as png:
                preview=png.convert('RGB');preview.thumbnail((700,480));x,y=(i%4)*720,(i//4)*540
                sheet.paste(preview,(x+(720-preview.width)//2,y+45+(480-preview.height)//2));draw.text((x+6,y+8),f'{i+1:02d} {name}',fill='black',font=font)
        sheet.save(root/('qa/pdf_render/atlas_contact_sheet.png' if is_atlas else 'qa/pdf_render/contact_sheet.png'))
    result=dict(status='PASSED' if args.visual_reviewed else 'STRUCTURE_PASSED_PENDING_VISUAL_REVIEW',
        atlas_pages=16,individual_figures=16,all_fonts_embedded=True,raster_images=0,
        latest_all_figure_renders_visually_reviewed=args.visual_reviewed,figures=records)
    result['latest_atlas_pages_visually_reviewed']=args.visual_reviewed
    result['atlas_latest_render_sha256']={str(i):hashlib.sha256(p.read_bytes()).hexdigest() for i,p in atlasrenders.items()}
    (root/'qa/vector_pdf_audit.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='figures'})

if __name__=='__main__':main()
