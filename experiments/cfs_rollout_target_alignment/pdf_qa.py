"""Read-only figure structure inspection plus diagnostic contact sheet.

Run with the bundled document Python (pypdf/Pillow), not the numerical venv.
The visual-review flag must be supplied only after inspecting all rendered figures.
"""
import argparse
import csv
import json
from pathlib import Path
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageFont


def embedded(font):
    font = font.get_object()
    if '/FontDescriptor' in font:
        descriptor = font['/FontDescriptor'].get_object()
        return any(k in descriptor for k in ['/FontFile', '/FontFile2', '/FontFile3'])
    return bool(font.get('/DescendantFonts')) and all(embedded(f) for f in font['/DescendantFonts'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual-reviewed', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'figures/plotting_config.json').read_text(encoding='utf-8'))
    names = list(config['figures'])
    assert len(names) == 16
    records = []
    for name in names:
        pdf = root / f'figures/{name}.pdf'
        reader = PdfReader(pdf)
        assert len(reader.pages) == 1
        page = reader.pages[0]
        fonts = page['/Resources']['/Font'].get_object()
        assert len(page.images) == 0
        assert fonts and all(embedded(f) for f in fonts.values())
        text = page.extract_text()
        assert len(text.strip()) > 50
        assert (root / f'figures/{name}.svg').is_file()
        with Image.open(root / f'figures/{name}.png') as png:
            width, height = png.size
            dpi = png.info.get('dpi', (0, 0))
            assert min(dpi) > 590 and width > 1700 and height > 1000
        with (root / config['figures'][name]['source']).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
            assert rows
        assert (root / f'qa/pdf_render/{name}.png').is_file()
        records.append(dict(figure=name, PDF_pages=1, raster_images=0, all_fonts_embedded=True,
            text_characters=len(text), PNG_width=width, PNG_height=height, PNG_dpi=list(dpi), source_rows=len(rows)))
    atlas = PdfReader(root / 'figures/figure_atlas.pdf')
    assert len(atlas.pages) == 16
    assert all(len(p.images) == 0 for p in atlas.pages)
    assert all(all(embedded(f) for f in p['/Resources']['/Font'].get_object().values()) for p in atlas.pages)
    sheet = Image.new('RGB', (2560, 2040), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    for i, name in enumerate(names):
        with Image.open(root / f'qa/pdf_render/{name}.png') as rendered:
            rendered = rendered.convert('RGB')
            rendered.thumbnail((630, 470))
            x, y = (i % 4) * 640, (i // 4) * 510
            sheet.paste(rendered, (x + (640-rendered.width)//2, y + 30 + (470-rendered.height)//2))
            draw.text((x + 8, y + 8), name, fill='black', font=font)
    sheet.save(root / 'qa/pdf_render/contact_sheet.png')
    result = dict(status='PASSED' if args.visual_reviewed else 'STRUCTURE_PASSED_PENDING_VISUAL_REVIEW',
        atlas_pages=16, individual_figures=16, raster_images=0, all_fonts_embedded=True,
        latest_all_figure_renders_visually_reviewed=args.visual_reviewed,
        checks=['PNG600dpi', 'PDF/SVG vector', 'embedded fonts', 'no raster XObjects',
            'source CSV', 'plotting config', 'all16 latest rendered pages'], figures=records)
    (root / 'qa/vector_pdf_audit.json').write_text(json.dumps(result, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    print({k:v for k,v in result.items() if k!='figures'})


if __name__ == '__main__':
    main()
