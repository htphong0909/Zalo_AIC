import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out_zip = root / 'scripts' / 'scratch' / 'code_latest.zip'
exclude_dirs = {
    '.git', '__pycache__', '.pytest_cache', 'archive', 'temp',
    'reports', 'data', 'weights', 'models', '.cache', 'experiments',
    'references', 'scratch', 'chunks', 'chunks_ckpt50'
}

count = 0
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in ['src', 'tests', 'configs', 'scripts']:
        for p in (root / folder).rglob('*'):
            if p.is_file() and p.suffix != '.zip' and not any(x in p.parts for x in exclude_dirs):
                z.write(p, p.relative_to(root))
                count += 1

print(f'Packaged {count} files into code_latest.zip: {out_zip.stat().st_size / 1024:.1f} KB')
