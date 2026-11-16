# ==============================================================================
# Zalo AI Challenge 2026 — 1-Click Google Drive Submission Transfer (Team 67Cat)
# Run this entire block inside Google Colab with GPU/CPU runtime.
# ==============================================================================
import json
import os
import urllib.request
from pathlib import Path

# 1. Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    dest_dir = Path('/content/drive/MyDrive')
except Exception:
    dest_dir = Path('./')

print("Fetching release assets from GitHub...", flush=True)
release_api = "https://api.github.com/repos/htphong0909/Zalo_AIC/releases/tags/v1.0.0-submission"
req = urllib.request.Request(release_api, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())

assets = sorted(data.get("assets", []), key=lambda x: x["name"])
print(f"Found {len(assets)} files to download:", flush=True)
for a in assets:
    print(f" - {a['name']} ({a['size'] / (1024*1024):.1f} MB)", flush=True)

# 2. Download all assets in parallel or sequentially using curl
for a in assets:
    fname = a["name"]
    url = a["browser_download_url"]
    print(f"Downloading {fname} ...", flush=True)
    os.system(f"curl -L -s -S '{url}' -o '{fname}'")

# 3. Reassemble chunks directly into Google Drive
target_tar = dest_dir / "zac2026_67Cat.tar.gz"
target_md5 = dest_dir / "zac2026_67Cat.tar.gz.md5"

print(f"\nReassembling chunks -> {target_tar} ...", flush=True)
os.system(f"cat zac2026_67Cat.tar.gz.part_* > '{target_tar}'")
os.system(f"cp zac2026_67Cat.tar.gz.md5 '{target_md5}'")

# 4. Verify MD5 checksum
print("\nVerifying MD5 checksum...", flush=True)
ret = os.system(f"cd '{dest_dir}' && md5sum -c '{target_md5.name}'")
if ret == 0:
    print("\n" + "="*70)
    print("SUCCESS! File zac2026_67Cat.tar.gz is READY in Google Drive!")
    print(f"Location: {target_tar}")
    print(f"Checksum: {target_md5}")
    print("You can now right-click zac2026_67Cat.tar.gz in Google Drive -> Share -> Copy Link to submit.")
    print("="*70)
else:
    print("\n[ERROR] MD5 checksum verification failed.")
