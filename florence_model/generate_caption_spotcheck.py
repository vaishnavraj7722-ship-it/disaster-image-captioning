"""
Randomly samples N rows from the disaster captions file and generates a single
HTML page showing each image next to its caption, so you can visually spot-check
caption accuracy in one scroll instead of manually cross-referencing Drive + sheet.
"""

import os
import base64
import random
import pandas as pd

CAPTIONS_FILE = "../data/Final_Dataset_Caption.xlsx"
IMAGE_DIR = "../data/disaster_images"
OUTPUT_HTML = "caption_spotcheck.html"
SAMPLE_SIZE = 50
SEED = 42

df = pd.read_excel(CAPTIONS_FILE, header=None, names=["filename", "caption"])
df["filename"] = df["filename"].astype(str).str.strip()

random.seed(SEED)
sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).reset_index(drop=True)

def image_to_base64(path):
    """Embed image directly in HTML so it's a single self-contained file - no broken links."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

rows_html = []
skipped = 0

for i, row in sample.iterrows():
    image_path = os.path.join(IMAGE_DIR, row["filename"])
    if not os.path.exists(image_path):
        skipped += 1
        continue

    ext = row["filename"].split(".")[-1].lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    b64 = image_to_base64(image_path)

    rows_html.append(f"""
    <div class="row">
        <div class="img-cell">
            <img src="data:image/{mime};base64,{b64}" />
        </div>
        <div class="text-cell">
            <div class="filename">{row['filename']}</div>
            <div class="caption">{row['caption']}</div>
        </div>
    </div>
    """)

html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Disaster Caption Spot Check</title>
<style>
    body {{ font-family: sans-serif; background: #111; color: #eee; padding: 20px; }}
    h1 {{ font-size: 18px; }}
    .row {{
        display: flex; align-items: center; gap: 20px;
        border-bottom: 1px solid #333; padding: 16px 0;
    }}
    .img-cell img {{ max-width: 280px; max-height: 200px; border-radius: 6px; }}
    .filename {{ font-size: 12px; color: #888; margin-bottom: 6px; }}
    .caption {{ font-size: 16px; line-height: 1.4; }}
</style>
</head>
<body>
<h1>Spot-checking {len(rows_html)} random disaster captions (skipped {skipped} missing images)</h1>
{''.join(rows_html)}
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_HTML} with {len(rows_html)} samples (skipped {skipped} missing files)")
print(f"Open it with: firefox {OUTPUT_HTML}  (or copy it to your laptop and open locally)")