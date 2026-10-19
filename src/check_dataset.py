from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/raw/trashnet")

print("Datasat exists:", DATA_DIR.exists())
print("Dataset path:",  DATA_DIR.resolve())

classes = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
print("Classes:", classes)

for cls in classes: 
    folder = DATA_DIR / cls
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    print(f"{cls:10s} -> {len(images)} images")

fig, axes = plt.subplots(2,3, figsize=(9, 6))
axes = axes.ravel()

for ax, cls in zip(axes, classes):
    img_path = next((DATA_DIR / cls).glob("*"))
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(cls)
    ax.axis("off")

plt.tight_layout()
plt.show()