from config import ACTIVE_CLASSES

with open("labelmap_voc.txt", "w") as f:
    f.write("background\n")
    for cls in ACTIVE_CLASSES:
        f.write(cls + "\n")