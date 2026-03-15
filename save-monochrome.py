import os
from PIL import Image
import last_folder_helper

def convert_indexed(img, colors=256):
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    pal = img.quantize(colors=colors, method=Image.FASTOCTREE, dither=Image.FLOYDSTEINBERG)
    return pal

def process_folder(directory):
    directory = os.path.expanduser(directory)
    out_dir = os.path.join(directory, "converted")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for name in os.listdir(directory):
        if not name.lower().endswith((".png",".jpg",".jpeg",".webp")):
            continue
        in_path = os.path.join(directory, name)
        out_name = os.path.splitext(name)[0] + ".png"
        out_path = os.path.join(out_dir, out_name)
        try:
            img = Image.open(in_path)
            out = convert_indexed(img, colors=256)
            out.save(out_path, optimize=True)
            print(f"converted {name}")
        except Exception as e:
            print(f"error {name}: {e}")

if __name__ == "__main__":
    default = last_folder_helper.get_last_folder()
    user_input = input(f"Input folder ({default}): ").strip()
    folder = user_input or default
    if not folder:
        folder = "."
    last_folder_helper.save_last_folder(folder)
    process_folder(folder)
