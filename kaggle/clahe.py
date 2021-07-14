
from PIL import Image, ImageFile, ImageFont, ImageDraw
import numpy as np
import cv2
import os
import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __call__(self, img):
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        res= Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        return res

def convert_dir_to_clahe(dirname):
    c = CLAHE()
    dest_dir = dirname + "_clahe"
    os.makedirs(dest_dir)
    extensions = {".jpg", ".png", ".gif"} #etc
    files = [f for f in os.listdir(dirname) if any(f.endswith(ext) for ext in extensions)]

    for filename in tqdm.tqdm(files):
        try:
            src = Image.open(os.path.join(dirname, filename)).convert('RGB')
            dst = c(src)
            dst.save(os.path.join(dest_dir, filename))
        except Exception as e:
            print("ERROR: ", e)



if __name__ == "__main__":
    convert_dir_to_clahe("test")
    convert_dir_to_clahe("train")
