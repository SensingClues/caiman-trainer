import json
import os
import pickle
import cv2
import numpy as np 
import pandas as pd
from PIL import Image, ImageFile, ImageFont, ImageDraw
from clahe import CLAHE, convert_dir_to_clahe

class Exporter:
    LOWER_THRESHOLD = 0.5

    def __init__(self):
        self.x_tot_list = []
        self.x2_tot_list = []

    def calc_x_and_x2_tot(self, img):
        # calculate mean and variance of a PIL image    
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img,(2,0,1))
                                
        return (img/255.0).reshape(-1,3).mean(0), ((img/255.0)**2).reshape(-1,3).mean(0)

    def export_detections(self, df, detector: str, size, threshold=LOWER_THRESHOLD):
        # construct a sensible (unique) path
        prefix = os.path.split(df["path"].unique()[0])[0]
        postfix = detector + "_" + str(size[0]) + "x" + str(size[1])
        dest_dir = "_" + prefix + "_bbox_" + postfix
        os.makedirs(dest_dir)
        self._dest_dir = dest_dir
    
        boxes = []
        for _rowid, row in df.iterrows():
            filename = row["filename"]
            image_id = row["image_id"]
        
            if not os.path.exists(filename):
                print("ERROR: file doesn't exist")
                continue
            
            try:      
                img = Image.open(filename)
            except:
                print(f"Passed {filename}. Fail to open image.")
                continue
        
            detections = row["detections"]
            for idx, detection in enumerate(detections, 1):

                if detection["conf"] < threshold:
                    continue

                if detection["category"] != "1":
                    continue
            
                crop_area = self.get_crop_area(detection["bbox"], img.size)
                img_cropped = img.crop(crop_area).resize(size)
            
                dest_file = os.path.join(dest_dir, f"{image_id}_{idx}.jpg")
                img_cropped.save(dest_file, format="jpeg")
                boxes.append((dest_dir, image_id, idx, dest_file, detection["conf"], row["category_id"]))
    
                x_tot, x2_tot = self.calc_x_and_x2_tot(img_cropped)
                self.x_tot_list.append(x_tot)
                self.x2_tot_list.append(x2_tot)
    
        self._df =  pd.DataFrame(boxes, columns=["path", "image_id","idx","filename","confidence","category_id"])
        return self._df


    def get_crop_area(self, bbox, image_size):
        x1, y1,w_box, h_box = bbox
        ymin,xmin,ymax, xmax = y1, x1, y1 + h_box, x1 + w_box
        area = (xmin * image_size[0], ymin * image_size[1], 
            xmax * image_size[0], ymax * image_size[1])
        return area

    def stats(self):
        img_avr =  np.array(x_tot_list).mean(0)
        img_std =  np.sqrt(np.array(x2_tot_list).mean(0) - img_avr**2)
        print('mean:',img_avr, ', std:', img_std)

    def to_csv(self):
        self._df.to_csv(self._dest_dir + ".csv", index=False)

