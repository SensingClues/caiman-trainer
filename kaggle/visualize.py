import os
from PIL import ImageDraw, ImageFont
import pandas as pd
from PIL import Image, ImageFilter

from matplotlib import pyplot as plt

def draw_bboxs(detections_list, name, im, results=None, fn=None, thresh=1.0):
    """
    detections_list: list of set includes bbox.
    im: image read by Pillow.
    """
    large_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 40)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 16)

    for idx, detection in enumerate(detections_list, 1):
        x1, y1,w_box, h_box = detection["bbox"]
        ymin,xmin,ymax, xmax=y1, x1, y1 + h_box, x1 + w_box
        draw = ImageDraw.Draw(im)
        
        imageWidth=im.size[0]
        imageHeight= im.size[1]
        (left, right, top, bottom) = (xmin * imageWidth, xmax * imageWidth,
                                      ymin * imageHeight, ymax * imageHeight)
        color='Green'
        
        text = str(idx)
        if results is not None:
            r = results[results["idx"] == str(idx)]
            if len(r):
                text += ","
                text += str(results[results["idx"] == str(idx)].iloc[0]["category_id"])
                text += ","
                text += str(results[results["idx"] == str(idx)].iloc[0]["confidence"])
                result_name = str(r.iloc[0]["name"])
                text += str(result_name)
#                if name != result_name:
                if r.iloc[0]["confidence"] < thresh:
                    color='Red'
            else:
                text = "empty"
                color = 'Gray'

        draw.text((int(left), int(top)), text, font=small_font)

#        draw.text((5,9), name, font=large_font)
        if fn is not None:
            draw.text((5,9), fn, font=small_font)

        draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=2, fill=color)


def draw_image(im_path, size, ann, overlay=True, results=None, thresh=1.0):
    im = Image.open(im_path)
    im = im.resize(size)
    if overlay:
        draw_bboxs(ann['detections'], ann['name'], im, results, os.path.basename(im_path), thresh=thresh)
    plt.imshow(im)

def visualize_sequence(merged, seq, path="train", overlay=True, size = (1024, 768), results=None, thresh=1.0):
    boxes = merged[merged["seq_id"] == seq]
    fig = plt.figure(figsize=(55, 46))

    images = boxes.groupby("image_id")
    for i, (idx,item) in enumerate(images):
        im_path = os.path.join(path, idx +'.jpg')
        ax = fig.add_subplot(4, 3, i+1, xticks=[], yticks=[])
        ax.margins(x=1, y=1)
#        if results is not None:
#            image_results = results[results["image_id"] == item["image_id"]]
#            draw_image(im_path, size, item, overlay, image_results)
#        else:
        for idx,box in item.iterrows():
            draw_image(im_path, size, box, overlay, item, thresh)
