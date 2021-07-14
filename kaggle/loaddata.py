import pandas as pd
import numpy as np
import os
import json
import glob
import reverse_geocoder


def load_gps():
    with open("metadata/gps_locations.json") as f:
        res = json.load(f)
    ccs = dict()
    for r in res:
        coordinates = res[r]["latitude"], res[r]["longitude"]
        ccs[r] = reverse_geocoder.search(coordinates)[0]['cc']
    return ccs
            
def loaddata():
    # load train
    with open('metadata/iwildcam2021_train_annotations.json', encoding='utf-8') as json_file:
        train_annotations =json.load(json_file)
    df_images = pd.DataFrame(train_annotations["images"])
    df_images = df_images.rename({"id" : "image_id"}, axis=1)
    df_categories = pd.DataFrame.from_records(train_annotations["categories"])
    df_categories = df_categories.rename({"id" : "category_id"}, axis=1)
    df_annotations = pd.DataFrame(train_annotations["annotations"])
    return df_annotations, df_categories, df_images


def loaddets_md3():
    with open('metadata/iwildcam2021_megadetector_results.json', encoding='utf-8') as json_file:
        megadetector_results =json.load(json_file)
    df_dets = pd.DataFrame(megadetector_results["images"])
    df_dets = df_dets.rename({"id" : "image_id"}, axis=1)
    df_dets['nr_boxes'] = df_dets['detections'].str.len()
    return df_dets

def load_md4(filename):
    with open(filename, encoding='utf-8') as json_file:
        results =json.load(json_file)
    df = pd.DataFrame(results["images"])
 #   df = df.rename({"id" : "image_id"}, axis=1)
    df["short"] = df["file"].apply(lambda x: os.path.basename(x))
    df["image_id"] = df["short"].apply(lambda x: os.path.splitext(x)[0])
    df['nr_boxes'] = df['detections'].str.len()
    return df

def loaddets_md4():
    df_train = load_md4('metadata/iwildcam2021_train_md4_results.json')
    df_test = load_md4('metadata/iwildcam2021_test_md4_results.json')
    return df_train, df_test

def load_merged():
    df_anno, df_categories, df_images = loaddata()
    df_dets = loaddets_md3()
    df_anno_cat = pd.merge(df_anno, df_categories, on="category_id")
    df_anno_cat_im = pd.merge(df_anno_cat, df_images, on="image_id")
    df_merged = pd.merge(df_anno_cat_im, df_dets, on="image_id")
    return df_merged

def load_merged_gps():
    location = load_gps()
    merged = load_merged()
    def f(x):
        if str(x) in location:
            return location[str(x)]
        return "ZZ"
    # add the country code to the loaded data
    merged["CC"] = merged["location"].apply(lambda x: f(x))
    return merged

def load_results():
    results = pd.read_csv("topk_ids.csv", header=None)
    return results


def load_testinfo():
    with open('metadata/iwildcam2021_test_information.json', encoding='utf-8') as json_file:
        test_information =json.load(json_file)
            
    df_test_info = pd.DataFrame(test_information["images"])[["id", "seq_id"]]
    df_test_info = df_test_info.rename({"id" : "image_id"}, axis=1)
    return df_test_info

def load_testinfo_gps():
    location = load_gps()
    with open('metadata/iwildcam2021_test_information.json', encoding='utf-8') as json_file:
        test_information =json.load(json_file)
            
    df_test_info = pd.DataFrame(test_information["images"])
    df_test_info = df_test_info.rename({"id" : "image_id"}, axis=1)
    def f(x):
        if str(x) in location:
            return location[str(x)]
        return "ZZ"
    # add the country code to the loaded data
    df_test_info["CC"] = df_test_info["location"].apply(lambda x: f(x))
    return df_test_info

def mymode(row):
    return row.mode()[0]

def loadmergedresults(dirs):
    files = sorted(glob.glob(os.path.join(dirs[0], "topk_id*.csv")))
    r = pd.read_csv(files[0], header=None)[[0,1,2]]
    count = 0
    for dirname in dirs:
        print("Dir", dirname)
        files = glob.glob(os.path.join(dirname, "topk_id*.csv"))
        print("Read ", files[0], len(r))
        for f in files[1:]:
            a = pd.read_csv(f, header=None)[[0,1,2]]
            print("Read ", f, len(a))
            r = pd.merge(r, a, on=0, how="outer", suffixes=('', '_y'))
            del a
            count += 1

    print("Mode")
    labels = r.keys()[1::2]
    confs = r.keys()[2::2]
#    mode = r[labels].mode(axis=1, numeric_only=True)
#    r["mode"] = mode[0]
    r["mode"] = r.apply(lambda row : mymode(row[labels]), axis = 1)
    def f(x):
#        print(x)
#        print(x["mode"])
#        print(x[confs])
#        print(x[labels] == x["mode"])
        mask = list(x[labels] == x["mode"])
        return np.mean(x[confs][mask])

    r["conf"] = r.apply(f, axis=1)
    results = r[[0,"mode","conf"]].rename(columns={"mode": 1, "conf" : 2})
    results = results.rename(columns={0 : "filename", 1 : "net_id", 2 : "confidence" })
    # extract the box id from the filename
    a = results["filename"].str.extractall(r"(?P<image_id>.*)_(?P<idx>\d+)\.jpg")
    a.unstack()
    a.index = a.index.droplevel(1)
    results = results.join(a)
    print("Total ", len(results))
    return results

def load_olddata(dirname):
    files = os.listdir(dirname)
    r = pd.read_csv(os.path.join(dirname, files[0]), header=None)[[0,1]]
    for f in files:
        a = pd.read_csv(os.path.join(dirname, f), header=None)[[0,1]]
        #df["image_id"] = df[0].str[:36]
        #df["idx"] = df[0].str[37:-4]
        #df["net_id"] = df[1]
        r = pd.merge(r, a, on=0, how="outer", suffixes=('', '_y'))
    labels = r.keys()[1::]
    r["filename"] = r[0]
    r["net_id"] = r[labels].mode(axis=1, numeric_only=True)[0]
    r["image_id"] = r[0].str[:36]
    r["idx"] = r[0].str[37:-4]
    return r[["filename","net_id","image_id","idx"]]
