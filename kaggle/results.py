from loaddata import *
from mapping import *

def resultstable(dirname):
    cm = ClassMapping()
    df_anno, df_categories, df_images = loaddata()
    results = loadmergedresults([dirname])
    results["seq_id"] = results.merge(df_images, on="image_id")["seq_id"]
    results["category_id"] = results["net_id"].apply(cm)
    results = results.merge(df_categories, on="category_id")
    print("Merged ", len(results))
    return results

def resultstabletest(dirname, df_test_info_gps=None):
    cm = ClassMapping()
    if df_test_info_gps is None:
        df_test_info_gps = load_testinfo_gps()
    df_anno, df_categories, df_images = loaddata()
    dets = loaddets_md3()
    print("all boxes",len(dets))
    dets = pd.merge(df_test_info_gps, dets, on="image_id")
    print("testset",len(dets))
    print("#seq-test", len(dets.seq_id.unique()))
    print("#img-test", len(dets.image_id.unique()))
    results = loadmergedresults([dirname])
    #results["seq_id","CC"] = results.merge(df_test_info_gps, on="image_id")["seq_id","CC"]
    results = dets.merge(results, on="image_id", how="right")
    print("#seq", len(results.seq_id.unique()))
    results["category_id"] = results["net_id"].apply(cm)
    results = results.merge(df_categories, on="category_id")
    print("Merged ", len(results))
    print("#seq", len(results.seq_id.unique()))
    return results

def netid(results):
    cm = ClassMapping()
    results["net_id"] = results["category_id"].apply(cm.reverse)
    return results
