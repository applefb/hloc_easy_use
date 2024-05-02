import tqdm

from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
    localize_sfm
)
from hloc.visualization import plot_images, read_image
from hloc.pipelines.Cambridge.utils import create_query_list_with_fixed_intrinsics, evaluate
from hloc.utils import viz_3d
from pprint import pformat
import  pdb
# images = Path("datasets/sacre_coeur_test")
all_images = Path("datasets/sacre_coeur_test/all_images")
my_model = Path("datasets/sacre_coeur_test/my_model")
outputs = Path("outputs/demo/")

sfm_pairs = outputs / "pairs-sfm.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / "sfm"
features = outputs / "features.h5"
matches = outputs / "matches.h5"

feature_conf = extract_features.confs["disk"]
matcher_conf = match_features.confs["disk+lightglue"]
# list the standard configurations available
retrieval_conf = extract_features.confs["netvlad"]
print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")# list the standard configurations available



#Extract local features for database and query images，总共提取81张照片，生成  feats-disk.h5
features = extract_features.main(feature_conf, all_images, outputs)
pdb.set_trace()

# 然后对 78张colmap重建的照片进行提取共视,生成 pairs-sfm.txt
pairs_from_covisibility.main(my_model, sfm_pairs, num_matched=20)
pdb.set_trace()

# 针对78张的共视信息，进行匹配，最大是1560项    生成 feats-disk_matches-disk-lightglue_pairs-sfm.h5
sfm_matches = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)
pdb.set_trace()

reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build  这个基本作为定位的参考地图

# 针对 78 张图片，借鉴my_model的相机位置，进行三角化       生成 sfm_superpoint+superglue
reconstruction = triangulation.main(
    reference_sfm, my_model, all_images, sfm_pairs, features, sfm_matches
)
pdb.set_trace()




# 先把所有81张图片进行提取全局特征，这里必须要用netvlad特征，，，生成global-feats-netvlad.h5
global_descriptors = extract_features.main(retrieval_conf, all_images, outputs)
pdb.set_trace()

#
#
# 然后对需要查询的图片，进行提取共视信息,,,,,生成pairs-loc.txt
pairs_from_retrieval.main(
    global_descriptors, loc_pairs, num_matched=10, db_prefix="frame", query_prefix="query"
)
pdb.set_trace()


#  针对 3张查询的图片，每张有10个共视   feats-disk_matches-disk-lightglue_pairs-loc.h5
loc_matches = match_features.main(
    matcher_conf, loc_pairs, feature_conf["output"], outputs
)


query_list = outputs / 'query_list_with_intrinsics.txt'
test_list  = my_model / 'list_test.txt'
# 生成  query_list_with_intrinsics.txt
# 这一步是在colmap模型中查询 要重定位图片的相机内参，这不就是要求，colmap模型中必须要有需要查询的图片吗？
# 我重写了这个函数，这里我全部用第一张图片的内参，代替，
create_query_list_with_fixed_intrinsics(my_model, query_list, test_list)


results = outputs / "Aachen_hloc_superpoint+superglue_netvlad20.txt"  # the result file

localize_sfm.main(
    reconstruction,
    query_list,
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False,
)  # not required with SuperPoint+SuperGlue



# # 在 matches.h5  的基础上，添加query的匹配
# match_features.main(
#     matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
# );







