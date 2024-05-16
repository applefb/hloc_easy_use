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
from hloc.pipelines.Cambridge.utils import create_query_list_with_fixed_intrinsics, evaluate,create_query_list_with_fixed_intrinsics_use_imagedir
from hloc.utils import viz_3d
from pprint import pformat
import pdb
import time


# images = Path("datasets/sacre_coeur_test")
colmap_images = Path("datasets/sacre_coeur_test/all_images/colmap_images")
query_images = Path("datasets/sacre_coeur_test/all_images/query")
all_images = Path("datasets/sacre_coeur_test/all_images")
my_model = Path("datasets/sacre_coeur_test/my_model")
outputs = Path("outputs/demo/")

sfm_pairs = outputs / "pairs-sfm.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / "sfm"
features = outputs / "features.h5"
matches = outputs / "matches.h5"
loc_matches = outputs / "loc_matches.h5"
globals_features = outputs / "globals_features.h5"

feature_conf = extract_features.confs["disk"]
matcher_conf = match_features.confs["disk+lightglue"]
# list the standard configurations available
retrieval_conf = extract_features.confs["netvlad"]
print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")  # list the standard configurations available



# Extract local features for d
# atabase and query images，总共提取78张照片，生成  "features.h5"  102.2M
features = extract_features.main(feature_conf, colmap_images, feature_path=features)
# pdb.set_trace()

# 然后对 78张colmap重建的照片进行提取共视,生成 pairs-sfm.txt
pairs_from_covisibility.main(my_model, sfm_pairs, num_matched=20)
# pdb.set_trace()

# 针对78张的共视信息，进行匹配，最大是1560项    生成 matches.h5  26.4M
sfm_matches = match_features.main(
    matcher_conf, sfm_pairs, features, matches=matches
)

# pdb.set_trace()

reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build  这个基本作为定位的参考地图

# 针对 78 张图片，借鉴my_model的相机位置，进行三角化       生成 sfm_superpoint+superglue
# reconstruction = triangulation.main(
#     reference_sfm, my_model, colmap_images, sfm_pairs, features, sfm_matches
# )
# pdb.set_trace()

# 先把所有78张图片进行提取全局特征，这里必须要用netvlad特征，，，生成"globals_features.h5"   800K
global_descriptors = extract_features.main(retrieval_conf, colmap_images, feature_path=globals_features)
# pdb.set_trace()
# fig = viz_3d.init_figure()
# viz_3d.plot_reconstruction(
#     fig, reconstruction, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
# )
# fig.show()


###################################################################################################################

# 针对需要定位的3帧，提取局部特征，总共提取3张照片，生成生成  "features.h5"  106M
features = extract_features.main(feature_conf, query_images, feature_path=features, overwrite=True)
# pdb.set_trace()
start_time = time.time()
# 针对需要定位的3帧，提取全局描述子，并添加到，，，生成"globals_features.h5"   831K==============主要耗时4.5s,总共才6.4s
global_descriptors = extract_features.main(retrieval_conf, query_images, feature_path=globals_features, overwrite=True)
# pdb.set_trace()
#
end_time = time.time()

execution_time = (end_time - start_time) * 1000  # 将秒转换为毫秒
print("Function execution time:", execution_time, "milliseconds")
# 然后对需要查询的图片，进行提取共视信息,,,,,生成pairs-loc.txt
pairs_from_retrieval.main(
    global_descriptors, loc_pairs, num_matched=10, db_prefix="frame", query_prefix="query"
)
# pdb.set_trace()

#  针对 3张查询的图片，每张有10个共视   "loc_matches.h5"
loc_matches = match_features.main(
    matcher_conf, loc_pairs, features, matches=loc_matches
)
# pdb.set_trace()

query_list = outputs / 'query_list_with_intrinsics.txt'
# query_list_my = outputs / 'query_list_my.txt'
test_list = my_model / 'list_test.txt'
# 生成  query_list_with_intrinsics.txt
# 这一步是在colmap模型中查询 要重定位图片的相机内参，这不就是要求，colmap模型中必须要有需要查询的图片吗？
# 我重写了这个函数，这里我全部用第一张图片的内参，代替，
# create_query_list_with_fixed_intrinsics(my_model, query_list, test_list)
create_query_list_with_fixed_intrinsics_use_imagedir(my_model, query_list, image_dir=query_images)
results = outputs / "loc.txt"  # the result file

localize_sfm.main(
    reference_sfm,
    query_list,
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False,
)  # not required with SuperPoint+SuperGlue


# pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
# viz_3d.plot_camera_colmap(
#     fig, pose, camera, color="rgba(0,255,0,0.5)", name=query, fill=True
# )

# # 在 matches.h5  的基础上，添加query的匹配
# match_features.main(
#     matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
# );
