import sys
sys.path.append('../ma-sh/')

import os
import torch
import open3d as o3d
from tqdm import tqdm
from math import sqrt, ceil

from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries



from fast_dit.Module.sampler import Sampler

def demo():
    model_file_path = './output/pretrain-v1/model_last.pt'
    sample_num = 9
    category_id = 18

    print(model_file_path)
    sampler = Sampler(model_file_path)

    print("start diffuse", sample_num, "mashs....")
    sampled_array = sampler.sample(sample_num, category_id)

    print(
        sampled_array.shape,
        sampled_array.max(),
        sampled_array.min(),
        sampled_array.mean(),
        sampled_array.std(),
    )

    object_dist = [2, 0, 2]

    row_num = ceil(sqrt(sample_num))

    mash_pcd_list = []

    mash_model = sampler.toInitialMashModel()

    for i in tqdm(range(sample_num)):
        mash_params = sampled_array[i]

        sh2d = 2 * sampler.sh_2d_degree + 1

        rotation_vectors = mash_params[:, :3]
        positions = mash_params[:, 3:6]
        mask_params = mash_params[:, 6 : 6 + sh2d]
        sh_params = mash_params[:, 6 + sh2d :]

        mash_model.loadParams(mask_params, sh_params, rotation_vectors, positions)
        mash_pcd = getPointCloud(toNumpy(torch.vstack(mash_model.toSamplePoints()[:2])))

        if True:
            translate = [
                int(i / row_num) * object_dist[0],
                0 * object_dist[1],
                (i % row_num) * object_dist[2],
            ]

            mash_pcd.translate(translate)

        mash_pcd_list.append(mash_pcd)

    if False:
        renderGeometries(mash_pcd_list, "sample mash point cloud")

    if True:
        os.makedirs('./output/', exist_ok=True)
        for i in range(len(mash_pcd_list)):
            o3d.io.write_point_cloud(
                "./output/sample_mash_pcd_" + str(i) + ".ply",
                mash_pcd_list[i],
                write_ascii=True,
            )

    return True
