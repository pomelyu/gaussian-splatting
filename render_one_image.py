import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torchvision

from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import SceneInfo, sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="path to training checkpoint(.ply)")
    parser.add_argument("--colmap", type=str, help="colmap project used in training ", default="data/treehill/")
    parser.add_argument("--frame", "-n", type=int, default=10, help="frame number")
    parser.add_argument("--output", "-o", type=str, default="results")
    # Model Params
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.model)
    # TODO: what is gaussian.create_from_pcd

    background = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.Tensor(background).to(device)

    # TODO: breakdown scene
    scene_info: SceneInfo = sceneLoadTypeCallbacks["Colmap"](
        path=args.colmap,
        images="images",
        depths="",
        eval=False,
        train_test_exp=False,
    )
    # TODO: breakdown camera
    train_cameras = scene_info.train_cameras
    train_cameras = cameraList_from_camInfos(
        train_cameras,
        resolution_scale=1.0,
        args=Namespace(resolution=-1, train_test_exp=False, data_device=device),
        is_nerf_synthetic=scene_info.is_nerf_synthetic,
        is_test_dataset=False,
    )
    print(len(train_cameras))
    camera = train_cameras[args.frame]

    # TODO: breakdown rendering
    pipeline_params = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    rendering = render(camera, gaussians, pipeline_params, background, use_trained_exp=False, separate_sh=False)["render"]
    gt = camera.original_image[0:3, :, :]
    print(rendering.shape, gt.shape)

    vis = torch.cat([gt, rendering], 2)
    torchvision.utils.save_image(vis, out_dir / f"out_{args.frame}.jpg")


if __name__ == "__main__":
    main()
