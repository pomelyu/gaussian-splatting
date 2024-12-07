import argparse
import math
from argparse import Namespace
from pathlib import Path

import torch
import torchvision
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)

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

    # pipeline_params = Namespace(
    #     convert_SHs_python=False,     # TODO: meaning?
    #     compute_cov3D_python=False,   # TODO: meaning?
    #     debug=False,
    #     antialiasing=False,
    # )
    # rendering = render(camera, gaussians, pipeline_params, background, use_trained_exp=False, separate_sh=False)["render"]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, depth_image = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=None,
        shs=gaussians.get_features,
        colors_precomp=None,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
    )
    rendered_image = rendered_image.clamp(0, 1)

    gt = camera.original_image[0:3, :, :]
    print(rendered_image.shape, gt.shape)

    vis = torch.cat([gt, rendered_image], 2)
    torchvision.utils.save_image(vis, out_dir / f"out_{args.frame}.jpg")


if __name__ == "__main__":
    main()
