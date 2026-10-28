# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
import scipy
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import colorsys

from sklearn.decomposition import PCA
pca = PCA(n_components=3, svd_solver='full')
from PIL import Image
import torchvision


def _load_sg_config():
    """REFINE section of config/<dataset>.py (--dataset-config / $DATASET_CONFIG, default vlsplat_render_scannetpp)."""
    from config.load_config import load_dataset_config

    return load_dataset_config()["REFINE"]


SG_CONFIG = _load_sg_config()

def _extra_checkpoint_path(base_checkpoint_path: str) -> str:
    """
    Build sidecar checkpoint path for code-book related states.
    Example: /path/chkpnt10000.pth -> /path/chkpnt10000_extra.pth
    """
    root, ext = os.path.splitext(base_checkpoint_path)
    if not ext:
        ext = ".pth"
    return root + "_extra" + ext


def _scene_graph_doc(scene_graph_cfg, sg_state):
    """The scene graph being refined: loaded once from --scene-graph, updated in place by every round."""
    if "doc" not in sg_state:
        from scene_graph_builder.graph_io import load_scene_graph

        sg_state["doc"] = load_scene_graph(scene_graph_cfg["path"])
    return sg_state["doc"]


def _save_scene_graph_snapshot(scene, iteration, scene_graph_cfg, sg_state) -> None:
    """Write the current scene graph as scene_graph.json next to the saved point cloud."""
    if not scene_graph_cfg:
        return
    from scene_graph_builder.graph_io import save_scene_graph

    path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}", "scene_graph.json")
    save_scene_graph(path, _scene_graph_doc(scene_graph_cfg, sg_state))
    print(f"[scene_graph] saved snapshot -> {path}", flush=True)


def _maybe_run_scene_graph_refine(iteration, gaussians, code_book, dataset, scene_graph_cfg, sg_state) -> None:
    """Run one intra-/inter-refine round every refine_interval iterations from refine_start to refine_end.

    The graph after each round is kept in <model_path>/scene_graph_log/scene_graph_<iteration>.json."""
    if not scene_graph_cfg:
        return
    interval = max(1, int(scene_graph_cfg["refine_interval"]))
    if not (int(scene_graph_cfg["refine_start"]) <= int(iteration) <= int(scene_graph_cfg["refine_end"]) and int(iteration) % interval == 0):
        return
    from scene_graph_builder.graph_io import save_scene_graph
    from scene_graph_builder.refine_scene_graph import run_scene_graph_refinement_cycle

    msg = run_scene_graph_refinement_cycle(
        iteration, gaussians, code_book, _scene_graph_doc(scene_graph_cfg, sg_state), dataset.model_path, scene_graph_cfg
    )
    if not scene_graph_cfg["quiet_node_refine"]:
        print(f"[scene_graph] {msg}", flush=True)
    save_scene_graph(os.path.join(dataset.model_path, "scene_graph_log", f"scene_graph_{iteration}.json"), _scene_graph_doc(scene_graph_cfg, sg_state))


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5


    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


@torch.no_grad()
def get_confience_map(feature, gt_obj):

    H,W = feature.shape[1],feature.shape[2]
    feature = feature.reshape(16, -1).T


    gt_obj = gt_obj.reshape(-1)
    wh = feature.shape[0]


    sam_t = gt_obj
    sam_o = feature


    sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6).detach()


    min_pixnum = 0
    cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
    cluster_ids = cluster_ids[cnums_all > min_pixnum]
    cnums = cnums_all[cnums_all > min_pixnum]
    cnum = cluster_ids.shape[0] # cluster number

    u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
    phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)


    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]
        u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
        phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))


    phi_list = torch.clip(phi_list * 0.1, min=0.1, max=1.0)
    phi_list = phi_list.detach()

    confidence = torch.zeros([wh,1]).cuda()
    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]

        dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]
        confidence[sam_t == cluster_ids[i]] = dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)


    confidence = confidence.reshape(H,W)
    confidence[confidence> 0.2] = 1
    confidence[confidence<= 0.2] = 0.0
    return confidence

def get_contrastive_loss(feature, gt_obj):


    feature = feature.reshape(16, -1).T

    loss_regularization =  ((torch.norm(feature, dim=-1, keepdim=True) - 1.0) ** 2).mean()

    gt_obj = gt_obj.reshape(-1)
    wh = feature.shape[0]


    batchsize = 32768
    random_idx = torch.randint(0, wh, [batchsize])

    sam_t = gt_obj[random_idx]
    sam_o = feature[random_idx]


    sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6).detach()


    min_pixnum = 20
    cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
    cluster_ids = cluster_ids[cnums_all > min_pixnum]
    cnums = cnums_all[cnums_all > min_pixnum]
    cnum = cluster_ids.shape[0] # cluster number

    u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
    phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)


    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]
        u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
        phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))


    phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)
    phi_list = phi_list.detach()

    ProtoNCE = torch.zeros([1], dtype=torch.float32, device=sam_o.device)

    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]

        dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]


        ProtoNCE += -torch.sum(torch.log(
            dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
            ))


    ProtoNCE = ProtoNCE/cnum
    return ProtoNCE,loss_regularization


def feature_to_rgb(features):
    # Input features shape: (16, H, W)

    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    features_reshaped =  features_reshaped / (torch.norm(features_reshaped, dim=-1, keepdim=True) + 1e-6).detach()
    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')
    return rgb_array

@torch.no_grad()
def create_virtual_gt_with_linear_assignment(labels_gt, predicted_scores):
    labels_gt = labels_gt.reshape(-1)
    predicted_scores = predicted_scores.reshape(256,-1).T
    labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
    predicted_probabilities = torch.softmax(predicted_scores, dim=-1)
    cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
    for lidx, label in enumerate(labels):
        cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0)).cpu().numpy()

    assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
    new_labels = torch.zeros_like(labels_gt)
    for aidx, lidx in enumerate(assignment[0]):
        new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
    return new_labels

def clustering_for_matching(objects, virtual_gt_labels, code_book, confidence_map):
    # objects: (16, N) rendered features; code_book: (K, 16).
    if (confidence_map>0.5).sum()==0:
        return torch.tensor(0., device=objects.device, requires_grad=True)
    confidence_map = confidence_map.reshape(-1)
    virtual_gt_labels = virtual_gt_labels.reshape(-1)[confidence_map>0.5]
    codebook_feature = code_book[virtual_gt_labels]
    objects = objects.detach().T
    objects = objects/(torch.norm(objects, dim=-1, keepdim=True) + 1e-6)
    objects = objects[confidence_map>0.5]
    clustering_loss = (torch.norm(objects-codebook_feature, dim=-1, keepdim=True)).mean()
    return clustering_loss


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    weight_loss,
    scene_graph_cfg=None,
):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1, bias=False)
    code_book = classifier.weight.reshape(256, 16).detach().clone().to(device="cuda").requires_grad_(True)

    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam([code_book], lr=5e-4)
    code_book.cuda()
    if checkpoint:
        # Local training checkpoints may contain pickled Python objects.
        # PyTorch 2.6 changed torch.load default weights_only=True, so we
        # explicitly opt into the previous behavior for trusted local files.
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        extra_ckpt = _extra_checkpoint_path(checkpoint)
        if os.path.exists(extra_ckpt):
            extra_state = torch.load(extra_ckpt, map_location="cuda", weights_only=False)
            if "code_book" in extra_state:
                loaded_code_book = extra_state["code_book"].to(device="cuda")
                if loaded_code_book.shape != code_book.shape:
                    raise RuntimeError(
                        f"Shape mismatch for code_book: expected {tuple(code_book.shape)} "
                        f"but got {tuple(loaded_code_book.shape)} from {extra_ckpt}"
                    )
                code_book.data.copy_(loaded_code_book)
            if "cls_optimizer" in extra_state:
                cls_optimizer.load_state_dict(extra_state["cls_optimizer"])
            print(f"[Checkpoint] Loaded extra states from {extra_ckpt}")
        else:
            print(f"[Checkpoint] Extra checkpoint not found: {extra_ckpt} (code_book starts fresh)")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    sg_state = {}
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Training progress",
        position=0,
        dynamic_ncols=True,
        leave=True,
    )
    if scene_graph_cfg and int(first_iter) == int(scene_graph_cfg.get("refine_start", -1)):
        with torch.no_grad():
            _maybe_run_scene_graph_refine(
                first_iter,
                gaussians,
                code_book,
                dataset,
                scene_graph_cfg,
                sg_state,
            )
            progress_bar.refresh()
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        gt_obj = viewpoint_cam.objects.cuda().long()
        objects_2d = objects
        contrastive_loss, regularization = get_contrastive_loss(objects_2d, gt_obj)
        confidence_map = get_confience_map(objects_2d, gt_obj)

        objects = objects.reshape(16, -1)
        logits = torch.matmul(code_book, objects.detach())
        H,W = gt_obj.shape[0], gt_obj.shape[1]
        logits = logits.reshape(256, H,W)

        virtual_gt_labels = create_virtual_gt_with_linear_assignment(gt_obj, logits)
        predicted_labels = logits.argmax(dim=0)
        virtual_gt_labels = virtual_gt_labels.reshape(H,W)

        noise_flag = (confidence_map>0.5).sum()
        if torch.any(virtual_gt_labels != predicted_labels) and noise_flag>0:  # classification loss only when the pseudo labels differ from the prediction
            loss_obj_cls = (((cls_criterion(logits.unsqueeze(0), virtual_gt_labels.unsqueeze(0))).squeeze())[confidence_map>0.5]).mean()
        else:
            loss_obj_cls = torch.tensor(0., device=logits.device, requires_grad=True)
        loss_obj_cls = loss_obj_cls / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

        if noise_flag>0:
            clustering_loss = clustering_for_matching(objects, virtual_gt_labels, code_book,confidence_map)
        else:
            clustering_loss = torch.tensor(0., device=logits.device, requires_grad=True)


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss_obj_3d = None
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + contrastive_loss * 1e-6  + loss_obj_cls * 1e-4  + clustering_loss * weight_loss +  regularization  * 1e-6

        loss.backward()
        if getattr(opt, "freeze_gaussian_xyz", False) and gaussians._xyz.grad is not None:
            gaussians._xyz.grad.zero_()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Fea: {contrastive_loss}, reg: {regularization}, classification: {loss_obj_cls}, clustering_loss: {clustering_loss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, contrastive_loss, loss_obj_cls, regularization, clustering_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(code_book, os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))
                _save_scene_graph_snapshot(
                    scene,
                    iteration,
                    scene_graph_cfg,
                    sg_state,
                )


            if iteration % 500 ==0:
                objects = objects.reshape(16, H, W)
                rgb_mask = feature_to_rgb(objects)

                confidence_map = (confidence_map.squeeze().cpu().numpy() * 255).astype(np.uint8)


                os.makedirs(os.path.join(scene.model_path, "save_img"), exist_ok = True)
                Image.fromarray(confidence_map).save(os.path.join(scene.model_path, "save_img/confidence_Feature_iteration_{0:05d}.png".format(iteration)))

                Image.fromarray(rgb_mask).save(os.path.join(scene.model_path, "save_img/PCA_Feature_iteration_{0:05d}.png".format(iteration)))
                torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_img/RGB_iteration_{0:05d}.png".format(iteration)))


                pred_obj = torch.argmax(logits,dim=0)
                pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                Image.fromarray(pred_obj_mask).save(os.path.join(scene.model_path, "save_img/classification_{0:05d}.png".format(iteration)))


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:

                gaussians.optimizer.step()
                gaussians.enforce_frozen_xyz()
                gaussians.optimizer.zero_grad(set_to_none = True)
                cls_optimizer.step()
                cls_optimizer.zero_grad()

            # Scene graph intra/inter-refine: every refine_interval from refine_start to refine_end
            # (inclusive).
            _maybe_run_scene_graph_refine(
                iteration,
                gaussians,
                code_book,
                dataset,
                scene_graph_cfg,
                sg_state,
            )
            progress_bar.refresh()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                torch.save((gaussians.capture(), iteration), ckpt_path)
                extra_ckpt_path = _extra_checkpoint_path(ckpt_path)
                torch.save(
                    {
                        "iteration": iteration,
                        "code_book": code_book.detach().clone(),
                        "cls_optimizer": cls_optimizer.state_dict(),
                    },
                    extra_ckpt_path,
                )

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, contrastive_loss, loss_obj_cls, regularization, clustering_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d):
    pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--weight_loss", type=float, default=1e-0, help="Weight for clustering loss")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible training.")
    # Scene graph + intra/inter-refine in-loop (requires a scene graph JSON + clip_refine deps)
    parser.add_argument(
        "--unified-lift",
        action="store_true",
        help="Unified-Lift: mask-based lifting + codebook training only. "
        "Skip scene-graph ROI update and intra/inter-refine even if --scene-graph is set.",
    )
    parser.add_argument(
        "--scene-graph",
        type=str,
        default=None,
        help="Path to scene graph JSON: scene_graph.json or merged_scene_graph.json "
        "(e.g. from 2_init_scene_graph.sh).",
    )
    parser.add_argument("--dataset-config", type=str, default=None, help="config/<name>.py with the REFINE settings (default: $DATASET_CONFIG or vlsplat_render_scannetpp).")
    parser.add_argument("--sg-refine-start", type=int, default=15_000, help="First refinement round (the shell script passes the step-1 length).")
    # Every REFINE key of the dataset config is also a flag: --sg-<key with dashes>.
    for _key, _value in SG_CONFIG.items():
        _flag = "--sg-" + _key.replace("_", "-")
        if isinstance(_value, bool):
            parser.add_argument(_flag, type=lambda v: str(v).lower() in ("1", "true", "yes"), default=_value)
        else:
            parser.add_argument(_flag, type=type(_value), default=_value)
    parser.add_argument(
        "--no-densify",
        action="store_true",
        help="Disable densification/pruning/opacity reset (sets densify_until_iter to 0).",
    )
    parser.add_argument(
        "--fixed-gaussian",
        action="store_true",
        help="3D-eval mode: freeze Gaussian xyz and disable ADC/densify from the initial points.",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Training-mode settings from the DATASET section of config/<dataset>.py
    from config.load_config import load_dataset_config

    dataset_cfg = load_dataset_config()["DATASET"]
    args.densify_until_iter = dataset_cfg["densify_until_iter"]
    args.num_classes = dataset_cfg["num_classes"]
    if dataset_cfg["freeze_gaussian_xyz"]:
        args.freeze_gaussian_xyz = True
    if getattr(args, "fixed_gaussian", False):
        args.no_densify = True
        args.freeze_gaussian_xyz = True
    if getattr(args, "no_densify", False):
        args.densify_until_iter = 0

    print("Optimizing " + args.model_path)
    if int(getattr(args, "densify_until_iter", 0)) <= 0:
        print("[train] Densify/prune/opacity-reset disabled (densify_until_iter<=0).", flush=True)
    if getattr(args, "freeze_gaussian_xyz", False):
        print("[train] Gaussian xyz positions frozen (freeze_gaussian_xyz).", flush=True)
    if getattr(args, "fixed_gaussian", False):
        print("[train] fixed-gaussian / 3D-eval mode: no ADC + frozen xyz.", flush=True)
    if bool(getattr(args, "unified_lift", False)):
        print("[train] Unified-Lift: mask + codebook only (no scene-graph refine)")

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    scene_graph_cfg = None
    if args.scene_graph and not args.unified_lift:
        scene_graph_cfg = {key: getattr(args, "sg_" + key) for key in SG_CONFIG}
        scene_graph_cfg.update(path=args.scene_graph, refine_start=args.sg_refine_start)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.weight_loss,
        scene_graph_cfg=scene_graph_cfg,
    )

    # Last step: attach node-less groups, re-vote the classes of non-target nodes, and write the final scene_graph.json.
    if scene_graph_cfg is not None:
        from scene_graph_builder.finalize_scene_graph import finalize_scene_graph

        object_dir = os.path.join(args.source_path, args.object_path)
        finalize_scene_graph(
            os.path.join(args.model_path, "point_cloud", f"iteration_{max(args.save_iterations)}"),
            scene_graph_cfg,
            assign=dict(
                source_path=args.source_path, object_path=object_dir, label_json_root=object_dir.rstrip("/") + "_json",
                images=args.images, resolution=dataset_cfg["graph_resolution"], cfg=load_dataset_config()["GRAPH"],
            ),
        )

    # All done
    print("\nTraining complete.")
    sys.exit(0)
