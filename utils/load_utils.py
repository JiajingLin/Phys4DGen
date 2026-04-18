from core.gaussian_model import GaussianModel
import open3d as o3d
import numpy as np
import torch

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        

def load_checkpoint(load, sh_degree=0):
    # Load guassians
    gaussians = GaussianModel(sh_degree) 
    gaussians.load_ply(load)    
    return gaussians


def load_particle(load, viz=False):
    pcd = o3d.io.read_point_cloud(load)     

    particle = np.asarray(pcd.points)  
    
    particle = torch.tensor(particle, dtype=torch.float32, device="cuda")
    
    return particle


def find_far_points(xyzs, selected_points, thres=0.05):
    """
    Args:
        xyzs: [N, 3]
        selected_points: [M, 3]
    Outs:
        freeze_mask: [N], 1 for points that are far away, 0 for points that are close
                    dtype=torch.int
    """
    chunk_size = 10000

    freeze_mask_list = []
    for i in range(0, xyzs.shape[0], chunk_size):

        end_index = min(i + chunk_size, xyzs.shape[0])
        xyzs_chunk = xyzs[i:end_index]
        # [M, N]
        cdist = torch.cdist(xyzs_chunk, selected_points)

        min_dist, _ = torch.min(cdist, dim=-1)
        freeze_mask = min_dist > thres
        freeze_mask = freeze_mask.type(torch.int)
        freeze_mask_list.append(freeze_mask)

    freeze_mask = torch.cat(freeze_mask_list, dim=0)

    # 1 for points that are far away, 0 for points that are close
    return freeze_mask


    device = freeze_pts.device

    grid_pts_cnt = torch.zeros(
        (grid_size, grid_size, grid_size), dtype=torch.int32, device=device
    )

    dx = grid_lim / grid_size
    inv_dx = 1.0 / dx

    freeze_pts = (freeze_pts * inv_dx).long()

    for x, y, z in freeze_pts:
        grid_pts_cnt[x, y, z] += 1

    freeze_grid_mask = grid_pts_cnt >= 1

    freeze_grid_mask_int = freeze_grid_mask.type(torch.int32)

    number_freeze_grid = freeze_grid_mask_int.sum().item()
    print("number of freeze grid", number_freeze_grid)

    mpm_solver.enforce_grid_velocity_by_mask(freeze_grid_mask_int)

    # add debug section:

    return freeze_grid_mask