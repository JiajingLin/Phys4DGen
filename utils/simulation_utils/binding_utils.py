import torch
import warp as wp
from scipy.stats import chi2
from tqdm.autonotebook import trange

import sys
from core.gaussian_model import GaussianModel

wp.init()
wp.config.verify_cuda = True

def compute_bindings_xyz(
    p_curr: torch.Tensor,
    p_prev: torch.Tensor,
    k_prev: torch.Tensor,
    bindings: torch.Tensor,
):
    """Compute updated location of gaussian kernels.

    Args:
        p_curr: Current particles xyz.
        p_prev: Previous particles xyz.
        k_prev: Previous kernels xyz.
        bindings: Binding matrix.
    
    Returns:
        k_curr: Updated kernels xyz.
    """
    delta_x = p_curr - p_prev.detach()

    # calculate means3D
    delta_means3D = torch.sparse.mm(bindings, delta_x)
    delta_means3D = delta_means3D.to_dense()
    k_curr = k_prev.detach() + delta_means3D

    return k_curr


def compute_bindings_F(
    deform_grad: torch.Tensor,
    bindings: torch.Tensor,
):
    """Compute updated deformation gradiant for each gaussian kernel.

    Args:
        deform_grad: Deformation gradient of each particle.
        bindings: Binding matrix.
    
    Returns:
        tensor_F: Deformation gradiant for each gaussian kernel.
    """

    # calculate deformation gradient
    tensor_F = torch.reshape(deform_grad, (-1, 9))
    tensor_F = torch.sparse.mm(bindings, tensor_F)
    tensor_F = tensor_F.to_dense()

    # reshape to (kernels, 3, 3)
    # tensor_F = torch.reshape(tensor_F, (-1, 3, 3))
    return tensor_F


def test_point_in_gaussians(point, mean, cov, confidence):
    """
    Test if a point is inside a Gaussian distribution.
    """
    point = torch.from_numpy(point)
    mean = torch.from_numpy(mean)
    cov = torch.from_numpy(cov)

    d = point - mean
    p = d @ torch.linalg.inv(cov) @ d

    return p <= confidence, p


@wp.kernel
def compute_inv_cov(
    cov: wp.array(dtype=float),
    out: wp.array(dtype=float)
):
    tid = wp.tid()

    cov_wp = wp.mat33(
        cov[tid * 6], cov[tid * 6 + 1], cov[tid * 6 + 2],
        cov[tid * 6 + 1], cov[tid * 6 + 3], cov[tid * 6 + 4],
        cov[tid * 6 + 2], cov[tid * 6 + 4], cov[tid * 6 + 5]
    )

    inv_cov_wp = wp.inverse(cov_wp)

    out[tid * 9 + 0] = inv_cov_wp[0, 0]
    out[tid * 9 + 1] = inv_cov_wp[0, 1]
    out[tid * 9 + 2] = inv_cov_wp[0, 2]
    out[tid * 9 + 3] = inv_cov_wp[1, 0]
    out[tid * 9 + 4] = inv_cov_wp[1, 1]
    out[tid * 9 + 5] = inv_cov_wp[1, 2]
    out[tid * 9 + 6] = inv_cov_wp[2, 0]
    out[tid * 9 + 7] = inv_cov_wp[2, 1]
    out[tid * 9 + 8] = inv_cov_wp[2, 2]


@wp.kernel
def test_point_in_gaussians_warp(
    point: wp.array(dtype=wp.vec3),
    mean: wp.array(dtype=float),
    cov: wp.array(dtype=float),
    threshold: wp.float32,
    out: wp.array(dtype=wp.int8),
    out_p: wp.array(dtype=float)
):
    tid = wp.tid()

    mean_wp = wp.vec3(mean[0], mean[1], mean[2])
    cov_wp = wp.mat33(
        cov[0], cov[1], cov[2],
        cov[1], cov[3], cov[4],
        cov[2], cov[4], cov[5]
    )

    d = point[tid] - mean_wp

    inv_cov_wp = wp.inverse(cov_wp)

    # manually calculate p1 = d * inv(cov)
    p11 = d[0] * inv_cov_wp[0, 0] + d[1] * inv_cov_wp[1, 0] + d[2] * inv_cov_wp[2, 0]
    p12 = d[0] * inv_cov_wp[0, 1] + d[1] * inv_cov_wp[1, 1] + d[2] * inv_cov_wp[2, 1]
    p13 = d[0] * inv_cov_wp[0, 2] + d[1] * inv_cov_wp[1, 2] + d[2] * inv_cov_wp[2, 2]

    # mannualy calculate p = p1 * d
    p1 = wp.vec3(p11, p12, p13)
    p = wp.dot(p1, d)

    if p <= threshold:
        out[tid] = wp.int8(1)
    else:
        out[tid] = wp.int8(0)

    out_p[tid] = p


@wp.kernel
def test_point_in_gaussians_with_inv_cov_warp(
    point: wp.array(dtype=wp.vec3),
    mean: wp.array(dtype=float),
    inv_cov: wp.array(dtype=float),
    threshold: wp.float32,
    out: wp.array(dtype=wp.int8),
    out_p: wp.array(dtype=float)
):
    tid = wp.tid()

    mean_wp = wp.vec3(mean[0], mean[1], mean[2])
    inv_cov_wp = wp.mat33(
        inv_cov[0], inv_cov[1], inv_cov[2],
        inv_cov[3], inv_cov[4], inv_cov[5],
        inv_cov[6], inv_cov[7], inv_cov[8]
    )

    d = point[tid] - mean_wp

    # manually calculate p1 = d * inv(cov)
    p11 = d[0] * inv_cov_wp[0, 0] + d[1] * inv_cov_wp[1, 0] + d[2] * inv_cov_wp[2, 0]
    p12 = d[0] * inv_cov_wp[0, 1] + d[1] * inv_cov_wp[1, 1] + d[2] * inv_cov_wp[2, 1]
    p13 = d[0] * inv_cov_wp[0, 2] + d[1] * inv_cov_wp[1, 2] + d[2] * inv_cov_wp[2, 2]

    # mannualy calculate p = p1 * d
    p1 = wp.vec3(p11, p12, p13)
    p = wp.dot(p1, d)
    
    

    if p <= threshold:
        out[tid] = wp.int8(1) 
    else:
        out[tid] = wp.int8(0) 

    out_p[tid] = p 


def gaussian_binding(
    gaussians: GaussianModel,
    particles: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):
    print(f'Start binding gaussians with particles ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gaussians.get_xyz.shape} {gaussians.get_xyz.device}')
    print(f'  p: {particles.shape} {particles.device}')

    # initialize (kernels, particles)
    flag_mat = torch.zeros(
        (gaussians.get_xyz.shape[0], particles.shape[0]),
        dtype=torch.bool, device=particles.device
    )    
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gaussians.get_xyz
    covs = gaussians.get_covariance()
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gaussians.get_xyz.shape[0]):  
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)

        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp, 
                out_p_wp 
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool() 
        out_p = wp.to_torch(out_p_wp) 
        flag_new = torch.zeros_like(flag) 

        if flag.sum() > max_particles: 
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles] 
            global_idx = torch.arange(flag.shape[0], device=particles.device) 
            global_idx = global_idx[flag][idx] 

            flag_new[global_idx] = True 
        else:
            flag_new[flag] = True 

        flag_mat[k] = flag_new 

    return flag_mat

# FIXME: Covariance may be inf, thus we need to cull Gaussians whose covs too small
def gaussian_binding_with_clip_v1(
    gaussians: GaussianModel,
    particles: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):    
    print(f'Start binding gaussians with particles (clip v1) ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gaussians.get_xyz.shape} {gaussians.get_xyz.device}')
    print(f'  p: {particles.shape} {particles.device}')
    
    count = 0

    # initialize (kernels, particles)
    weight_mat = torch.zeros(
        (gaussians.get_xyz.shape[0], particles.shape[0]),
        dtype=torch.float32, device=particles.device
    )
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gaussians.get_xyz
    covs = gaussians.get_covariance()
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gaussians.get_xyz.shape[0]):
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)

        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp,
                out_p_wp
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool()
        out_p = wp.to_torch(out_p_wp)
        weight = torch.zeros_like(out_p)

        
        if flag.sum() > max_particles:
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles]
            global_idx = torch.arange(flag.shape[0], device=particles.device)
            global_idx = global_idx[flag][idx]

            temp_p = out_p[flag][idx].clone()
            temp_p = torch.ones_like(temp_p)  
            # compute softmax of temp_p
            temp_weight = torch.softmax(-temp_p, dim=-1)
            weight[global_idx] = temp_weight
        else:
            temp_p = out_p[flag].clone()
            temp_p = torch.ones_like(temp_p)
            # compute softmax of temp_p
            temp_weight = torch.softmax(-temp_p, dim=-1)
            weight[flag] = temp_weight

        if weight.sum() == 0:
            # print(f'k: {k}, weight: {weight.sum()} flag: {flag.sum()} out_p: {out_p.argmin()} {out_p.min()}')
            # print(f'cov: {covs[k]}')
            # print(f'inv_cov: {inv_cov}')
            
            idx = torch.argmin(out_p)
            weight[idx] = 1.0
            count += 1
            
        assert weight.sum() != 0

        weight_mat[k] = weight
        
    return weight_mat

# FIXME: Covariance may be inf, thus we need to cull Gaussians whose covs too small
def gaussian_binding_with_clip_v1_v1(
    gaussians: GaussianModel,
    particles: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):
    print(f'Start binding gaussians with particles (clip v1) ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gaussians.get_xyz.shape} {gaussians.get_xyz.device}')
    print(f'  p: {particles.shape} {particles.device}')
    
    count = 0
    
    num_gs = gaussians.get_xyz.shape[0]
    num_pts = particles.shape[0]
    

    # initialize (kernels, particles)
    weight_mat = torch.zeros(
        (num_gs, num_gs + num_pts),
        dtype=torch.float32, device=particles.device
    )
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gaussians.get_xyz
    covs = gaussians.get_covariance()
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gaussians.get_xyz.shape[0]):
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)
        
        
        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp,
                out_p_wp
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool()
        out_p = wp.to_torch(out_p_wp)
        
        weight = torch.zeros((num_gs + num_pts,), device=particles.device) 

        
        if flag.sum() > max_particles:
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles]
            global_idx = torch.arange(flag.shape[0], device=particles.device)
            global_idx = global_idx[flag][idx]

            temp_p = out_p[flag][idx].clone()
            temp_p = torch.ones_like(temp_p)  
            # compute softmax of temp_p
            # temp_weight = torch.softmax(-temp_p, dim=-1)
            weight[global_idx + num_gs] = temp_p
        else:
            idx = torch.nonzero(flag).flatten() 
            
            temp_p = out_p[flag].clone()
            temp_p = torch.ones_like(temp_p)
            # compute softmax of temp_p
            # temp_weight = torch.softmax(-temp_p, dim=-1)
            weight[idx + num_gs] = temp_p

        # if weight.sum() == 0:
        #     # print(f'k: {k}, weight: {weight.sum()} flag: {flag.sum()} out_p: {out_p.argmin()} {out_p.min()}')
        #     # print(f'cov: {covs[k]}')
        #     # print(f'inv_cov: {inv_cov}')
        
        #     idx = torch.argmin(out_p)
        #     weight[idx] = 1.0
        #     count += 1
        
        
        weight[k] = 1.0
        
        weight = weight / weight.sum()
        
        assert weight.sum() != 0

        weight_mat[k] = weight

    return weight_mat




def save_sparse_chunks(dense_tensor, save_path, chunk_size=1000):
    shape = dense_tensor.shape
    sparse_chunks = []
    
    
    for i in range(0, shape[0], chunk_size):
        chunk = dense_tensor[i:i+chunk_size]
        sparse_chunk = chunk.to_sparse_coo()
        sparse_chunks.append(sparse_chunk)
    
    
    save_info = {
        'original_shape': shape,
        'chunk_size': chunk_size,
        'sparse_chunks': sparse_chunks
    }
    torch.save(save_info, save_path)


def load_sparse_chunks(save_path):
    
    save_info = torch.load(save_path)
    sparse_chunks = save_info['sparse_chunks']
    
    
    full_sparse = torch.cat(sparse_chunks, dim=0)
    return full_sparse



# FIXME: Covariance may be inf, thus we need to cull Gaussians whose covs too small
def gaussian_binding_with_surface_define(
    gaussians: GaussianModel,
    particles: torch.Tensor,
    is_surface: torch.Tensor,
    grid_cell_size: float,
    xyz1: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):
    print(f'Start binding gaussians with particles (clip v1) ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gaussians.get_xyz.shape} {gaussians.get_xyz.device}')
    print(f'  p: {particles.shape} {particles.device}')
    
    count = 0
    
    num_gs = gaussians.get_xyz.shape[0]
    num_pts = particles.shape[0]
    

    # initialize (kernels, particles)
    weight_mat = torch.zeros(
        (num_gs, num_gs + num_pts),
        dtype=torch.float32, device=particles.device
    )
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gaussians.get_xyz
    covs = gaussians.get_covariance()
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gaussians.get_xyz.shape[0]):
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)
        
        
        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp,
                out_p_wp
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool()
        out_p = wp.to_torch(out_p_wp)
        
        weight = torch.zeros((num_gs + num_pts,), device=particles.device) 

        
        if flag.sum() > max_particles:
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles]
            global_idx = torch.arange(flag.shape[0], device=particles.device)
            global_idx = global_idx[flag][idx]

            temp_p = out_p[flag][idx].clone()
            temp_p = torch.ones_like(temp_p)  
            weight[global_idx + num_gs] = temp_p
        else:
            idx = torch.nonzero(flag).flatten() 
            
            temp_p = out_p[flag].clone()
            temp_p = torch.ones_like(temp_p)
            weight[idx + num_gs] = temp_p

        weight[k] = 1.0
        weight = weight / weight.sum()
        assert weight.sum() != 0
        weight_mat[k] = weight
        
        
        valid_pts = particles[idx]
        valid_cell = ((valid_pts - xyz1) / grid_cell_size).round().long()
        
        is_surface[valid_cell[:, 0], valid_cell[:, 1], valid_cell[:, 2]] = True    

    return weight_mat


# FIXME: Covariance may be inf, thus we need to cull Gaussians whose covs too small
def gaussian_binding_with_surface_define_noweightmat(
    gaussians: GaussianModel,
    particles: torch.Tensor,
    is_surface: torch.Tensor,
    grid_cell_size: float,
    xyz1: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):    
    print(f'Start binding gaussians with particles (clip v1) ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gaussians.get_xyz.shape} {gaussians.get_xyz.device}')
    print(f'  p: {particles.shape} {particles.device}')
    
    count = 0
    
    num_gs = gaussians.get_xyz.shape[0]
    num_pts = particles.shape[0]
    

    # initialize (kernels, particles)
    # weight_mat = torch.zeros(
    #     (num_gs, num_gs + num_pts),
    #     dtype=torch.float32, device=particles.device
    # )
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gaussians.get_xyz
    covs = gaussians.get_covariance()
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gaussians.get_xyz.shape[0]):
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)
        
        
        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp,
                out_p_wp
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool()
        out_p = wp.to_torch(out_p_wp)
        
        weight = torch.zeros((num_gs + num_pts,), device=particles.device) 

        
        if flag.sum() > max_particles:
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles]
            global_idx = torch.arange(flag.shape[0], device=particles.device)
            global_idx = global_idx[flag][idx]

            temp_p = out_p[flag][idx].clone()
            temp_p = torch.ones_like(temp_p)  
            weight[global_idx + num_gs] = temp_p
        else:
            idx = torch.nonzero(flag).flatten() 
            
            temp_p = out_p[flag].clone()
            temp_p = torch.ones_like(temp_p)
            weight[idx + num_gs] = temp_p
        
        weight[k] = 1.0
        
        weight = weight / weight.sum()
        
        assert weight.sum() != 0        
        valid_pts = particles[idx]
        
        
        valid_cell = ((valid_pts - xyz1) / grid_cell_size).round().long()
        
        is_surface[valid_cell[:, 0], valid_cell[:, 1], valid_cell[:, 2]] = True    

    # return weight_mat



# FIXME: Covariance may be inf, thus we need to cull Gaussians whose covs too small
def gaussian_rebinding(
    gs_means: torch.Tensor,
    gs_covs: torch.Tensor,
    particles: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):  
    print(f'Start binding gaussians with particles (clip v1) ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gs_means.shape} {gs_means.device}')
    print(f'  p: {particles.shape} {particles.device}')
    
    num_gs = gs_means.shape[0]
    num_pts = particles.shape[0]

    # initialize (kernels, particles)
    weight_mat = torch.zeros(
        (num_gs, num_gs + num_pts),
        dtype=torch.float32, device=particles.device
    )
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gs_means
    covs = gs_covs
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gs_means.shape[0]):
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)
        
        
        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp,
                out_p_wp
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool()
        out_p = wp.to_torch(out_p_wp)
        
        weight = torch.zeros((num_gs + num_pts,), device=particles.device) 

        
        if flag.sum() > max_particles:
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles]
            global_idx = torch.arange(flag.shape[0], device=particles.device)
            global_idx = global_idx[flag][idx]

            temp_p = out_p[flag][idx].clone()
            temp_p = torch.ones_like(temp_p)  
            weight[global_idx + num_gs] = temp_p
        else:
            idx = torch.nonzero(flag).flatten() 
            
            temp_p = out_p[flag].clone()
            temp_p = torch.ones_like(temp_p)
            weight[idx + num_gs] = temp_p

        
        
        weight[k] = 1.0
        
        weight = weight / weight.sum()
        
        assert weight.sum() != 0

        weight_mat[k] = weight
    
    return weight_mat