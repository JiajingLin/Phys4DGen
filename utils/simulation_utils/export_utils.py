import torch
from utils.general_utils import strip_symmetric, build_scaling_rotation


def build_quaternion_from_rotation(R):
    assert R.shape[-2:] == (3, 3)

    
    n = R.shape[0]
    q = torch.zeros((n, 4), dtype=R.dtype, device=R.device)

    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    
    positive_trace_indices = trace > 0
    s1 = torch.sqrt(trace[positive_trace_indices] + 1.0) * 2
    q[positive_trace_indices, 0] = 0.25 * s1
    q[positive_trace_indices, 1] = (R[positive_trace_indices, 2, 1] - R[positive_trace_indices, 1, 2]) / s1
    q[positive_trace_indices, 2] = (R[positive_trace_indices, 0, 2] - R[positive_trace_indices, 2, 0]) / s1
    q[positive_trace_indices, 3] = (R[positive_trace_indices, 1, 0] - R[positive_trace_indices, 0, 1]) / s1

    
    negative_trace_indices = ~positive_trace_indices

    
    max_diag_indices = torch.argmax(R[negative_trace_indices][:, [0, 1, 2], [0, 1, 2]], dim=1)

    
    condition_1 = max_diag_indices == 0
    
    indices_1 = negative_trace_indices.nonzero(as_tuple=True)[0][condition_1]
    s2 = torch.sqrt(1.0 + R[indices_1, 0, 0] - R[indices_1, 1, 1] - R[indices_1, 2, 2]) * 2
    q[indices_1, 0] = 0.25 * s2
    q[indices_1, 1] = (R[indices_1, 0, 1] + R[indices_1, 1, 0]) / s2
    q[indices_1, 2] = (R[indices_1, 0, 2] + R[indices_1, 2, 0]) / s2
    q[indices_1, 3] = (R[indices_1, 2, 1] - R[indices_1, 1, 2]) / s2

    
    condition_2 = max_diag_indices == 1
    indices_2 = negative_trace_indices.nonzero(as_tuple=True)[0][condition_2]
    s3 = torch.sqrt(1.0 + R[indices_2, 1, 1] - R[indices_2, 0, 0] - R[indices_2, 2, 2]) * 2
    q[indices_2, 0] = (R[indices_2, 0, 1] + R[indices_2, 1, 0]) / s3
    q[indices_2, 1] = 0.25 * s3
    q[indices_2, 2] = (R[indices_2, 1, 2] + R[indices_2, 2, 1]) / s3
    q[indices_2, 3] = (R[indices_2, 0, 2] - R[indices_2, 2, 0]) / s3

    
    condition_3 = max_diag_indices == 2
    indices_3 = negative_trace_indices.nonzero(as_tuple=True)[0][condition_3]
    s4 = torch.sqrt(1.0 + R[indices_3, 2, 2] - R[indices_3, 0, 0] - R[indices_3, 1, 1]) * 2
    q[indices_3, 0] = (R[indices_3, 0, 2] + R[indices_3, 2, 0]) / s4
    q[indices_3, 1] = (R[indices_3, 1, 2] + R[indices_3, 2, 1]) / s4
    q[indices_3, 2] = 0.25 * s4
    q[indices_3, 3] = (R[indices_3, 1, 0] - R[indices_3, 0, 1]) / s4

    return q



def build_rotation_scaling_from_covariance(covariance):    
    covariance_matrix = torch.zeros((covariance.shape[0], 3, 3), device="cuda")
    covariance_matrix[:, 0, 0] = covariance[:, 0]
    covariance_matrix[:, 0, 1] = covariance[:, 1]
    covariance_matrix[:, 0, 2] = covariance[:, 2]
    covariance_matrix[:, 1, 1] = covariance[:, 3]
    covariance_matrix[:, 1, 2] = covariance[:, 4]
    covariance_matrix[:, 2, 2] = covariance[:, 5]
    covariance_matrix[:, 1, 0] = covariance[:, 1]
    covariance_matrix[:, 2, 0] = covariance[:, 2]
    covariance_matrix[:, 2, 1] = covariance[:, 4]

    
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)    
    
    
    rotation_matrix = eigenvectors
    
    singularvalues = torch.sqrt(eigenvalues)    
    
    
    
    
    # covariation = strip_symmetric(rotation_matrix @ scaling_matrix @ (rotation_matrix @ scaling_matrix).transpose(1, 2))
    # flag = torch.all(covariation == covariance)


    # rotation_matrix_test = rotation_matrix[6].unsqueeze(0)
    
    rotation = build_quaternion_from_rotation(rotation_matrix) 
    # rotation = torch.zeros((rotation_matrix.shape[0], 4), dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    # rotation_4_1 = get_quaternion(rotation_matrix_test[0].detach().cpu().numpy())

    # rotation_4 = rotation_4 / torch.norm(rotation_4, dim=-1, keepdim=True)
    
    
    
    
    # rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)

    # rotation = rotation_4
    
    scaling = singularvalues

    
    
    
    # torch.cuda.synchronize()
    
    # torch.cuda.empty_cache()
    
    return rotation, scaling


def build_covariance_from_scaling_rotation(scaling, rotation): # scaling_modifier 1.0
    L = build_scaling_rotation(scaling, rotation)    # shape = (n, 3, 3) L = RS
    actual_covariance = L @ L.transpose(1, 2)   
    symm = strip_symmetric(actual_covariance)   
    return symm

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])    

    q = r / norm[:, None]   

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0] 
    x = q[:, 1] 
    y = q[:, 2] 
    z = q[:, 3] 
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R