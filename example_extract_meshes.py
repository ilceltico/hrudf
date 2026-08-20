import os
from pyexpat import features
import time

import trimesh
import igl
import numpy as np
import torch
from trimesh.transformations import scale_matrix
import argparse
from pathlib import Path

import core.utils as utils
from core.meshing import compute_pseudo_sdf, mesh_marching_cubes, mesh_dual_mesh_udf

from DeepSDF.model import get_model, get_latents
import torch.nn.functional as F

deepsdf_dirs = {
    "cars": "example_neural_UDFs/shapenet_cars_normalized",
    "chairs": "example_neural_UDFs/shapenet_chairs_normalized",
    "planes": "example_neural_UDFs/shapenet_planes_normalized",
}


def main():
    #Parse arguments from command line
    parser = argparse.ArgumentParser(description='Extract a mesh from a UDF.')
    parser.add_argument('--resolution', type=int, default=256, help='Number of grid samples per axies used to extract the mesh.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for computing the UDF and gradients.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for computing the UDF and gradients.')
    parser.add_argument('--model_dir', type=str, default="weights", help='Path to the model directory.')
    parser.add_argument('--meshing_algo', type=str, default="marching_cubes", help='Meshing algorithm to use. Options are "marching_cubes" and "dual_mesh_udf".')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory where extracted meshes are saved.')
    parser.add_argument('--dataset', type=str, default="planes", help='Dataset to use for extraction (cars, chairs, planes).')
    
    args = parser.parse_args()
    # pretty print the arguments
    print(args)

    output_dir = f"{args.output_dir}/{args.dataset}/{args.meshing_algo}/{args.resolution}"
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # If CUDA is not available, use CPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        args.device = "cpu"

    # Load the model
    model, config = utils.load_model(args.model_dir, args.device)

    # NOTE: The model is trained on normalized meshes in [-1,1]^3, so it assumes the input UDF is predicted on a normalized mesh. It should work anyway, but normalized meshes are recommended.

    # Load the neural UDF model
    deepsdf_model, deepsdf_latents = load_deepsdf(deepsdf_dirs[args.dataset])
    clamp_distance = 0.1 # Set it to the value you used during the training of the neural UDF.

    # For every latent, extract the pseudo-SDF and the mesh. 
    for latent_index in range(deepsdf_latents.num_embeddings):
        print(f"Meshing {latent_index+1} of {deepsdf_latents.num_embeddings}...")
        latent_vec = deepsdf_latents(torch.LongTensor([latent_index]).cuda()).detach()
        mesh_name = f"{latent_index}"

        # Pseudo-SDF computation, returned as a list of numpy arrays, one for each recursion. The last one is the final result.
        # export_recursions controls which recursions are exported, by default only the last one is exported. 
        # In this example, we export all recursions, so we can see the evolution of the meshes.
        # We iterate for 5 recursions, which we found good for most neural UDFs. If you have an already good UDF, you can set it to a lower number.
        pseudo_sdfs = compute_pseudo_sdf(
            model, 
            recursion_type=config['recursion_type'], 
            num_recursions=5, 
            udf_and_grad_f=lambda query_points: udf_and_grad_f_deepsdf(deepsdf_model, latent_vec, clamp_distance, query_points), 
            n_grid_samples=args.resolution, 
            batch_size_udf=args.batch_size, 
            batch_size_pseudosdf=args.batch_size, 
            export_recursions="all",
            clamp_distance=clamp_distance,
            )

        # We can now extract the mesh
        for recursion_index, pseudo_sdf in enumerate(pseudo_sdfs):
            current_recursion_dir = output_dir / f"recursion_{recursion_index}"
            os.makedirs(current_recursion_dir, exist_ok=True)
            if args.meshing_algo == "marching_cubes":
                print(f"Extracting mesh {mesh_name} at recursion {recursion_index} using Marching Cubes...", end=" ", flush=True)
                start = time.time()
                mesh = mesh_marching_cubes(pseudo_sdf)
                print(f"Done in: {time.time() - start} seconds")
                
                # You can de-normalize the mesh to the original size, if needed/possible. For example:
                # mesh = mesh.apply_transform(scale_matrix(max(gt_mesh_extents) / 1.99999))
                # mesh = mesh.apply_translation(gt_mesh_bounds)

                mesh.export(str(current_recursion_dir / f"{mesh_name}.ply"))

                # The mesh can be postprocessed to fill small cracks, holes, smooth the surface, remove degenerate faces, etc.
                # Results in the paper do not include any postprocessing.
                # Example of simple postprocessing:
                # print("Postprocessing mesh...")
                # mesh.fill_holes()
                # mesh.update_faces(mesh.unique_faces())
                # mesh.remove_unreferenced_vertices()
                # mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, validate=True)
                # mesh.export(str(current_recursion_dir / f"{mesh_name}_postprocessed.ply"))

            if args.meshing_algo == "dual_mesh_udf":
                # We can also extract the mesh using a modified version of DualMesh-UDF. We refer to the paper for more details.
                # In short: we relax the parameters of DualMesh-UDF to allow for more triangles in the mesh, and filter out unwanted ones using our Pseudo-SDF.
                # We also add triangles in cells where DualMesh-UDF fails but the Pseudo-SDF predicts a surface.
                print("Extracting mesh using DualMesh-UDF...")
                dmudf_mesh, pseudosdf_dmudf_mesh = mesh_dual_mesh_udf(pseudo_sdf, lambda query_points: udf_f_deepsdf_dmudf(deepsdf_model, latent_vec, query_points), lambda query_points: udf_grad_f_deepsdf_dmudf(deepsdf_model, latent_vec, query_points), batch_size=args.batch_size, device=args.device)

                # You can de-normalize the mesh to the original size, if needed/possible. For example:
                # pseudosdf_dmudf_mesh = pseudosdf_dmudf_mesh.apply_transform(scale_matrix(max(gt_mesh_extents) / 1.99999))
                # pseudosdf_dmudf_mesh = pseudosdf_dmudf_mesh.apply_translation(gt_mesh_bounds)
                # dmudf_mesh = dmudf_mesh.apply_transform(scale_matrix(max(gt_mesh_extents) / 1.99999))
                # dmudf_mesh = dmudf_mesh.apply_translation(gt_mesh_bounds)
                pseudosdf_dmudf_mesh.export(str(current_recursion_dir / f"{mesh_name}.ply"))
                # If you want to compare with the original DualMesh-UDF, you can export it as follows:
                # dmudf_mesh.export(str(current_recursion_dir / f"{mesh_name}_original_dmudf.ply"))



# Define a function that extracts UDF and gradients and returns them as Torch Tensors. 
# def udf_and_grad_f(query_points, mesh):
#     udf, facet_indices, closest_points = igl.point_mesh_squared_distance(query_points.cpu().detach().numpy(), mesh.vertices, mesh.faces) #This function computes the squared distance, so we need to take the square root
#     udf = np.sqrt(udf)
#     udf = torch.Tensor(udf)

#     # IMPORTANT: the gradients point away from the surface.
#     udf_grads = query_points - closest_points
#     udf_grads = torch.Tensor(udf_grads)
#     udf_grads_normalized = utils.normalize(udf_grads, dim=1)

#     # Some query points are exactly on the surface and can produce NaN gradients
#     # The UDF gradient does not exist on the surface, so here we set it to zero.
#     udf_grads_normalized = torch.nan_to_num(udf_grads_normalized, nan=0.0)

#     return udf, udf_grads_normalized


# Here is an example of the above function, but for neural UDFs (udf_autodecoder in this example)
# For speed purposes, batching is IMPORTANT.
def udf_and_grad_f_deepsdf(udf_autodecoder, latent, clamp_distance, query_points):
    udf_autodecoder.eval()

    # Prepare data
    xyz_all = query_points.view(-1, 3)
    xyz_all.requires_grad = False
    n_points = len(xyz_all)
    grad = torch.zeros(n_points,3)

    latent_rep = latent.expand(xyz_all.shape[0], -1)
    inputs = torch.cat([latent_rep, xyz_all], dim=-1)

    # Predict UDF
    udf = udf_autodecoder(inputs)
    udf = udf.squeeze(1).detach()

    # Compute gradients only when needed
    grad_mask = udf < clamp_distance
    norm_idx = torch.where(grad_mask)[0]
    xyz_all_for_grad = xyz_all[norm_idx].cuda()
    xyz_all_for_grad.requires_grad = True
    inputs = torch.cat([latent_rep[:len(xyz_all_for_grad)], xyz_all_for_grad], dim=-1)
    udf_for_grad = udf_autodecoder(inputs)
    udf_for_grad.sum().backward(retain_graph=True)
    angle = xyz_all_for_grad.grad.detach()
    # IMPORTANT: the gradients point AWAY from the surface.
    grad[norm_idx] = F.normalize(angle, dim=1).cpu()
        
    return udf, grad


# Utils
def load_deepsdf(dir):
    d = 12
    hd = 1024
    ld = 512
    outdim=1
    def modelinit():
        return get_model(
                "DeepSDF",
            hidden_dim = hd,
            n_layers = d,
            dropout = 0.0,
            weight_norm = True,
            activation = "relu",
            features = None,
            latent_dim=ld,
            out_dim =outdim).cuda()

    lat_vecs = get_latents(20, ld, 1.0).cuda()
    lat_vecs.load_state_dict(torch.load(f"{dir}/latent/latents_10000.pth"))
    decoder = modelinit().cuda()
    decoder.load_state_dict(torch.load(f"{dir}/model/model_10000.pth"))

    return decoder, lat_vecs


###################################
######## DUALMESH-UDF only ########
###################################

# The following functions are examples used in the DualMesh-UDF code.
# They are similar to the ones above, but they return the results in sligthly different formats.
# Implement your own functions for your UDFs.
# def udf_f_dmudf(query_points, mesh):
#     return np.sqrt(igl.point_mesh_squared_distance(query_points, mesh.vertices, mesh.faces)[0]).reshape(-1,1)

# def udf_grad_f_dmudf(query_points, mesh):
#     udf, facet_indices, closest_points = igl.point_mesh_squared_distance(query_points, mesh.vertices, mesh.faces)
#     udf = np.sqrt(udf)

#     udf_grads = query_points - closest_points
#     udf_grads = torch.Tensor(udf_grads)
#     # udf_grads_normalized = udf_grads / torch.linalg.norm(udf_grads, axis=1).reshape(-1,1)
#     udf_grads_normalized = utils.normalize(udf_grads, dim=1).reshape(-1,1)
#     return udf.reshape(-1,1), udf_grads_normalized.reshape(-1,3,1)

# Here is also an example of the above functions, but for neural UDFs.
def udf_f_deepsdf_dmudf(net, latent_vec, pts):
    net.eval()
    with torch.no_grad():
        device = next(net.parameters()).device
        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        xyz = torch.from_numpy(pts).to(device)
        
        batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, target_shape[0], 1)
        input = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), xyz.reshape(-1, xyz.shape[-1]).float()], dim=1)
        udf_p = net(input)
        
        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

    return udf_p

def udf_grad_f_deepsdf_dmudf(net, latent_vec, pts):
    net.eval()
    device = next(net.parameters()).device
    target_shape = list(pts.shape)

    pts = pts.reshape(-1, 3)
    pts = torch.from_numpy(pts).to(device)
    pts.requires_grad = True

    batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, target_shape[0], 1)
    input = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), pts.reshape(-1, pts.shape[-1]).float()], dim=1)
    udf_p = net(input)

    udf_p.sum().backward()
    grad_p = pts.grad.detach()
    grad_p = utils.normalize(grad_p)

    grad_p = grad_p.reshape(target_shape).detach().cpu().numpy()
    target_shape[-1] = 1
    udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

    return udf_p, grad_p



if __name__ == "__main__":
    main()