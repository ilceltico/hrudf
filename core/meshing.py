
from core.utils import get_query_points, df_and_grad_to_input_cells
import core.utils as utils
import sys
sys.path.append("custom_mc")
from _marching_cubes_lewiner import pseudosdf_mc_lewiner
#Import DualMesh-UDF if present, otherwise skip
try:
    from DualMeshUDF.extract_mesh import extract_mesh_mod, extract_mesh
except:
    print("DualMesh-UDF not found. Skipping import.")
    pass
import time

import torch
import numpy as np
import trimesh
from tqdm import tqdm

def compute_pseudo_sdf(model, udf_and_grad_f, n_grid_samples=128, batch_size_udf=10000, batch_size_pseudosdf=10000, num_recursions=5, recursion_type="sigmoid", only_iterate_uncertain=1.0, clamp_distance=0.1, export_recursions="last", verbose=True):
    """
    Computes the pseudo-sdf of a mesh using a neural network model and a UDF and gradient function.
    It returns a list of pseudo-sdfs, one for each recursion, in order.
    
    Args:
        model: The neural network model.
        udf_and_grad_f: A function that takes query points and returns the UDF and gradients.
        n_grid_samples: The number of grid samples in each dimension.
        batch_size_udf: The batch size to use when computing the UDF and gradients. You can play with this if you run of out memory.
        batch_size_pseudosdf: The batch size to use when computing the pseudo-sdf. You can play with this if you run of out memory.
        num_recursions: The number of recursions to run the pseudo-sdf computation for.
        recursion_type: The type of recursion to use. Options are "direct", "sigmoid" and "softmax".
        only_iterate_uncertain: Only iterate over voxels with softmax scores below this threshold. A lower threshold will result in a faster computation, but may miss some voxels that need to be iterated over. Set it to 0.0 to iterate over all voxels.
        clamp_distance: The distance used to clamp the UDF values during the training of the neural UDF. Used to speed up the computation of the pseudo-sdf without affecting the final result. Set it to None to run the pseudo-sdf computation on all cells.
        export_recursions: Whether to export the pseudo-sdf after each recursion. Options are "all" and "last".
    """
    if export_recursions == "all":
        export_recursions = list(range(num_recursions+1))
    elif export_recursions == "last":
        export_recursions = [num_recursions]
    
    device = next(model.parameters()).device
    POWERS_OF_2 = utils.POWERS_OF_2.to(device)

    bbox = [(-1., -1., -1.), (1., 1., 1.)]
    query_points = get_query_points(bbox, n_grid_samples).view(-1,3).to(device)

    if verbose:
        print(f"Extracting UDF and gradients with batch size {batch_size_udf}")
    start = time.time()
    # udf, grads = udf_and_grad_f(query_points)
    extracted_udf = torch.zeros((query_points.shape[0]))
    extracted_grads = torch.zeros((query_points.shape[0], 3))
    iterator = range(0, query_points.shape[0], batch_size_udf)
    if verbose:
        iterator = tqdm(iterator, desc="Extracting UDF and gradients", unit="batch")
    for i in iterator:
        extracted_udf[i:i+batch_size_udf], extracted_grads[i:i+batch_size_udf] = udf_and_grad_f(query_points[i:i+batch_size_udf])
    if verbose:
        print(f"Done in: {time.time() - start} seconds")

    if verbose:
        print("Computing pseudo-SDF...")
    start = time.time()

    extracted_udf = extracted_udf.view(n_grid_samples,n_grid_samples,n_grid_samples)
    extracted_grads = extracted_grads.view(n_grid_samples,n_grid_samples,n_grid_samples,3)
    
    shape = (n_grid_samples - 1, n_grid_samples - 1, n_grid_samples - 1)

    voxel_size = 2.0 / (n_grid_samples - 1)
    # Limits the meshing distance using the cell's average and maximum distances. 
    # This speeds up the computation but can also result in many missing cells at high resolutions,
    # as we argue in the paper. Thus we do not use this filtering.
    max_avg_distance = clamp_distance
    max_max_distance = 2*clamp_distance
    udf_cells, grad_cells, indices = df_and_grad_to_input_cells(extracted_udf, extracted_grads, max_avg_distance, max_max_distance)
    indices = indices.long()

    result = []

    with torch.no_grad():
        # Build the input by putting the cells together
        input = torch.zeros((len(indices), 32))
        input[:,:8] = udf_cells[:,indices].T
        input[:,8:] = grad_cells[:,indices,:].permute(1,0,2).reshape(-1,24)
        udf = input[:,:8].clone()
        # Normalize the UDF values with the voxel size. This is key for the network to work at multiple resolutions.
        input[:,:8] = input[:,:8] / voxel_size

        shape_padded = (shape[0]+2, shape[1]+2, shape[2]+2)
        pseudo_sdf = torch.zeros(tuple(shape) + (8,))
        output_grid = torch.zeros(tuple(shape_padded) + (128,))

        for recursion_index in range(num_recursions+1):
            if verbose:
                print(f"Recursion {recursion_index}/{num_recursions}...")
            # Slower code but more memory efficient
            if recursion_index != 0:
                # Unravel indices and add 1 to each dimension to account for the padding
                unraveled_indices = [x+1 for x in np.unravel_index(indices, shape)]

                if only_iterate_uncertain > 0.0:
                    softmaxed_output_max = torch.nn.functional.softmax(output, dim=1).max(axis=1).values.reshape(-1,1)
                    uncertain_indices = softmaxed_output_max < only_iterate_uncertain
                    uncertain_indices = uncertain_indices.reshape(-1)
                    indices = indices[uncertain_indices]
                    input = input[uncertain_indices]
                    udf = udf[uncertain_indices]

                if recursion_type == "direct":
                    # Use the output as-is
                    pass
                elif recursion_type == "softmax":
                    # Use the output after softmax
                    output = torch.nn.functional.softmax(output, dim=1)
                elif recursion_type == "sigmoid":
                    # Use the output after sigmoid
                    output = torch.sigmoid(output)

                output_grid[unraveled_indices] = output

            output = torch.zeros((input.shape[0], 128))
            batch_head = 0
            iterator = range(0, input.shape[0], batch_size_pseudosdf)
            if verbose:
                iterator = tqdm(iterator, desc="Computing pseudo-SDF", unit="batch")
            for batch_head in iterator:
                batch_tail = min(batch_head + batch_size_pseudosdf, input.shape[0])
                input_batch = input[batch_head:batch_tail].to(device)
                if recursion_index == 0:
                    if model.layers[0].weight.shape[1] >= 7*128:
                        #Stack a zero tensor to the input for the first network pass
                        input_extra = torch.zeros((input_batch.shape[0], 128+6*128)).to(device)
                        input_batch = torch.cat((input_batch, input_extra), axis=1)

                else:
                    #Recurrence on previous output
                    unraveled_indices_batch = [x+1 for x in np.unravel_index(indices[batch_head:batch_tail], shape)]
                    prev_output_batch = output_grid[unraveled_indices_batch].to(device)
                    input_batch = torch.cat((input_batch, prev_output_batch), axis=1)

                    #Recurrence on previous output from neighbors in the grid
                    neighbors_recurrence_starting_index = input_batch.shape[1]
                    input_batch = torch.cat((input_batch, torch.zeros((input_batch.shape[0], 6*128)).to(device)), axis=1)

                    #Determine unraveled indices of neighbors and add them to the input
                    neighbors_indices = unraveled_indices_batch.copy()
                    neighbors_indices[0] -= 1
                    input_batch[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices].to(device)
                    neighbors_recurrence_starting_index += 128
                    neighbors_indices = unraveled_indices_batch.copy()
                    neighbors_indices[0] += 1
                    input_batch[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices].to(device)
                    neighbors_recurrence_starting_index += 128
                    neighbors_indices = unraveled_indices_batch.copy()
                    neighbors_indices[1] -= 1
                    input_batch[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices].to(device)
                    neighbors_recurrence_starting_index += 128
                    neighbors_indices = unraveled_indices_batch.copy()
                    neighbors_indices[1] += 1
                    input_batch[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices].to(device)
                    neighbors_recurrence_starting_index += 128
                    neighbors_indices = unraveled_indices_batch.copy()
                    neighbors_indices[2] -= 1
                    input_batch[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices].to(device)
                    neighbors_recurrence_starting_index += 128
                    neighbors_indices = unraveled_indices_batch.copy()
                    neighbors_indices[2] += 1
                    input_batch[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices].to(device)
            
                output_batch = model(input_batch)
                output[batch_head:batch_tail] = output_batch.to("cpu")

                pseudo_signs_batch = output_batch.argmax(axis=1).unsqueeze(-1).bitwise_and(POWERS_OF_2).ne(0).int()
                pseudo_signs_batch[pseudo_signs_batch == 0] = -1
                # Multiply the pseudo-signs with the udf values to get a pseudo-sdf

                pseudo_sdf[np.unravel_index(indices[batch_head:batch_tail], shape)] = torch.hstack((udf[batch_head:batch_tail,0].reshape(-1,1), udf[batch_head:batch_tail,1:8] * pseudo_signs_batch.cpu()))

            if recursion_index in export_recursions:
                result.append(pseudo_sdf.detach().numpy().copy())
    if verbose:
        print(f"Done in: {time.time() - start} seconds")

    return result



def mesh_marching_cubes(pseudo_sdf):
    """Extracts a mesh from a pseudo-SDF using the marching cubes algorithm."""

    resolution = pseudo_sdf.shape[0] + 1
    voxel_size = 2.0 / (resolution - 1)
    try:
        vertices, faces, normals, values = pseudosdf_mc_lewiner(pseudo_sdf, spacing=[voxel_size] * 3)
    except:
        print(f"Failed to mesh")
        return None
    vertices = vertices - 1 # Since voxel origin is [-1,-1,-1]
    mesh = trimesh.Trimesh(vertices, faces)

    return mesh



def mesh_dual_mesh_udf(pseudo_sdf, udf_f_dmudf, udf_grad_f_dmudf, batch_size=10000, device="cpu"):
    depth = int(np.ceil(np.log2(pseudo_sdf.shape[0])))
    grid_points = pseudo_sdf.shape[0] + 1
    if (np.log2(pseudo_sdf.shape[0]) != depth):
        raise ValueError("The pseudo-SDF must have a resolution that is a power of 2 in order to mesh it with DualMesh-UDF. This amounts to a grid resolution of 2^depth + 1. Try 129 or 257.")

    # Extract the mesh using the normal DualMesh-UDF, untuned.
    mesh_v_orig, mesh_f_orig, _, _, _ = extract_mesh(udf_f_dmudf, udf_grad_f_dmudf, batch_size=batch_size, max_depth=depth)

    # Now we extract the mesh using a "relaxed" version of DualMesh-UDF, to make sure we include as many faces as possible.
    mesh_v, mesh_f, _, _, _ = extract_mesh_mod(udf_f_dmudf, udf_grad_f_dmudf, batch_size=batch_size, max_depth=depth)


    torch_mesh_v = torch.Tensor(mesh_v).to(device)
    #Get which cell each vertex belongs to
    #TODO, for simplicity I'm not filtering vertices outside of the grid
    #In fact, the spurious vertices we've added to the mesh cannot go out the grid because they are taken as cell centers, and those are the ones that the network should filter anyway
    cell_indices = torch.floor(torch_mesh_v / (2.0 / (grid_points - 1)) + (grid_points-1)/2).int().to('cpu')

    #To do so, set out of bounds cell_indices as grid_points,grid_points,grid_points
    cell_indices[cell_indices < 0] = grid_points-1
    cell_indices[cell_indices >= grid_points-1] = grid_points-1

    # Define additional cells for the pseudo-SDF, out of the normal bounds, with additional 0 values for grid_points,grid_points,grid_points
    # So that these vertices are not filtered out
    current_pseudo_sdf = torch.zeros((grid_points,grid_points,grid_points,8))
    current_pseudo_sdf[:-1,:-1,:-1] = torch.Tensor(pseudo_sdf).to(device)

    #Get the pseudo-SDF for each cell index
    cell_pseudo_sdf = current_pseudo_sdf[tuple(cell_indices.T)].unsqueeze(1).reshape(-1,8)

    #Filter out faces that contain a vertex whose cell has a pseuso-SDF with no negative values (i.e. there are no predicted sign flips)
    filtered_mesh_f_neural = mesh_f[torch.all(torch.any(cell_pseudo_sdf[mesh_f] <= 0, axis=2), axis=1)]
    
    # Merge the original and the neural meshes in trimesh
    full_mesh_v = np.vstack((mesh_v, mesh_v_orig))
    full_mesh_f = np.vstack((filtered_mesh_f_neural, mesh_f_orig + mesh_v.shape[0]))

    pseudosdf_dmudf_mesh = trimesh.Trimesh(full_mesh_v, full_mesh_f)
    #Remove duplicated faces
    pseudosdf_dmudf_mesh.remove_duplicate_faces()

    #Retrieve also the original DualMesh-UDF mesh
    dmudf_mesh = trimesh.Trimesh(mesh_v_orig, mesh_f_orig)

    return dmudf_mesh, pseudosdf_dmudf_mesh
