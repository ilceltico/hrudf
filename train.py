import gc
import json

import torch
import core.data as data
from torch.utils.data import DataLoader
import core.models as models
import time
import numpy as np
import trimesh
import os
import logging
import argparse
from tqdm import tqdm
import random

POWERS_OF_2 = data.POWERS_OF_2

def main():

    #Parse arguments from command line
    parser = argparse.ArgumentParser(description='Train a neural network to predict the sign configuration of a UDF voxel grid')
    parser.add_argument('--save_base_location', type=str, default="./experiments", help='Base save location for the experiment')
    parser.add_argument('--dataset_location', type=str, default="./datasets/ABC_precomputed_grid", help='Location of the training dataset')

    # Load an existing experiment. NOTE: All command line parameters will be ignored
    parser.add_argument('--load', type=str, default=None, help='Load an existing experiment from a directory. NOTE: Ignores all other command line parameters')

    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size. Has no effect on results, only on memory usage.')
    parser.add_argument('--hid_size', type=int, default=1024, help='Hidden size of the network')
    parser.add_argument('--num_hid_layers', type=int, default=2, help='Number of hidden layers of the network')
    parser.add_argument('--grid_points_list', type=int, nargs='+', default=[128], help='List of resolutions to train on. Default is 128 only.')
    parser.add_argument('--noise_udf', type=float, default=1.0, help='Gaussian noise to augment the distance field. Set to zero to disable.')
    parser.add_argument('--noise_udf_type', type=str, default="scale", help='add | scale | add_exp | scale_exp')
    parser.add_argument('--noise_grad', type=float, default=1.0, help='Gaussian noise to augment the gradients. Set to zero to disable.')
    parser.add_argument('--noise_grad_type', type=str, default="scale", help='add | scale | add_exp | scale_exp')
    parser.add_argument('--noise_grad_swap', type=float, default=0.0, help='Swap the gradients with a probability. Set to zero to disable.')
    parser.add_argument('--noise_model', type=str, default="cell-independent", help='cell-independent, cell-consistency')
    parser.add_argument('--renormalize_gradients', default=False, action="store_true", help='Normalizes noisy training gradients to unitary norm')
    parser.add_argument('--balanced', default=False, action="store_true", help='Rebalance the CE loss using the class weights')
    parser.add_argument('--num_recursions', type=int, default=5, help='Number of extra recursions, 0 for a non-recurrent network')
    parser.add_argument('--recursion_type', type=str, default="sigmoid", help='What to do with the network output before recursion. Options: direct, softmax, sigmoid')
    parser.add_argument('--randomize_rec', action="store_true", help='Randomize the number of recursions during training')
    parser.add_argument('--no-randomize_rec', dest='randomize_rec', action='store_false', help='Do not randomize the number of recursions during training')
    parser.set_defaults(randomize_rec=True)

    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cpu, mps, cuda)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloader. We recommend setting it to 0 when training with CPU.')
    parser.add_argument('--config_only', action="store_true", help='Only export the configuration file and exit.')
    
    args = parser.parse_args()

    if args.load is not None:
        # Load the configuration file
        with open(os.path.join(args.load, "config.json"), "r") as f:
            config = json.load(f)
        loading_dir = args.load
        args = argparse.Namespace(**config)
        args.load = loading_dir
        args.config_only = False

    device = torch.device(args.device)
    global POWERS_OF_2
    POWERS_OF_2 = POWERS_OF_2.to(device)

    epochs = args.epochs

    if args.load is not None:
        save_location = args.load
    else:
        today = time.strftime("%Y%m%d-%H%M%S")
        grid_points_list_string = "_".join([str(x) for x in args.grid_points_list])
        save_location = os.path.join(args.save_base_location, f"{today}_{grid_points_list_string}{'_normalized'}{f'_noiseudf{args.noise_udf_type}{args.noise_udf}' if args.noise_udf > 0 else ''}{f'_noisegrad{args.noise_grad_type}{args.noise_grad}' if args.noise_grad > 0 else ''}{'renormgrads' if args.renormalize_gradients else ''}{'_balanced' if args.balanced else ''}{f'_{args.epochs}ep'}{f'_{args.num_recursions}rec'}{f'_{args.recursion_type}' if args.recursion_type != 'direct' else ''}{f'_randrec' if args.randomize_rec else ''}")

        os.makedirs(save_location, exist_ok=True)

        os.makedirs(os.path.join(save_location, "model"), exist_ok=True)
        os.makedirs(os.path.join(save_location, "optimizer"), exist_ok=True)

        # Copy the training code to the save location for reproducibility
        os.makedirs(os.path.join(save_location, "code"), exist_ok=True)
        os.system(f"cp train.py {os.path.join(save_location, 'code')}")
        # Copy also the core
        os.system(f"cp -r core {os.path.join(save_location, 'code')}")

    # Setup the logger with the save directory
    logger = logging.getLogger()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_location, "log.txt")),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Arguments: {args}")

    input_dims = 32
    input_dims += 128 #Including the previously computed signs from the current voxel
    input_dims += 6*128 #Including the previously computed signs from neightboring voxels
    layer_sizes = [args.hid_size] * args.num_hid_layers
    layer_sizes.insert(0, input_dims)
    layer_sizes.append(128)
    model = models.MLP(layer_sizes, torch.nn.LeakyReLU)
    model = model.to(device)
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(
            [
                {
                    "params": model.parameters(),
                    "lr": 5e-4,
                }
            ]
        )
    
    # Export configuration file
    if args.load is None:
        config = vars(args)
        with open(os.path.join(save_location, "config.json"), "w") as f:
            json.dump(config, f)
        logger.info(f"Exported configuration to {os.path.join(save_location, 'config.json')}")
    if args.config_only:
        return

    
    if args.load is not None:
        # The provided path contains multiple model state dicts named "model_checkpoint_n.pt", load the one with the highest n
        model_path = os.path.join(args.load, "model")
        model_files = [f for f in os.listdir(model_path) if f.startswith("model_") and f.endswith(".pt")]
        model_files = sorted(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(model_files) == 0:
            logging.info("No model checkpoints found, starting from scratch")
            elapsed_epochs = 0
        else:
            latest_model = model_files[-1]
            model_epoch = int(latest_model.split("_")[-1].split(".")[0])
            # Same for the optimizer
            optimizer_path = os.path.join(args.load, "optimizer")
            optimizer_files = [f for f in os.listdir(optimizer_path) if f.startswith("optimizer_") and f.endswith(".pt")]
            optimizer_files = sorted(optimizer_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_optimizer = optimizer_files[-1]
            optimizer_epoch = int(latest_optimizer.split("_")[-1].split(".")[0])
            if model_epoch != optimizer_epoch:
                logging.warning("Latest model and optimizer checkpoints do not match, loading the latest matching pair")
                model_epoch = min(model_epoch, optimizer_epoch)
                optimizer_epoch = model_epoch
            try:
                model.load_state_dict(torch.load(os.path.join(model_path, latest_model)))
                optimizer.load_state_dict(torch.load(os.path.join(optimizer_path, latest_optimizer)))
            except Exception as e:
                logging.error(f"Failed to load model and optimizer: {e}")
                return
        
            logging.info(f"Loaded checkpoint at epoch {model_epoch}")
            elapsed_epochs = model_epoch+1
            logging.info(f"Continuing training from epoch {elapsed_epochs}, {epochs-elapsed_epochs} epochs remaining")
    else:
        elapsed_epochs = 0
        

    train(logger=logger, 
          model=model, 
          optimizer=optimizer, 
          epochs=epochs, 
          num_workers=args.num_workers,
          recursions=args.num_recursions, 
          grid_points_list=args.grid_points_list, 
          dataset_location_base=args.dataset_location, 
          renormalize_gradients=args.renormalize_gradients,
          balanced=args.balanced,
          noise_udf=args.noise_udf, 
          noise_udf_type=args.noise_udf_type, 
          noise_grad=args.noise_grad, 
          noise_grad_type=args.noise_grad_type, 
          noise_grad_swap=args.noise_grad_swap,
          recursion_type=args.recursion_type, 
          randomize_rec=args.randomize_rec, 
          save_location=save_location, 
          starting_epoch=elapsed_epochs, 
          noise_model=args.noise_model,
          device=device,
          )

    #Save the model and the optimizer
    torch.save(model.state_dict(), os.path.join(save_location, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_location, "optimizer.pt"))

    torch.cuda.empty_cache()
    gc.collect()

    # evaluate_lowmem(logger, 
    #                 model, 
    #                 [os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "64/train"), os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "64/val")], 
    #                 64, 
    #                 normalize_df, 
    #                 export_meshes, 
    #                 os.path.join(save_location,"exports/shapenet_hieu_nothresh/"), 
    #                 "shapenet_hieu_nothresh64", 
    #                 num_recursions, 
    #                 export_recursions, 
    #                 recursion_type = args.recursion_type, 
    #                 batch_size=10000, 
    #                 grid_preprocessed=False, 
    #                 drop_uncertain=args.drop_uncertain_testing, 
    #                 only_iterate_uncertain=args.only_iterate_uncertain_testing, 
    #                 architecture=args.architecture
    #                 )
    # evaluate_lowmem(logger, model, [os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "128/train"), os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "128/val")], 128, normalize_df, export_meshes, os.path.join(save_location,"exports/shapenet_hieu_nothresh/"), "shapenet_hieu_nothresh128", num_recursions, export_recursions, recursion_type = args.recursion_type, batch_size=10000, grid_preprocessed=False, drop_uncertain=args.drop_uncertain_testing, only_iterate_uncertain=args.only_iterate_uncertain_testing, architecture=args.architecture)
    # evaluate_lowmem(logger, model, [os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "256/train"), os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "256/val")], 256, normalize_df, export_meshes, os.path.join(save_location,"exports/shapenet_hieu_nothresh/"), "shapenet_hieu_nothresh256", num_recursions, export_recursions, recursion_type = args.recursion_type, batch_size=10000, grid_preprocessed=False, drop_uncertain=args.drop_uncertain_testing, only_iterate_uncertain=args.only_iterate_uncertain_testing, architecture=args.architecture)
    # evaluate_lowmem(logger, model, [os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "512/train"), os.path.join(dataset_location_shapenet_hieu_nothresh_no_grid_preprocessing, "512/val")], 512, normalize_df, export_meshes, os.path.join(save_location,"exports/shapenet_hieu_nothresh/"), "shapenet_hieu_nothresh512", num_recursions, export_recursions, recursion_type = args.recursion_type, batch_size=10000, grid_preprocessed=False, drop_uncertain=args.drop_uncertain_testing, only_iterate_uncertain=args.only_iterate_uncertain_testing, architecture=args.architecture)

    print(f"Done")



def train(logger, 
          model, 
          optimizer, 
          epochs, 
          num_workers,
          recursions, 
          grid_points_list, 
          dataset_location_base, 
          renormalize_gradients,
          balanced,
          noise_udf, 
          noise_udf_type, 
          noise_grad, 
          noise_grad_type, 
          noise_grad_swap,
          recursion_type, 
          randomize_rec=False, 
          save_location = "", 
          starting_epoch=0, 
          noise_model=None,
          device=torch.device("cpu"),
          ):

    dataloaders = []
    for grid_points in grid_points_list:
        voxel_size = 2.0 / (grid_points - 1)
        dataset_location = os.path.join(dataset_location_base, f"{grid_points}")
        training_dataset = data.PrecomputedGridDataset([os.path.join(dataset_location, "train")], 
                                                    noise_udf, 
                                                    noise_udf_type, 
                                                    noise_grad, 
                                                    noise_grad_type, 
                                                    noise_grad_swap=noise_grad_swap,
                                                    noise_model=noise_model, 
                                                    max_avg_distance=1.05*voxel_size, 
                                                    max_max_distance=1.74*voxel_size, 
                                                    class_balanced_weights=True, 
                                                    compute_gt=True, 
                                                    )
        dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        dataloaders.append(dataloader)


    for epoch in range(starting_epoch, epochs):

        epoch_loss = 0.0
        epoch_num_valid = 0
        epoch_total = 0
        epoch_non_empty_num_valid = 0
        epoch_non_empty_total = 0
        epoch_close_num_valid = 0
        epoch_close_total = 0

        start = time.time()

        for dataloader, grid_points in zip(dataloaders, grid_points_list):

            dataloader_loss = 0.0
            dataloader_num_valid = 0
            dataloader_total = 0
            dataloader_non_empty_num_valid = 0
            dataloader_non_empty_total = 0
            dataloader_close_num_valid = 0
            dataloader_close_total = 0

            voxel_size = 2.0 / (grid_points - 1)

            for input_dataset_full, gt_dataset_full, indices, class_weights, sdf_shape, i in tqdm(dataloader):

                model.train()
                optimizer.zero_grad()

                input_dataset_full = input_dataset_full.to(device)
                gt_dataset_full = gt_dataset_full.to(device)

                shape_loss = 0.0    
                shape_num_valid = 0
                shape_total = 0
                shape_non_empty_num_valid = 0
                shape_non_empty_total = 0
                shape_close_num_valid = 0
                shape_close_total = 0

                class_weights = class_weights.to(device)

                if balanced:
                    loss_fn1 = torch.nn.CrossEntropyLoss(weight=class_weights[0])
                    loss_fn2 = torch.nn.CrossEntropyLoss(weight=class_weights[0])
                else:
                    loss_fn1 = torch.nn.CrossEntropyLoss()
                    loss_fn2 = torch.nn.CrossEntropyLoss()

                for shape_i in range(1):
                    gt = gt_dataset_full[0]
                    input_dataset = input_dataset_full

                    
                    input_dataset[0][:,:8] = input_dataset[0][:,:8] / voxel_size

                    # Normalize gradients to unitary norm, which is the assumption of this method
                    if renormalize_gradients:
                        input_dataset[0][:,8:] = (input_dataset[0][:,8:].reshape(-1,3) / torch.linalg.norm(input_dataset[0][:,8:].reshape(-1,3), axis=1).reshape(-1,1)).reshape(-1,input_dataset[0].shape[1]-8)

                    input = input_dataset[0]
                    #Stack a zero tensor to the input for the first network pass
                    input_extra = torch.zeros((input.shape[0], 128+6*128)).to(device)

                    input = torch.cat((input, input_extra), axis=1)

                    output = model(input)

                    batch_loss = loss_fn1(output, gt)                    

                    sdf_shape_padded = (sdf_shape[0]+2, sdf_shape[1]+2, sdf_shape[2]+2)
                    randomrec = recursions if not randomize_rec else random.randint(0,recursions)
                    for _ in range(randomrec):
                        input = input_dataset[0]

                        output_grid = torch.zeros(tuple(sdf_shape_padded) + (128,)).to(device)
                        # Set the output_grid to have 1 on the last element of every grid cell
                        # output_grid[:,:,:,127] = 1.0
                        # Unravel indices and add 1 to each dimension to account for the padding
                        unraveled_indices = [x+1 for x in np.unravel_index(indices[0], sdf_shape)]

                        if recursion_type == "direct":
                            # Use the output as-is
                            pass
                        elif recursion_type == "softmax":
                            #Use the output after softmax
                            output = torch.nn.functional.softmax(output, dim=1)
                        elif recursion_type == "sigmoid":
                            #Use the output after sigmoid
                            output = torch.sigmoid(output)

                        output_grid[unraveled_indices] = output
                        
                        #Recurrence on previous output
                        input = torch.cat((input, output), axis=1)

                        #Recurrence on previous output from neighbors in the grid
                        neighbors_recurrence_starting_index = input.shape[1]
                        input = torch.cat((input, torch.zeros((input.shape[0], 6*128)).to(device)), axis=1)
                        #Determine unraveled indices of neighbors and add them to the input
                        neighbors_indices = unraveled_indices.copy()
                        neighbors_indices[0] -= 1
                        input[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices]
                        neighbors_recurrence_starting_index += 128
                        neighbors_indices = unraveled_indices.copy()
                        neighbors_indices[0] += 1
                        input[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices]
                        neighbors_recurrence_starting_index += 128
                        neighbors_indices = unraveled_indices.copy()
                        neighbors_indices[1] -= 1
                        input[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices]
                        neighbors_recurrence_starting_index += 128
                        neighbors_indices = unraveled_indices.copy()
                        neighbors_indices[1] += 1
                        input[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices]
                        neighbors_recurrence_starting_index += 128
                        neighbors_indices = unraveled_indices.copy()
                        neighbors_indices[2] -= 1
                        input[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices]
                        neighbors_recurrence_starting_index += 128
                        neighbors_indices = unraveled_indices.copy()
                        neighbors_indices[2] += 1
                        input[:,neighbors_recurrence_starting_index:neighbors_recurrence_starting_index+128] = output_grid[neighbors_indices]

                        output = model(input)

                        batch_loss += loss_fn2(output, gt)                
                    
                    pred_index = torch.argmax(output, axis=1)
                    gt_index = torch.argmax(gt, axis=1)
                    
                    num_valid = (pred_index == gt_index).sum()
                    total = pred_index.shape[0]

                    non_empty_num_valid = ((pred_index == gt_index)*(gt_index != 127)).sum()
                    non_empty_total = (gt_index != 127).sum()

                    udf = input[:,:8]

                    #Same conditions as MeshUDF
                    close_num_valid = ((pred_index == gt_index) * (udf.mean(axis=1) < 1.05) * (udf.max(axis=1).values <= 1.74)).sum()
                    close_total = ((udf.mean(axis=1) < 1.05) * (udf.max(axis=1).values <= 1.74)).sum()

                    batch_loss.backward()
                    shape_loss += batch_loss
                    shape_num_valid += num_valid
                    shape_total += total
                    shape_non_empty_num_valid += non_empty_num_valid
                    shape_non_empty_total += non_empty_total
                    shape_close_num_valid += close_num_valid
                    shape_close_total += close_total

                dataloader_loss += shape_loss
                dataloader_num_valid += shape_num_valid
                dataloader_total += shape_total
                dataloader_non_empty_num_valid += shape_non_empty_num_valid
                dataloader_non_empty_total += shape_non_empty_total
                dataloader_close_num_valid += shape_close_num_valid
                dataloader_close_total += shape_close_total

                optimizer.step()
            
            if len(dataloaders) > 1:
                logger.log(logging.INFO, f"Dataloader {grid_points}, loss = {dataloader_loss}, close valid {dataloader_close_num_valid}/{dataloader_close_total} {dataloader_close_num_valid/dataloader_close_total*100}%, valid {dataloader_num_valid}/{dataloader_total} {dataloader_num_valid/dataloader_total * 100}%, non empty valid {dataloader_non_empty_num_valid}/{dataloader_non_empty_total} {dataloader_non_empty_num_valid/dataloader_non_empty_total * 100}%, time {time.time() - start}")

            epoch_loss += dataloader_loss
            epoch_num_valid += dataloader_num_valid
            epoch_total += dataloader_total
            epoch_non_empty_num_valid += dataloader_non_empty_num_valid
            epoch_non_empty_total += dataloader_non_empty_total
            epoch_close_num_valid += dataloader_close_num_valid
            epoch_close_total += dataloader_close_total

        logger.log(logging.INFO, f"End of epoch {epoch}/{epochs}, loss = {epoch_loss}, close valid {epoch_close_num_valid}/{epoch_close_total} {epoch_close_num_valid/epoch_close_total*100}%, valid {epoch_num_valid}/{epoch_total} {epoch_num_valid/epoch_total * 100}%, non empty valid {epoch_non_empty_num_valid}/{epoch_non_empty_total} {epoch_non_empty_num_valid/epoch_non_empty_total * 100}%, time {time.time() - start}")
        # Save model and optimizer checkpoints
        torch.save(model.state_dict(), os.path.join(save_location, "model", f"model_{epoch}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_location, "optimizer", f"optimizer_{epoch}.pt"))



if __name__ == '__main__':
    main()
