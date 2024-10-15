
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset, DriveDatasetPickle, BoundaryLossDatasetPickle
from model import AttU_Net, NestedUNet, SalsaNext #build_unet
from model_unet import build_unet
from transunet import U_Transformer
from loss import *
from utils import seeding, create_dir, epoch_time, calculate_sha256, save_json, backup_checkpoint
from models.unet import UNet
def train(model, loader, optimizer, loss_fn, dis_map, device):
    epoch_loss = 0.0

    model.train()
    if not dis_map:
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    else:
        for x, y, dst in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            dst = dst.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y, dst)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, dis_map, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        if not dis_map:
            for x, y in loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                epoch_loss += loss.item()
        else:
            for x, y, dst in loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                dst = dst.to(device, dtype=torch.float32)

                y_pred = model(x)
                loss = loss_fn(y_pred, y, dst)
                epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def run(training_plan):
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    TRAIN_DIST_MAP_DIR = "asdasda"
    VAL_DIST_MAP_DIR = "asdasda"
    train_x = sorted(glob("/env/csc413_project/boostnet_labeldata_vicky/data/training/*"))
    train_y = sorted(glob("/env/csc413_project/boostnet_labeldata_vicky/labels/training/*"))
    train_dist_map = sorted(glob(TRAIN_DIST_MAP_DIR))

    valid_x = sorted(glob("/env/csc413_project/boostnet_labeldata_vicky/data/valid/*"))
    valid_y = sorted(glob("/env/csc413_project/boostnet_labeldata_vicky/labels/valid/*"))
    valid_dist_map = sorted(glob(VAL_DIST_MAP_DIR))


    #data_pickle = "/env/csc413_project/boostnet_fixed/data/augmentedTrainData128by352.pickle"
    #mask_pickle = "/env/csc413_project/boostnet_fixed/labels/augmentedTrainDC128by352_3.pickle"
    #data_pickle = "/env/csc413_project/boostnet_labeldata_luke_aug/data/augmentedTrainData128by352.pickle"
    #mask_pickle = "/env/csc413_project/boostnet_labeldata_luke_aug/labels/augmentedTrainLabel128by352_1.pickle"
    #dist_pickle = "/env/csc413_project/boostnet_labeldata_luke_aug/labels/augmentedTrainDC128by352_1.pickle"
    data_pickle = "/env/csc413_project/boostnet_aug_new/data/augmentedTrainData.pickle"
    mask_pickle = "/env/csc413_project/boostnet_aug_new/labels/augmentedTrainLabel_normalLosses.pickle"
    if (training_plan["loss"] == "Boundary" or training_plan["loss"] == "BoundaryModified"):
        mask_pickle = "/env/csc413_project/boostnet_aug_new/labels/augmentedTrainLabel_boundaryLosses.pickle"
    dist_pickle = "/env/csc413_project/boostnet_aug_new/labels/augmentedTrainDistMap_boundaryLosses.pickle"

    # ==============626
    data_pickle = "/env/csc413_project/aug626/data/normalTraining626.pickle"
    mask_pickle = "/env/csc413_project/aug626/labels/normalLosses626.pickle"
    if (training_plan["loss"] == "Boundary" or training_plan["loss"] == "BoundaryModified"):
        mask_pickle = "/env/csc413_project/aug626/labels/boundaryLosses626.pickle"
    dist_pickle = "/env/csc413_project/aug626/labels/boundaryLossesDistmap626.pickle"

    """ Hyperparameters """
    loss = training_plan["loss"]
    H = 512
    W = 256
    size = (W, H)
    batch_size = training_plan["batch_size"] #2
    num_epochs = training_plan["epochs"] #100
    lr = 1e-4
    checkpoint_path = f"files/{training_plan['name']}.pth"
    backup_checkpoint(checkpoint_path)

    print(f"Hyperparameters:\nBatch Size: {batch_size} - Epochs: {num_epochs} - Learning Rate: {lr}")
    print(f"Image Size: {size}")
    print(f"Loss: {loss}")
    print(f"Model: {training_plan['name']}")
    print(f"Checkpoint Path: {checkpoint_path}")



    """ Dataset and loader """
    if loss == "Boundary" or loss == "BoundaryModified":
        #train_dataset = DriveDataset(train_x, train_y,train_dist_map, image_size=size)
        #valid_dataset = DriveDataset(valid_x, valid_y, valid_dist_map, image_size=size)
        total_dataset = BoundaryLossDatasetPickle(data_pickle, mask_pickle, dist_pickle, image_size=size)
        #total_dataset = DriveDatasetPickle(data_pickle, mask_pickle, dist_pickle, image_size=size)
        print("Total Data Size:", len(total_dataset))
        total_dataset_size = len(total_dataset)
        training_size = int(total_dataset_size * 0.8)
        validation_size = total_dataset_size - training_size
        train_dataset, valid_dataset = torch.utils.data.random_split(total_dataset, [training_size, validation_size])

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        #train_dataset = DriveDataset(train_x, train_y, image_size=size)
        #valid_dataset = DriveDataset(valid_x, valid_y, image_size=size)
        total_dataset = DriveDatasetPickle(data_pickle, mask_pickle, image_size=size)
        print("Total Data Size:", len(total_dataset))
        total_dataset_size = len(total_dataset)
        training_size = int(total_dataset_size*0.8)
        validation_size = total_dataset_size - training_size
        train_dataset, valid_dataset = torch.utils.data.random_split(total_dataset, [training_size, validation_size])

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    data_str = f"Dataset Size:\nTrain: {len(train_dataset)} - Valid: {len(valid_dataset)}\n"
    print(data_str)
    device = torch.device('cuda')
    model = training_plan["model"] #UNet(1, 1)#SalsaNext() #U_Transformer() #
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = None
    dis_map = False
    if loss == "DiceBCE":
        loss_fn = DiceBCELoss()
    elif loss == "BCE":
        loss_fn = BCELoss()
    elif loss == "MSE":
        loss_fn = MSELoss()
    elif loss == "Boundary":
        loss_fn = Boundary_Loss()
        dis_map = True
    elif loss == "BoundaryModified":
        loss_fn = Boundary_Loss_Modified()
        dis_map = True
    elif loss == "LossAll":
        loss_fn = LossAll()

    """ Training the model """
    best_valid_loss = float("inf")
    epoch_history = []
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, dis_map, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, dis_map, device)
        improve_log = ""
        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            improve_log = data_str
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

        epoch_log_entry = {
            "epoch": f"{epoch+1:02}",
            "epoch_time": f"{epoch_mins}m {epoch_secs}s",
            "train_loss": f"{train_loss:.3f}",
            "valid_loss": f"{valid_loss:.3f}",
            "improve": f"{improve_log}",
        }
        epoch_history.append(epoch_log_entry)

        summary_log = {
            "dataset": {
                "train": f"{len(train_dataset)}",
                "valid": f"{len(valid_dataset)}",
            },
            "hyperparameters": {
                "batch_size": f"{batch_size}",
                "epochs": f"{num_epochs}",
                "learning_rate": f"{lr}",
                "H": f"{H}",
                "W": f"{W}",
                "loss": f"{loss}",
            },
            "model": {
                "name": f"{training_plan['name']}",
                "checkpoint_path": f"{checkpoint_path}",
                "sha256": f"{calculate_sha256(checkpoint_path)}",
            },
            "epoch_history": epoch_history,
            "model_details": f"{training_plan['model']}",
        }
        save_json(f"files/{training_plan['name']}.json", summary_log)


if __name__ == "__main__":
    training_plan1 = {
        "name": "UNet_AUGNEW_2886_Boundary_epoch60",
        "model": UNet(1, 2, activation="softmax"),
        "loss": "Boundary",
        "batch_size": 16,
        "epochs": 60,
    }
    training_plan_boundary_modified = {
        "name": "UNet_AUGNEW_2886_Boundary_Modified_epoch60",
        "model": UNet(1, 2, activation="softmax"),
        "loss": "BoundaryModified",
        "batch_size": 16,
        "epochs": 60,
    }
    training_plans_salsa = {
        "name": "Salsa_ModifiedBoundary_epoch30",
        "model": SalsaNext(1, 2, activation="softmax"),
        "loss": "BoundaryModified",
        "batch_size": 8,
        "epochs": 30,
    }
    #run(training_plans_salsa)


    transformer = {
        "name": "U_Transformer_BoundaryModified",
        "model": U_Transformer(1, 2, activation="softmax"),
        "loss": "BoundaryModified",
        "batch_size": 1,
        "epochs": 30,
    }
    #run(transformer)

    transformer = {
        "name": "U_Transformer_Boundary",
        "model": U_Transformer(1, 2, activation="softmax"),
        "loss": "Boundary",
        "batch_size": 1,
        "epochs": 30,
    }
    #run(transformer)
    #============================04/17 Morning============================
    plan = {
        "name": "UNetRes50_Boundary",
        "model": UNetRes50(1, 2, activation="softmax"),
        "loss": "Boundary",
        "batch_size": 16,
        "epochs": 60,
    }
    #run(plan)

    plan = {
        "name": "UNetRes50_BoundaryModified",
        "model": UNetRes50(1, 2, activation="softmax"),
        "loss": "BoundaryModified",
        "batch_size": 16,
        "epochs": 60,
    }
    #run(plan)

    plan = {
        "name": "UNetRes50_MSE",
        "model": UNetRes50(1, 1),
        "loss": "MSE",
        "batch_size": 16,
        "epochs": 60,
    }
    #run(plan)

    plan = {
        "name": "UNetRes50_BCE",
        "model": UNetRes50(1, 1),
        "loss": "BCE",
        "batch_size": 16,
        "epochs": 60,
    }
    #run(plan)

    plan = {
        "name": "RegressionRes34",
        "model": create_net(),
        "loss": "LossAll",
        "batch_size": 16,
        "epochs": 1,
    }
    #run(plan)

    # ============================04/17 626 Aug============================

    plan = {
        "name": "UNet626_MSE",
        "model": UNet(),
        "loss": "MSE",
        "batch_size": 16,
        "epochs": 60,
    }

    #run(plan)

    plan = {
        "name": "UNet626_BCE",
        "model": UNet(),
        "loss": "BCE",
        "batch_size": 16,
        "epochs": 60,
    }

    #run(plan)

    plan = {
        "name": "UNet626_Boundary",
        "model": UNet(1, 2, activation="softmax"),
        "loss": "Boundary",
        "batch_size": 16,
        "epochs": 60,
    }

    #run(plan)

    plan = {
        "name": "UNet626_BoundaryModified",
        "model": UNet(1, 2, activation="softmax"),
        "loss": "BoundaryModified",
        "batch_size": 16,
        "epochs": 60,
    }

    #run(plan)

    plan = {
        "name": "AttU_Net626_MSE",
        "model": AttU_Net(1, 1),
        "loss": "MSE",
        "batch_size": 16,
        "epochs": 60,
    }

    run(plan)

    plan = {
        "name": "AttU_Net626_BCE",
        "model": AttU_Net(1, 1),
        "loss": "BCE",
        "batch_size": 16,
        "epochs": 60,
    }

    #run(plan)

    plan = {
        "name": "AttU_Net626_Boundary",
        "model": AttU_Net(1, 2, activation="softmax"),
        "loss": "Boundary",
        "batch_size": 16,
        "epochs": 60,
    }

    #run(plan)

    plan = {
        "name": "AttU_Net626_BoundaryModified",
        "model": AttU_Net(1, 2, activation="softmax"),
        "loss": "BoundaryModified",
        "batch_size": 16,
        "epochs": 60,
    }

    #run(plan)

    #print(glob("/*"))
    #run("Boundary")
    #run("BoundaryModified")
