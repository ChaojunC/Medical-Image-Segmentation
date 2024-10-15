
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from models.unet_res50 import UNetRes50
from model import AttU_Net, NestedUNet, SalsaNext #build_unet
from transunet import U_Transformer
from utils import create_dir, seeding, save_json
from models.unet import UNet
from evaluate_smape import evaluate_smape
from visualization.standard_visualizer import visualize_boundary, visualize_standard
def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


def run_test(testing_plan, threads=16):
    """ Seeding """
    seeding(42)
    print("Testing plan: ", testing_plan['name'])
    """ Folders """
    create_dir("results")

    """ Landmarks Evaluation """
    test_x_path = "/env/csc413_project/boostnet_labeldata/data/test"
    filenames_csv_path = "/env/csc413_project/boostnet_labeldata/labels/test/filenames.csv"
    angles_csv_path = "/env/csc413_project/boostnet_labeldata/labels/test/angles.csv"

    """ Load dataset """
    test_x = sorted(glob("/env/csc413_project/boostnet_labeldata_vicky/data/test/*"))
    test_y = sorted(glob("/env/csc413_project/boostnet_labeldata_vicky/data/test/*"))

    """ Hyperparameters """
    H = 512  # 512
    W = 256
    size = (W, H)



    checkpoint_path = f"files/{testing_plan['name']}.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = testing_plan["model"]
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    save_dir = os.path.join("results", testing_plan["name"])

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        x = x.replace("\\", "/")
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)  ## (512, 512, 3)
        image = cv2.resize(image, size)
        image = image.reshape((image.shape[0], image.shape[1], -1))
        x = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)
        """
        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)

            total_time = time.time() - start_time
            time_taken.append(total_time)

            pred_y = testing_plan["post_processing"](pred_y)


        """ Saving masks """
        # ori_mask = mask_parse(mask)
        #pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [pred_y], axis=1
        )

        save_file_name = os.path.join(save_dir, name + ".jpg")
        if i == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        cv2.imwrite(save_file_name, cat_images)

    # jaccard = metrics_score[0]/len(test_x)
    # f1 = metrics_score[1]/len(test_x)
    # recall = metrics_score[2]/len(test_x)
    # precision = metrics_score[3]/len(test_x)
    # acc = metrics_score[4]/len(test_x)
    # print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)
    if testing_plan["calculate_smape"]:
        smape, output_text, smape_summary = evaluate_smape(test_x_path, save_dir, filenames_csv_path, angles_csv_path,
                                            testing_plan["name"], threads)
        print("SMAPE: " + str(smape))

        smape_output_dir = os.path.join("results/smape", testing_plan['name'])
        smape_summary['fps'] = fps
        save_json(os.path.join(smape_output_dir, "smape_summary.json"), smape_summary)



if __name__ == "__main__":

    standard = {
        "name": "AttU_Net626_BCE",
        "model": AttU_Net(1, 1),
        "calculate_smape": True,
        "post_processing": visualize_standard,
    }
    boundary_test = {
        "name": "AttU_Net626_BoundaryModified",
        "model": AttU_Net(1, 2, activation="softmax"),
        "calculate_smape": True,
        "post_processing": visualize_boundary,
    }

    run_test(boundary_test, 16)