import numpy as np
import pandas as pd
import os
import cv2
import torch
import matplotlib.pyplot as plt
from utils import save_json
import concurrent.futures
import multiprocessing

class CsvDB():
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.db = self.get_db()

    def get_db(self):
        if not os.path.exists(self.csv_file):
            raise Exception('File does not exist: ' + self.csv_file)
        return pd.read_csv(self.csv_file, header= None)

    def get(self, index):
        return self.db.loc[index].values

class FilenamesDB(CsvDB):
    def __init__(self, csv_file):
        super().__init__(csv_file)

    def get_filenames_enumerator(self):
        return enumerate(self.db.iloc[:, 0])
    def get_filenames(self):
        return self.db.iloc[:, 0]

class LandmarksDB(CsvDB):
    def __init__(self, csv_file):
        super().__init__(csv_file)

    def get_landmarks_enumerator(self):
        return enumerate(self.db.iloc[:, 1:])
    def get(self, index):
        landmarks = super().get(index)
        return np.array(landmarks)

class LandmarksHelper():
    def __init__(self, landmarks):
        self.landmarks = landmarks.reshape(2, -1)
        self.LANDMARKS_PER_IMAGE = 68
    def get_pointxy(self, box_index, point_index):
        return self.landmarks[:, box_index * 4 + point_index]

    def get_avg_slope(self, box_index, w, h):
        p0 = self.get_pointxy(box_index, 0)
        p1 = self.get_pointxy(box_index, 1)
        p2 = self.get_pointxy(box_index, 2)
        p3 = self.get_pointxy(box_index, 3)
        slope1 = (round(p1[1]*h) - round(p0[1]*h)) / (round(p1[0]*w) - round(p0[0]*w) + 1e-7)
        slope2 = (round(p3[1]*h) - round(p2[1]*h)) / (round(p3[0]*w) - round(p2[0]*w) + 1e-7)
        return (slope1 + slope2) / 2

    def get_cobb_angles(self, w, h):
        avg_slopes = []
        for i in range(self.LANDMARKS_PER_IMAGE // 4):
            avg_slopes.append(self.get_avg_slope(i, w, h))

        cobb_angle = self.find_cobb_angles(avg_slopes)
        return cobb_angle
    def find_cobb_angles(self, vertebra_slopes):

        cobb_angles = [0.0, 0.0, 0.0]
        if not isinstance(vertebra_slopes, np.ndarray):
            vertebra_slopes = np.array(vertebra_slopes)

        max_slope = np.amax(vertebra_slopes)
        min_slope = np.amin(vertebra_slopes)

        lower_MT = np.argmax(vertebra_slopes)
        upper_MT = np.argmin(vertebra_slopes)

        if lower_MT < upper_MT:
            lower_MT, upper_MT = upper_MT, lower_MT

        upper_max_slope = np.amax(vertebra_slopes[0:upper_MT + 1])
        upper_min_slope = np.amin(vertebra_slopes[0:upper_MT + 1])

        lower_max_slope = np.amax(vertebra_slopes[lower_MT:17])
        lower_min_slope = np.amin(vertebra_slopes[lower_MT:17])

        cobb_angles[0] = abs(np.rad2deg(np.arctan((max_slope - min_slope) / (1 + max_slope * min_slope))))
        cobb_angles[1] = abs(
            np.rad2deg(np.arctan((upper_max_slope - upper_min_slope) / (1 + upper_max_slope * upper_min_slope))))
        cobb_angles[2] = abs(
            np.rad2deg(np.arctan((lower_max_slope - lower_min_slope) / (1 + lower_max_slope * lower_min_slope))))

        return cobb_angles

def get_dot_landmarks_to_norm_landmarks(get_dot_landmarks):
    landmarks = get_dot_landmarks.copy()
    landmarks.resize((len(landmarks) // 2, 2))
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    landmarks_norm = np.concatenate((landmarks_x, landmarks_y))
    return landmarks_norm

def get_dots(curr_predict, IMG_SIZE_X, IMG_SIZE_Y):
    ######## P #########
    processed = np.zeros((IMG_SIZE_Y, IMG_SIZE_X, 3))
    thresholded = np.zeros((IMG_SIZE_Y, IMG_SIZE_X))

    for i in range(IMG_SIZE_Y):
        for j in range(IMG_SIZE_X):

            if curr_predict[i][j][0] == 1:
                processed[i][j] = (1, 1, 1)
                thresholded[i][j] = 1

            else:
                processed[i][j] = (0, 0, 0)
                thresholded[i][j] = 0

    c, h = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ################################################ size filteration ###################################
    size = [cv2.contourArea(item) for item in c]
    difference_factor = [0.6, 0.6]
    for i in range(2):
        midpoint = len(size) // 2
        upper = size[:midpoint]
        lower = size[midpoint:]
        if (upper == [] or lower == []):
            break
        upper_avg = sum(size[:midpoint]) / len(size[:midpoint])
        lower_avg = sum(size[midpoint:]) / len(size[midpoint:])
        if (upper_avg == 0 or lower_avg == 0):
            break
        for item in upper:
            if (upper_avg - item) / upper_avg > difference_factor[i]:
                size.remove(item)

        for item in lower:
            if (lower_avg - item) / lower_avg > difference_factor[i]:
                size.remove(item)

        midpoint_round_2 = len(size) // 2

    processed_c = []
    for item in c:
        if cv2.contourArea(item) in size:
            processed_c.append(item)
    ##################################################################################################
    ################################## variables for location filteration ########################
    curr_landmarks = []
    trigger = False
    previous_bot_left_x = 0
    len_c = len(processed_c)
    #################################################################################################
    for item in processed_c[::-1]:
        curr_counter = item.reshape(-1, 2)

        rect = cv2.minAreaRect(curr_counter)
        box = cv2.boxPoints(rect).astype(np.float64)
        # print("box: ", box.shape)
        ######################################### rearrange box points ################################
        x_sorted = box[np.argsort(box[:, 0])]
        y_sorted_left = x_sorted[:2][np.argsort(x_sorted[:2, 1])]
        y_sorted_right = x_sorted[2:][np.argsort(x_sorted[2:, 1])]
        box = np.asarray([y_sorted_left[0], y_sorted_right[0], y_sorted_left[1], y_sorted_right[1]])
        ###################################################################################################

        ################################### location filteration ########################################
        if trigger:
            if np.abs(previous_bot_left_x - box[0][0]) >= 40:
                len_c -= 1
            if np.abs(previous_bot_left_x - box[0][0]) < 40:
                previous_bot_left_x = box[2][0]

                # for (x, y) in box:
                #   x = int(x)
                #   y = int(y)
                #   cv2.circle(processed, (x, y), 1, (1, 0, 0), 1)

                box[:, 0] = box[:, 0] / IMG_SIZE_X
                box[:, 1] = box[:, 1] / IMG_SIZE_Y

                curr_landmarks.append(box)
        if not trigger:
            previous_bot_left_x = box[2][0]
            trigger = True

            box[:, 0] = box[:, 0] / IMG_SIZE_X
            box[:, 1] = box[:, 1] / IMG_SIZE_Y
            curr_landmarks.append(box)

        ##################################################################################################

    if len_c < 17:
        for i in range(17 - len_c):
            curr_landmarks.append(curr_landmarks[-1])
    elif len_c > 17:
        curr_landmarks = curr_landmarks[-17:]
    curr_landmarks = np.ndarray.flatten(np.asarray(curr_landmarks))
    return curr_landmarks, processed

def get_width_height(image_file_name):
    img = cv2.imread(image_file_name)
    height, width, _ = img.shape
    return width, height


def process_image_to_landmarks(index, filename, pred_image_path, original_image_path):
    # filename = filename.replace(".jpg", ".png")
    img = cv2.imread(os.path.join(pred_image_path, filename), cv2.IMREAD_GRAYSCALE)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    h, w, _ = img.shape
    # print(w, h)
    original_w, original_h = get_width_height(os.path.join(original_image_path, filename))
    if True:
        scale = original_w / 256
        # h = imgsize_db.get(i)[1]
        h = np.round(original_h / scale).astype(np.int)
        w = 256  # np.round(imgsize_db.get(i)[0] / scale).astype(np.int)
    ####print(index, w, h)

    img = cv2.resize(img, (w, h))
    img = img.reshape((img.shape[0], img.shape[1], 1))
    curr_landmarks, processed = get_dots(torch.tensor(img) // 255, w, h)


    norm_landmarks = get_dot_landmarks_to_norm_landmarks(curr_landmarks)
    # print(w, h)
    rw, rh = original_w, original_h
    helper = LandmarksHelper(norm_landmarks)
    cobb_angles = helper.get_cobb_angles(rw, rh)
    return (index, (cobb_angles, norm_landmarks))

def execute_in_processes(filenames, pred_image_path, original_image_path, num_processes):
    results = [None] * len(filenames)  # Pre-allocate the results list
    # Create a list of argument tuples for each task
    args_list = [(index, filename, pred_image_path, original_image_path) for index, filename in enumerate(filenames)]
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map tasks to the process pool
        output = pool.starmap(process_image_to_landmarks, args_list)

        # Collect results and place them in the appropriate position
        for index, result in output:
            results[index] = result

    return results


def image_to_landmarks(original_image_path, pred_image_path, filenames_db, output_landmarks_csv, threads=16):
    cobb_angles_list = []
    landmark_list = []
    filenames = filenames_db.get_filenames()
    results = execute_in_processes(filenames, pred_image_path, original_image_path, threads)

    for r in results:
        angle, landmarks = r
        cobb_angles_list.append(angle)
        landmark_list.append(landmarks)
    df = pd.DataFrame(landmark_list)
    df.to_csv(output_landmarks_csv, index=False, header=False)
    return landmark_list, cobb_angles_list

def landmarks_to_cobb_angle_list(original_data_path, filenames_db, landmarks_db):
    cobb_angles_list = []
    h, w = 0, 0
    for i, filename in filenames_db.get_filenames_enumerator():
        landmarks_entry = landmarks_db.get(i)
        # Assume the image have the same size
        #if h == 0 or w == 0:
        #print(os.path.join(self.dir.training_data, filename))
        img = cv2.imread(os.path.join(original_data_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        landmarkshelper = LandmarksHelper(landmarks_entry)
        cobb_angles = landmarkshelper.get_cobb_angles(w, h)
        cobb_angles_list.append(cobb_angles)

    return cobb_angles_list

def np_to_csv(npdata, output_filename):
    df = pd.DataFrame(npdata)
    df.to_csv(output_filename, index=False, header=False)

def evaluate_cobb_angles(angles, angles_gt):
    '''

    :param angles: predicted (n * 3)
    :param angles_gt: groundtruth (n * 3)
    :return:
    '''
    output = []
    DETAIL = True
    if (angles.shape != angles_gt.shape):
        raise Exception('The shape of angles and angles_gt is not equal')
    n, _ = angles.shape

    smape_lt_5 = 0
    smape_5_10 = 0
    smape_10_20 = 0
    smape_gt_20 = 0

    smape = 0
    diffs = []
    angles_diff = np.abs(angles - angles_gt)
    total_diff = angles_diff.sum(axis=1)

    bench_lt_5 = np.sum(angles_diff <= 5)
    bench_5_10 = np.sum((angles_diff > 5) & (angles_diff <= 10))
    bench_10_20 = np.sum((angles_diff > 10) & (angles_diff <= 20))
    bench_gt_20 = np.sum(angles_diff > 20)
    denom = angles + angles_gt
    for i in range(n):
        if (DETAIL):
            output.append(str(i + 1) + " prediction: " + str(angles[i]))
            output.append(str(i + 1) + " groundtruth: " + str(angles_gt[i]))
            output.append(str(i + 1) + " difference: " + str(angles_diff[i]))
            output.append(str(i + 1) + " diff_sum: " + str(np.sum(angles_diff[i])))
        diffs += list(angles_diff[i])
        sam = total_diff[i] / denom[i].sum()
        smape += sam
        if sam >= 0.2:
            smape_gt_20 += 1
        elif sam >= 0.1:
            smape_10_20 += 1
        elif sam >= 0.05:
            smape_5_10 += 1
        else:
            smape_lt_5 += 1
        if DETAIL:
            output.append("SMAPE: " + str(sam))
            output.append("\n")
    smape /= n
    output.append("SMAPE: " + str(smape))
    if DETAIL:

        output.append("Difference less than 5 degree: " + str(bench_lt_5 / (n * 3)))
        output.append("Difference between 5 and 10 degree: " + str(bench_5_10 / (n * 3)))
        output.append("Difference between 10 and 20 degree: " + str(bench_10_20 / (n * 3)))
        output.append("Difference above 20 degree: " + str(bench_gt_20 / (n * 3)))
        output.append("\n")
        output.append("Difference less than 5 degree: " + str(bench_lt_5))
        output.append("Difference between 5 and 10 degree: " + str(bench_5_10))
        output.append("Difference between 10 and 20 degree: " + str(bench_10_20))
        output.append("Difference above 20 degree: " + str(bench_gt_20))
        output.append("\n")
    output.append("# SMAPE less than 5%: " + str(smape_lt_5))
    output.append("# SMAPE between 5% and 10%: " + str(smape_5_10))
    output.append("# SMAPE between 10% and 20%: " + str(smape_10_20))
    output.append("# SMAPE above 20%: " + str(smape_gt_20))
    plt.hist(diffs)
    plt.xlabel("absolute angle difference")
    plt.ylabel("num cases")
    plt.show()
    output_text = "\n".join(output)
    return smape, output_text


def evaluate_smape(test_x_path, test_y_path, filenames_csv_path, gt_angles_csv_path, model_name, threads=16):

    filenames_db = FilenamesDB(filenames_csv_path)
    smape_output_dir = os.path.join("results/smape", model_name)
    if not os.path.exists(smape_output_dir):
        os.makedirs(smape_output_dir)
    else:
        print(smape_output_dir)
        print("Error: smape output directory already exists. Results may be overwritten.")

    test_landmarks_csv_path = os.path.join(smape_output_dir, "test_landmarks.csv")
    image_to_landmarks(test_x_path, test_y_path, filenames_db, test_landmarks_csv_path, threads)

    test_landmarks_db = LandmarksDB(test_landmarks_csv_path)
    test_angles = landmarks_to_cobb_angle_list(test_x_path, filenames_db, test_landmarks_db)
    test_angles_csv_path = os.path.join(smape_output_dir, "test_angles.csv")
    np_to_csv(test_angles, test_angles_csv_path)

    gt_angles_db = CsvDB(gt_angles_csv_path)

    smape, output_text = evaluate_cobb_angles(np.array(test_angles), gt_angles_db.db.values)

    smape_summary = {
        "name": model_name,
        "smape": smape,
        "output_text": output_text
    }
    save_json(os.path.join(smape_output_dir, "smape_summary.json"), smape_summary)

    return smape, output_text, smape_summary



if __name__ == '__main__':
    test_x_path = "/env/csc413_project/boostnet_labeldata/data/test"
    filenames_csv_path = "/env/csc413_project/boostnet_labeldata/labels/test/filenames.csv"
    angles_csv_path = "/env/csc413_project/boostnet_labeldata/labels/test/angles.csv"

    model_pred_path = "/env/csc413_project/Vertebra-Segmentation/results/UNet_2156_MSE"
    model_pred_path = "/env/csc413_project/Vertebra-Segmentation/results/UNet_LUKE_AUG_BCE"
    #model_pred_path = "/Projects/notebook/results/test_justin_attn_unet481"


    model_name = os.path.basename(model_pred_path)
    smape, output_text = evaluate_smape(test_x_path, model_pred_path, filenames_csv_path, angles_csv_path, model_name)
    print(output_text)
    print("SMAPE: " + str(smape))