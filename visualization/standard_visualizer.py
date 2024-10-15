import numpy as np
import matplotlib.pyplot as plt
import torch
def visualize_standard(pred_y):
    ## (1, 512, 256)
    pred_y = torch.round(pred_y)
    pred_y = pred_y[0].cpu().numpy()
    pred_y = np.squeeze(pred_y, axis=0)
    pred_y = np.array(pred_y, dtype=np.uint8) * 255
    #print(pred_y.shape, np.max(pred_y), np.min(pred_y))
    return pred_y

def visualize_boundary(pred_y):
    pred_y = torch.round(pred_y)
    pred_y = pred_y[0].cpu().numpy()
    pred_y = pred_y[0]
    pred_y = np.array(pred_y, dtype=np.uint8) * 255



    #pred1 = pred_y[1]
    #pred2 = -pred0
    #print(np.sum(pred2 == pred1))
    # draw two images
    #fig, ax = plt.subplots(1, 2)
    #ax[0].imshow(pred0, cmap='gray')
    #ax[1].imshow(pred1, cmap='gray')
    #ax[2].imshow(pred2, cmap='gray')
    #plt.show()
    return pred_y
