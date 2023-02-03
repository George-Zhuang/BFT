'''
The IOU distance of two input detection boxes is calculated. 
A function called get_highest_iou is defined, which accepts 
a predicted box and a list of multiple detection boxes, calculates
the IOU distance between the predicted box and each detection box,
and returns the highest score and the corresponding box.
'''

'''
This is a function that takes as input the path to the predicted
box file, the path to the candidate box file, and a threshold.
It reads each prediction box and candidate box in the file,
then calculates the IOU distance between the prediction box
and each candidate box, and finds the candidate box with the 
largest IOU distance. If the maximum IOU distance is not higher
than the threshold, an exception is thrown. Otherwise, 
it prints the maximum IOU distance and the corresponding candidate box.
'''

import os

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_highest_iou(pred_box, gt_boxes):
    ious = []
    for gt_box in gt_boxes:
        ious.append(iou(pred_box, gt_box))
    highest_iou = max(ious)
    highest_iou_index = ious.index(highest_iou)
    return highest_iou, gt_boxes[highest_iou_index]

# def predict_iou_score(predict_file, candidate_path, threshold):
#     # read the prediction boxes from the predict_file
#     with open(predict_file, 'r') as f:
#         predict_boxes = [list(map(int, line.strip().split())) for line in f]

#     candidate_files = os.listdir(candidate_path).sort()
#     # iterate through each prediction box
#     for i, predict_box in enumerate(predict_boxes):
#         max_iou = -1
#         max_iou_candidate = None

#         candidate_file_name = candidate_files[i]
#         candidate_file = os.path.join(candidate_path, candidate_file_name)

#         # read the candidate boxes from the candidate_file
#         with open(candidate_file, 'r') as f:
#             candidate_boxes = [list(map(int, line.strip().split())) for line in f]

#         # get candidate box with the highest iou score
#         max_iou, max_iou_candidate = get_highest_iou(predict_box, candidate_boxes)

#         # check if the max iou is greater than the threshold
#         if max_iou <= threshold:
#             raise Exception("IOU score is below threshold.")
#         else:
#             print("Max IOU score:", max_iou)
#             print("Corresponding candidate box:", max_iou_candidate)

# def predict_iou_score(predict_box, candidate_boxes, threshold):
#     # get candidate box with the highest iou score
#     max_iou, max_iou_candidate = get_highest_iou(predict_box, candidate_boxes)

#     # check if the max iou is greater than the threshold
#     if max_iou <= threshold:
#         raise Exception("IOU score is below threshold.")
#     else:
#         print("Max IOU score:", max_iou)
#         print("Corresponding candidate box:", max_iou_candidate)


if __name__ == '__main__':
    # Example usage
    pred_box = [0, 0, 10, 10]
    gt_boxes = [[0, 0, 20, 20], [10, 10, 30, 30], [5, 5, 15, 15]]
    highest_iou, corresponding_box = get_highest_iou(pred_box, gt_boxes)
    print("highest iou: ", highest_iou)
    print("corresponding box: ", corresponding_box)

