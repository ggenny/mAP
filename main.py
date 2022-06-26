import argparse
import logging
import json
import shutil
from utils import *
import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )

    stdout_logger = logging.getLogger('stdout')

    # Parsing input arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    parser.add_argument('--iou', metavar='N', type=float, nargs='+', help='iou overlap', default=0.5)
    parser.add_argument('--result-path', type=str, help='result dir path', default='input/detection-results')
    parser.add_argument('--truth-path', type=str, help='ground truth dir path', default='input/ground-truth')
    parser.add_argument('--image-path', type=str, help='image dir path', default='input/images-optional')
    parser.add_argument('--data-format', type=str, help='input data format, legacy or yolo', default='legacy')

    args = parser.parse_args()

    '''
        0,0 ------> x (width)
         |
         |  (Left,Top)
         |      *_________
         |      |         |
                |         |
         y      |_________|
      (height)            *
                    (Right,Bottom)
    '''

    yolo_format = args.data_format == 'yolo'
    MINOVERLAP = args.iou
    GT_PATH = args.truth_path
    DR_PATH = args.result_path
    IMG_PATH = args.image_path

    logging.info("IOU         : %s", MINOVERLAP)
    logging.info("Result path : %s", GT_PATH)
    logging.info("Truth path  : %s", DR_PATH)
    logging.info("Image path  : %s", IMG_PATH)

    # if there are no classes to ignore then replace None by empty list
    if args.ignore is None:
        args.ignore = []

    specific_iou_flagged = False
    if args.set_class_iou is not None:
        specific_iou_flagged = True

    # make sure that the cwd() is the location of the python script (so that every path makes sense)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                # no image files found
                args.no_animation = True
    else:
        args.no_animation = True

    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = not args.no_animation
    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = not args.no_plot

    """
     Create a ".temp_files/" and "output/" directory
    """
    output_files_path = "output"
    if os.path.exists(output_files_path): # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)

    TEMP_FILES_PATH = "output/data"
    if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)

    if not os.path.exists(output_files_path): # if it doesn't exist already
        os.makedirs(output_files_path)

    if draw_plot:
        os.makedirs(os.path.join(output_files_path, "classes"))

    if show_animation:
        os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))

    specific_iou_classes = parse_specific_class(args, specific_iou_flagged)

    gt_counter_per_class, counter_images_per_class, gt_files, ground_truth_files_list, gt_classes, n_classes = init_ground_truth(GT_PATH, DR_PATH, TEMP_FILES_PATH, args.ignore, yolo_format)
    dr_files_list = init_detection(DR_PATH, GT_PATH, TEMP_FILES_PATH, gt_classes, yolo_format)

    count_true_positives, lamr_dictionary, ap_dictionary, mAP = compute(output_files_path, TEMP_FILES_PATH, IMG_PATH, specific_iou_classes, MINOVERLAP,
            gt_classes, show_animation, draw_text_in_image, n_classes, gt_counter_per_class,
            voc_ap, counter_images_per_class, log_average_miss_rate, draw_plot)

    """
     Draw false negatives
    """
    if show_animation:
        pink = (203,192,255)
        for tmp_file in gt_files:
            ground_truth_data = json.load(open(tmp_file))
            #print(ground_truth_data)
            # get name of corresponding image
            start = TEMP_FILES_PATH + '/'
            img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
            img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
            img = cv2.imread(img_cumulative_path)
            if img is None:
                img_path = IMG_PATH + '/' + img_id + ".jpg"
                img = cv2.imread(img_path)
            # draw false negatives
            for obj in ground_truth_data:
                if not obj['used']:
                    bbgt = [ int(round(float(x))) for x in obj["bbox"].split() ]
                    cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),pink,2)
            cv2.imwrite(img_cumulative_path, img)

    # remove the temp_files directory
    # shutil.rmtree(TEMP_FILES_PATH)

    """
     Count total of detection-results
    """
    # iterate through all the files
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        # get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            # count that object
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_name] = 1
    #print(det_counter_per_class)
    dr_classes = list(det_counter_per_class.keys())


    """
     Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = output_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            )

    """
     Write number of ground-truth objects per class to results.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
     Finish counting true positives
    """
    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    #print(count_true_positives)

    """
     Plot the total number of occurences of each class in the "detection-results" folder
    """
    if draw_plot:
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        plot_title += "(" + str(len(dr_files_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = output_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(
            det_counter_per_class,
            len(det_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
            )

    """
     Write number of detected objects per class to output.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            output_file.write(text)

    """
     Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = output_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    """
     Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        x_label = "Average Precision"
        output_path = output_files_path + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )
