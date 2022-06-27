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

    config = Config(logging)

    parser.add_argument('-na', '--no-animation', help="no animation is shown.", default=not config.show_animation)
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", default=not config.draw_plot)
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", default=config.quiet)
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    parser.add_argument('--iou', metavar='N', type=float, nargs='+', help='iou overlap', default=config.iou)
    parser.add_argument('--result-path', type=str, help='result dir path', default=config.result_path)
    parser.add_argument('--truth-path', type=str, help='ground truth dir path', default=config.ground_truth_path)
    parser.add_argument('--image-path', type=str, help='image dir path', default=config.image_path)
    parser.add_argument('--data-format', type=str, help='input data format, legacy or yolo', default=config.yolo_format)

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

    config.yolo_format = args.data_format == 'yolo'
    config.iou = args.iou
    config.ground_truth_path = args.truth_path
    config.result_path = args.result_path
    config.image_path = args.image_path

    config.print()

    # if there are no classes to ignore then replace None by empty list
    if args.ignore is None:
        args.ignore = []

    specific_iou_flagged = False
    if args.set_class_iou is not None:
        specific_iou_flagged = True

    # make sure that the cwd() is the location of the python script (so that every path makes sense)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if os.path.exists(config.image_path):
        for dirpath, dirnames, files in os.walk(config.image_path):
            if not files:
                # no image files found
                args.no_animation = True
    else:
        args.no_animation = True

    # try to import OpenCV if the user didn't choose the option --no-animation
    config.show_animation = not args.no_animation
    # try to import Matplotlib if the user didn't choose the option --no-plot
    config.draw_plot = not args.no_plot

    """
     Create a ".temp_files/" and "output/" directory
    """
    if os.path.exists(config.output_files_path): # if it exist already
        # reset the output directory
        shutil.rmtree(config.output_files_path)

    if not os.path.exists(config.temp_file_path): # if it doesn't exist already
        os.makedirs(config.temp_file_path)

    if not os.path.exists(config.output_files_path): # if it doesn't exist already
        os.makedirs(config.output_files_path)

    if config.draw_plot:
        os.makedirs(os.path.join(config.output_files_path, "classes"))

    if config.show_animation:
        os.makedirs(os.path.join(config.output_files_path, "images", "detections_one_by_one"))

    specific_iou_classes = parse_specific_class(args, specific_iou_flagged)

    gt_counter_per_class, counter_images_per_class, \
    gt_files, ground_truth_files_list, gt_classes, n_classes = init_ground_truth(config, args.ignore)

    dr_files_list = init_detection(config, gt_classes)

    count_true_positives, lamr_dictionary, ap_dictionary, mAP = compute(config, specific_iou_classes,
            gt_classes, draw_text_in_image, n_classes, gt_counter_per_class,
            voc_ap, counter_images_per_class, log_average_miss_rate)

    """
     Draw false negatives
    """
    if config.show_animation:
        pink = (203,192,255)
        for tmp_file in gt_files:
            ground_truth_data = json.load(open(tmp_file))
            #print(ground_truth_data)
            # get name of corresponding image
            start = config.temp_file_path + '/'
            img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
            img_cumulative_path = config.output_files_path + "/images/" + img_id + ".jpg"
            img = cv2.imread(img_cumulative_path)
            if img is None:
                img_path = config.image_path + '/' + img_id + ".jpg"
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
    if config.draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = config.output_files_path + "/ground-truth-info.png"
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
    with open(config.output_files_path + "/output.txt", 'a') as output_file:
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
    if config.draw_plot:
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        plot_title += "(" + str(len(dr_files_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = config.output_files_path + "/detection-results-info.png"
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
    with open(config.output_files_path + "/output.txt", 'a') as output_file:
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
    if config.draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = config.output_files_path + "/lamr.png"
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
    if config.draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        x_label = "Average Precision"
        output_path = config.output_files_path + "/mAP.png"
        to_show = False
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
