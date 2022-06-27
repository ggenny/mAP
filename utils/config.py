class Config:
    def __init__(self, logging):

        self.logging = logging
        self.iou = 0.5
        self.ground_truth_path = 'input/ground-truth'
        self.result_path = 'input/detection-results'
        self.image_path = 'input/images-optional'
        self.yolo_format = "legacy"
        self.output_files_path = "output"
        self.temp_file_path = "output/data"

        self.show_animation = True
        self.draw_plot = True
        self.quiet = False
        self.ignore_classes = []
        self.set_class_iou = []

    def print(self):
        self.logging.info("IOU         : %s", self.iou)
        self.logging.info("Result path : %s", self.ground_truth_path)
        self.logging.info("Truth path  : %s", self.result_path)
        self.logging.info("Image path  : %s", self.image_path)
