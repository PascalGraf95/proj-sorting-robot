import sys
from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from modules.camera_controller import CameraController
from modules.gui.output import Ui_MainWindow
from modules.image_processing import *

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Setup Live Camera Image
        self.cam = CameraController()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)

        # Setup Event for Image Processing Update
        self.ProcessingSelectionTable.doubleClicked.connect(self.new_processing_selected)

        # Setup Delete Button
        self.push_btn_reset.clicked.connect(self.delete_processing_operation)

        # Setup Output Button
        self.push_btn_output.clicked.connect(self.set_output_image)
        self.output_image_idx = -1

        # Setup Up and Down Button
        self.push_btn_up.clicked.connect(self.move_operation_up)
        self.push_btn_down.clicked.connect(self.move_operation_down)

        # Setup Export Button
        self.export_stack = False
        self.push_btn_export.clicked.connect(self.activate_stack_export)

        self.stack_string = ""

    def activate_stack_export(self):
        self.export_stack = True

    def move_operation_up(self):
        selected_items = self.CurrentProcessingTable.selectedItems()
        if len(selected_items):
            row = self.CurrentProcessingTable.row(selected_items[0])
            if 0 < row:
                item_at_previous_row = (self.CurrentProcessingTable.item(row-1, 0).text(),
                                        self.CurrentProcessingTable.item(row-1, 1).text())
                item_at_row = (self.CurrentProcessingTable.item(row, 0).text(),
                               self.CurrentProcessingTable.item(row, 1).text())
                self.CurrentProcessingTable.setItem(row - 1, 0, QtWidgets.QTableWidgetItem(item_at_row[0]))
                self.CurrentProcessingTable.setItem(row - 1, 1, QtWidgets.QTableWidgetItem(item_at_row[1]))
                self.CurrentProcessingTable.setItem(row, 0, QtWidgets.QTableWidgetItem(item_at_previous_row[0]))
                self.CurrentProcessingTable.setItem(row, 1, QtWidgets.QTableWidgetItem(item_at_previous_row[1]))
                self.CurrentProcessingTable.selectRow(row - 1)

    def move_operation_down(self):
        selected_items = self.CurrentProcessingTable.selectedItems()
        if len(selected_items):
            row = self.CurrentProcessingTable.row(selected_items[0])
            if row < self.CurrentProcessingTable.rowCount() - 1:
                item_at_next_row = (self.CurrentProcessingTable.item(row+1, 0).text(),
                                    self.CurrentProcessingTable.item(row+1, 1).text())
                item_at_row = (self.CurrentProcessingTable.item(row, 0).text(),
                               self.CurrentProcessingTable.item(row, 1).text())
                self.CurrentProcessingTable.setItem(row + 1, 0, QtWidgets.QTableWidgetItem(item_at_row[0]))
                self.CurrentProcessingTable.setItem(row + 1, 1, QtWidgets.QTableWidgetItem(item_at_row[1]))
                self.CurrentProcessingTable.setItem(row, 0, QtWidgets.QTableWidgetItem(item_at_next_row[0]))
                self.CurrentProcessingTable.setItem(row, 1, QtWidgets.QTableWidgetItem(item_at_next_row[1]))
                self.CurrentProcessingTable.selectRow(row + 1)

    def set_output_image(self):
        selected_items = self.CurrentProcessingTable.selectedItems()
        if len(selected_items):
            row = self.CurrentProcessingTable.row(selected_items[0])
            self.output_image_idx = row

    def delete_processing_operation(self):
        selected_items = self.CurrentProcessingTable.selectedItems()
        if len(selected_items):
            row = self.CurrentProcessingTable.row(selected_items[0])
            self.CurrentProcessingTable.removeRow(row)
            row_count = self.CurrentProcessingTable.rowCount()
            print(row_count)
            if row_count > row:
                self.CurrentProcessingTable.selectRow(row)
            elif row_count > 0:
                self.CurrentProcessingTable.selectRow(row_count-1)
            else:
                self.output_image_idx = -1

    def update_image(self):
        image = self.cam.capture_image()
        row_count = self.CurrentProcessingTable.rowCount()
        self.stack_string = ""

        for row in range(row_count):
            operation_name = self.CurrentProcessingTable.item(row, 0).text()
            parameters = self.parse_parameters(self.CurrentProcessingTable.item(row, 1).text())

            if operation_name == "Blur":
                image = cv2.blur(image, (parameters[0], parameters[0]))
                self.stack_string += "image = cv2.blur(image, ({}, {}))\n".format(parameters[0], parameters[0])
            elif operation_name == "Gaussian Blur":
                image = cv2.GaussianBlur(image, (parameters[0], parameters[0]), 0)
                self.stack_string += "image = cv2.GaussianBlur(image, ({}, {}), 0)\n".format(parameters[0],
                                                                                             parameters[0])
            elif operation_name == "Median Blur":
                image = cv2.medianBlur(image, parameters[0])
                self.stack_string += "image = cv2.medianBlur(image, {})\n".format(parameters[0])
            elif operation_name == "Otsu Binarization":
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.stack_string += "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.stack_string += "_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
            elif operation_name == "Adaptive Thresholding":
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.stack_string += "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              parameters[0], parameters[1])
                self.stack_string += "image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, " \
                                     "cv2.THRESH_BINARY, {}, {})\n".format(parameters[0], parameters[1])
            elif operation_name == "Erosion":
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.stack_string += "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                kernel = np.ones((parameters[0], parameters[0]), np.uint8)
                image = cv2.erode(image, kernel, iterations=parameters[1])
                self.stack_string += "kernel = np.ones(({}, {}), np.uint8)\n".format(parameters[0], parameters[0])
                self.stack_string += "image = cv2.erode(image, kernel, iterations={})\n".format(parameters[1])
            elif operation_name == "Dilation":
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.stack_string += "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                kernel = np.ones((parameters[0], parameters[0]), np.uint8)
                image = cv2.dilate(image, kernel, iterations=parameters[1])
                self.stack_string += "kernel = np.ones(({}, {}), np.uint8)\n".format(parameters[0], parameters[0])
                self.stack_string += "image = cv2.dilate(image, kernel, iterations={})\n".format(parameters[1])
            elif operation_name == "Opening":
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.stack_string += "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                kernel = np.ones((parameters[0], parameters[0]), np.uint8)
                image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,
                                         iterations=parameters[1])
                self.stack_string += "kernel = np.ones(({}, {}), np.uint8)\n".format(parameters[0], parameters[0])
                self.stack_string += "image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, " \
                                     "iterations={})\n".format(parameters[1])
            elif operation_name == "Closing":
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.stack_string += "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                kernel = np.ones((parameters[0], parameters[0]), np.uint8)
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,
                                         iterations=parameters[1])
                self.stack_string += "kernel = np.ones(({}, {}), np.uint8)\n".format(parameters[0], parameters[0])
                self.stack_string += "image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, " \
                                     "iterations={})\n".format(parameters[1])
            elif operation_name == "Edge":
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.stack_string += "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                image = cv2.Canny(image, parameters[0], parameters[1])
                self.stack_string += "image = cv2.Canny(image, {}, {})\n".format(parameters[0], parameters[1])
            elif operation_name == "White Balance":
                if len(image.shape) == 3:
                    if parameters[0]:
                        image_patch = get_image_patch(image, (parameters[0], parameters[1]), parameters[2])
                        mean_vals = get_mean_patch_value(image_patch)
                        self.stack_string += "image_patch = get_image_patch(image, ({}, {}]), {})\n".format(
                            parameters[0], parameters[1], parameters[2])
                        self.stack_string += "mean_vals = get_mean_patch_value(image_patch)\n"
                    else:
                        mean_vals = get_mean_patch_value(image)
                        self.stack_string += "mean_vals = get_mean_patch_value(image)\n"
                    correction_factors = get_white_balance_parameters(mean_vals)
                    image = correct_image_white_balance(image, correction_factors)
                    self.stack_string += "correction_factors = get_white_balance_parameters(mean_vals)\n"
                    self.stack_string += "image = correct_image_white_balance(image, correction_factors)\n"
            elif operation_name == "Histogram Equalization":
                if len(image.shape) == 3:
                    image = equalize_histograms(image, parameters[0], parameters[1], (parameters[2], parameters[2]))
                    self.stack_string += "image = equalize_histograms(image, {}, " \
                                         "{}, ({}, {}))\n".format(parameters[0], parameters[1], parameters[2],
                                                                  parameters[2])
            elif operation_name == "Crop":
                image = get_image_patch(image, (parameters[0], parameters[1]), parameters[2])
                image = cv2.resize(image, (1200, 1200))
                self.stack_string += "image = get_image_patch(image, ({}, {}), {})\n".format(parameters[0],
                                                                                             parameters[1],
                                                                                             parameters[2])
                self.stack_string += "image = cv2.resize(image, (1200, 1200))\n"
            elif operation_name == "Adjust Brightness":
                if len(image.shape) == 3:
                    image = increase_brightness(image, parameters[0])
                    self.stack_string += "image = increase_brightness(image, {})\n".format(parameters[0])
            elif operation_name == "Invert":
                if len(image.shape) == 2:
                    image = cv2.bitwise_not(image)
                    self.stack_string += "image = cv2.bitwise_not(image)\n"
            elif operation_name == "Find Contours":
                if len(image.shape) == 2:
                    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    self.stack_string += "contours, hierarchy = " \
                                         "cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)\n"
                    filtered_contours = []
                    for c in contours:
                        if cv2.contourArea(c) >= parameters[0]:
                            filtered_contours.append(c)

                    rects = []
                    for c in filtered_contours:
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        rects.append(box)

                    self.stack_string += """
                    filtered_contours = []
                    for c in contours:
                        if cv2.contourArea(c) >= {}:
                            filtered_contours.append(c)
                    """.format(parameters[0])

                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(image, rects, -1, (0, 0, 255), 2)
                    # cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 3)
            if row == self.output_image_idx:
                break

        if self.export_stack:
            self.export_stack = False
            f = open("modules/image_processing_stack.py", "w")
            f.write(self.stack_string)
            f.close()

        if len(image.shape) == 2:
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_Grayscale8)
        else:
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
        self.Image.setPixmap(QPixmap.fromImage(convert))

    def new_processing_selected(self):
        selected_item = self.ProcessingSelectionTable.selectedItems()[0].text()
        self.add_new_preprocessing(selected_item)

    def parse_parameters(self, row):
        parameter_string_list = row.split(",")
        parameter_string_list = [s.strip().replace(" ", "") for s in parameter_string_list]
        parameter_list = []
        for string_par in parameter_string_list:
            if str.isnumeric(string_par):
                parameter_list.append(int(string_par))
            elif "." in string_par:
                parameter_list.append(float(string_par))
            elif string_par == "False":
                parameter_list.append(False)
            elif string_par == "True":
                parameter_list.append(True)
            else:
                parameter_list.append(string_par)
        return parameter_list

    def add_new_preprocessing(self, name):
        row_count = self.CurrentProcessingTable.rowCount()
        self.CurrentProcessingTable.insertRow(row_count)
        self.CurrentProcessingTable.setItem(row_count, 0, QtWidgets.QTableWidgetItem(name))

        if name == "Blur" or name == "Gaussian Blur" or name == "Median Blur":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("3"))
        elif name == "Otsu Binarization" or name == "Invert":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem(""))
        elif name == "Adaptive Thresholding":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("11, 2"))
        elif name == "Erosion" or name == "Dilation" or name == "Opening" or name == "Closing":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("3, 1"))
        elif name == "Edge":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("100, 200"))
        elif name == "Histogram Equalization":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("False, 1.8, 8"))
        elif name == "Crop":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("500, 500, 300"))
        elif name == "White Balance":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("False, 500, 500, 300"))
        elif name == "Adjust Brightness":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("20"))
        elif name == "Find Contours":
            self.CurrentProcessingTable.setItem(row_count, 1, QtWidgets.QTableWidgetItem("50"))

        self.CurrentProcessingTable.selectRow(row_count)
        self.output_image_idx = -1



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()