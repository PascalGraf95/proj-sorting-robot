# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
    QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_sortingGui(object):
    def setupUi(self, sortingGui):
        if not sortingGui.objectName():
            sortingGui.setObjectName(u"sortingGui")
        sortingGui.resize(1662, 872)
        self.verticalLayoutWidget_3 = QWidget(sortingGui)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(20, 20, 686, 831))
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_connection = QLabel(self.verticalLayoutWidget_3)
        self.label_connection.setObjectName(u"label_connection")
        self.label_connection.setEnabled(True)
        font = QFont()
        font.setBold(True)
        self.label_connection.setFont(font)
        self.label_connection.setTextFormat(Qt.AutoText)
        self.label_connection.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_connection)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_hardware_status = QLabel(self.verticalLayoutWidget_3)
        self.label_hardware_status.setObjectName(u"label_hardware_status")
        self.label_hardware_status.setEnabled(True)
        self.label_hardware_status.setFont(font)
        self.label_hardware_status.setTextFormat(Qt.AutoText)
        self.label_hardware_status.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.label_hardware_status, 0, Qt.AlignLeft)

        self.label_status_roboter = QLabel(self.verticalLayoutWidget_3)
        self.label_status_roboter.setObjectName(u"label_status_roboter")
        self.label_status_roboter.setEnabled(True)
        self.label_status_roboter.setFont(font)
        self.label_status_roboter.setTextFormat(Qt.AutoText)
        self.label_status_roboter.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.label_status_roboter, 0, Qt.AlignRight)


        self.gridLayout.addLayout(self.horizontalLayout_6, 1, 1, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.button_dobot_homing = QPushButton(self.verticalLayoutWidget_3)
        self.button_dobot_homing.setObjectName(u"button_dobot_homing")

        self.horizontalLayout_4.addWidget(self.button_dobot_homing)

        self.button_dobot_standby = QPushButton(self.verticalLayoutWidget_3)
        self.button_dobot_standby.setObjectName(u"button_dobot_standby")

        self.horizontalLayout_4.addWidget(self.button_dobot_standby)


        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 1, 1, 1)

        self.button_connect_Hardware = QPushButton(self.verticalLayoutWidget_3)
        self.button_connect_Hardware.setObjectName(u"button_connect_Hardware")

        self.gridLayout.addWidget(self.button_connect_Hardware, 0, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_object_detection_method = QLabel(self.verticalLayoutWidget_3)
        self.label_object_detection_method.setObjectName(u"label_object_detection_method")
        self.label_object_detection_method.setEnabled(True)
        self.label_object_detection_method.setFont(font)
        self.label_object_detection_method.setTextFormat(Qt.AutoText)
        self.label_object_detection_method.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_object_detection_method)

        self.radio_classic = QRadioButton(self.verticalLayoutWidget_3)
        self.buttonGroup = QButtonGroup(sortingGui)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.addButton(self.radio_classic)
        self.radio_classic.setObjectName(u"radio_classic")

        self.horizontalLayout_3.addWidget(self.radio_classic, 0, Qt.AlignHCenter)

        self.radio_yoloV7 = QRadioButton(self.verticalLayoutWidget_3)
        self.buttonGroup.addButton(self.radio_yoloV7)
        self.radio_yoloV7.setObjectName(u"radio_yoloV7")
        self.radio_yoloV7.setChecked(True)

        self.horizontalLayout_3.addWidget(self.radio_yoloV7)


        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)

        self.label_status_camera = QLabel(self.verticalLayoutWidget_3)
        self.label_status_camera.setObjectName(u"label_status_camera")
        self.label_status_camera.setEnabled(True)
        self.label_status_camera.setFont(font)
        self.label_status_camera.setTextFormat(Qt.AutoText)
        self.label_status_camera.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_status_camera, 3, 1, 1, 1, Qt.AlignRight)

        self.verticalSpacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 4, 0, 1, 1)

        self.label_seperator_connection = QLabel(self.verticalLayoutWidget_3)
        self.label_seperator_connection.setObjectName(u"label_seperator_connection")
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_seperator_connection.sizePolicy().hasHeightForWidth())
        self.label_seperator_connection.setSizePolicy(sizePolicy)
        self.label_seperator_connection.setMinimumSize(QSize(30, 0))

        self.gridLayout.addWidget(self.label_seperator_connection, 4, 3, 1, 1)

        self.label_camera_connection = QLabel(self.verticalLayoutWidget_3)
        self.label_camera_connection.setObjectName(u"label_camera_connection")
        sizePolicy.setHeightForWidth(self.label_camera_connection.sizePolicy().hasHeightForWidth())
        self.label_camera_connection.setSizePolicy(sizePolicy)
        self.label_camera_connection.setMinimumSize(QSize(30, 0))
        self.label_camera_connection.setMaximumSize(QSize(30, 16777215))

        self.gridLayout.addWidget(self.label_camera_connection, 3, 3, 1, 1)

        self.label_dobot_connection = QLabel(self.verticalLayoutWidget_3)
        self.label_dobot_connection.setObjectName(u"label_dobot_connection")
        sizePolicy.setHeightForWidth(self.label_dobot_connection.sizePolicy().hasHeightForWidth())
        self.label_dobot_connection.setSizePolicy(sizePolicy)
        self.label_dobot_connection.setMinimumSize(QSize(30, 0))

        self.gridLayout.addWidget(self.label_dobot_connection, 1, 3, 1, 1)

        self.label_conveyor_connection = QLabel(self.verticalLayoutWidget_3)
        self.label_conveyor_connection.setObjectName(u"label_conveyor_connection")
        sizePolicy.setHeightForWidth(self.label_conveyor_connection.sizePolicy().hasHeightForWidth())
        self.label_conveyor_connection.setSizePolicy(sizePolicy)
        self.label_conveyor_connection.setMinimumSize(QSize(30, 0))

        self.gridLayout.addWidget(self.label_conveyor_connection, 2, 3, 1, 1)

        self.label_status_conveyor = QLabel(self.verticalLayoutWidget_3)
        self.label_status_conveyor.setObjectName(u"label_status_conveyor")
        self.label_status_conveyor.setEnabled(True)
        self.label_status_conveyor.setFont(font)
        self.label_status_conveyor.setTextFormat(Qt.AutoText)
        self.label_status_conveyor.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_status_conveyor, 2, 1, 1, 1, Qt.AlignRight)

        self.label_status_seperator = QLabel(self.verticalLayoutWidget_3)
        self.label_status_seperator.setObjectName(u"label_status_seperator")
        self.label_status_seperator.setEnabled(True)
        self.label_status_seperator.setFont(font)
        self.label_status_seperator.setTextFormat(Qt.AutoText)
        self.label_status_seperator.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_status_seperator, 4, 1, 1, 1, Qt.AlignRight)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_contour_method = QLabel(self.verticalLayoutWidget_3)
        self.label_contour_method.setObjectName(u"label_contour_method")
        self.label_contour_method.setEnabled(True)
        self.label_contour_method.setFont(font)
        self.label_contour_method.setTextFormat(Qt.AutoText)
        self.label_contour_method.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.label_contour_method, 0, Qt.AlignLeft)

        self.radio_contour_cut = QRadioButton(self.verticalLayoutWidget_3)
        self.radio_contour_cut.setObjectName(u"radio_contour_cut")

        self.horizontalLayout_5.addWidget(self.radio_contour_cut, 0, Qt.AlignLeft)


        self.gridLayout.addLayout(self.horizontalLayout_5, 2, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.verticalSpacer_3 = QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_3)

        self.label_phases = QLabel(self.verticalLayoutWidget_3)
        self.label_phases.setObjectName(u"label_phases")
        self.label_phases.setFont(font)
        self.label_phases.setTextFormat(Qt.AutoText)
        self.label_phases.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_phases)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_sorting = QLabel(self.verticalLayoutWidget_3)
        self.label_sorting.setObjectName(u"label_sorting")
        sizePolicy.setHeightForWidth(self.label_sorting.sizePolicy().hasHeightForWidth())
        self.label_sorting.setSizePolicy(sizePolicy)
        self.label_sorting.setMinimumSize(QSize(30, 0))

        self.gridLayout_2.addWidget(self.label_sorting, 6, 2, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_2, 8, 0, 1, 1)

        self.button_data_collection = QPushButton(self.verticalLayoutWidget_3)
        self.button_data_collection.setObjectName(u"button_data_collection")

        self.gridLayout_2.addWidget(self.button_data_collection, 1, 0, 1, 1)

        self.SortingType = QComboBox(self.verticalLayoutWidget_3)
        self.SortingType.addItem("")
        self.SortingType.addItem("")
        self.SortingType.addItem("")
        self.SortingType.setObjectName(u"SortingType")

        self.gridLayout_2.addWidget(self.SortingType, 1, 1, 1, 1)

        self.label_data_loading = QLabel(self.verticalLayoutWidget_3)
        self.label_data_loading.setObjectName(u"label_data_loading")
        sizePolicy.setHeightForWidth(self.label_data_loading.sizePolicy().hasHeightForWidth())
        self.label_data_loading.setSizePolicy(sizePolicy)
        self.label_data_loading.setMinimumSize(QSize(30, 0))

        self.gridLayout_2.addWidget(self.label_data_loading, 3, 2, 1, 1)

        self.button_load_and_cluster_data = QPushButton(self.verticalLayoutWidget_3)
        self.button_load_and_cluster_data.setObjectName(u"button_load_and_cluster_data")

        self.gridLayout_2.addWidget(self.button_load_and_cluster_data, 5, 0, 1, 1)

        self.button_stop = QPushButton(self.verticalLayoutWidget_3)
        self.button_stop.setObjectName(u"button_stop")

        self.gridLayout_2.addWidget(self.button_stop, 7, 0, 1, 1)

        self.button_sorting = QPushButton(self.verticalLayoutWidget_3)
        self.button_sorting.setObjectName(u"button_sorting")

        self.gridLayout_2.addWidget(self.button_sorting, 6, 0, 1, 1)

        self.label_data_collection = QLabel(self.verticalLayoutWidget_3)
        self.label_data_collection.setObjectName(u"label_data_collection")
        sizePolicy.setHeightForWidth(self.label_data_collection.sizePolicy().hasHeightForWidth())
        self.label_data_collection.setSizePolicy(sizePolicy)
        self.label_data_collection.setMinimumSize(QSize(30, 0))

        self.gridLayout_2.addWidget(self.label_data_collection, 1, 2, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.check_feature_area = QCheckBox(self.verticalLayoutWidget_3)
        self.check_feature_area.setObjectName(u"check_feature_area")

        self.gridLayout_3.addWidget(self.check_feature_area, 0, 0, 1, 1)

        self.check_feature_aspect = QCheckBox(self.verticalLayoutWidget_3)
        self.check_feature_aspect.setObjectName(u"check_feature_aspect")

        self.gridLayout_3.addWidget(self.check_feature_aspect, 0, 2, 1, 1)

        self.check_feature_color = QCheckBox(self.verticalLayoutWidget_3)
        self.check_feature_color.setObjectName(u"check_feature_color")

        self.gridLayout_3.addWidget(self.check_feature_color, 0, 1, 1, 1)

        self.check_feature_length = QCheckBox(self.verticalLayoutWidget_3)
        self.check_feature_length.setObjectName(u"check_feature_length")

        self.gridLayout_3.addWidget(self.check_feature_length, 0, 3, 1, 1)

        self.check_feature_hu = QCheckBox(self.verticalLayoutWidget_3)
        self.check_feature_hu.setObjectName(u"check_feature_hu")

        self.gridLayout_3.addWidget(self.check_feature_hu, 1, 0, 1, 1)

        self.check_feature_extent = QCheckBox(self.verticalLayoutWidget_3)
        self.check_feature_extent.setObjectName(u"check_feature_extent")

        self.gridLayout_3.addWidget(self.check_feature_extent, 1, 1, 1, 1)

        self.check_feature_solidity = QCheckBox(self.verticalLayoutWidget_3)
        self.check_feature_solidity.setObjectName(u"check_feature_solidity")

        self.gridLayout_3.addWidget(self.check_feature_solidity, 1, 2, 1, 1)


        self.horizontalLayout.addLayout(self.gridLayout_3)


        self.gridLayout_2.addLayout(self.horizontalLayout, 3, 1, 1, 1)

        self.label_clustering = QLabel(self.verticalLayoutWidget_3)
        self.label_clustering.setObjectName(u"label_clustering")
        sizePolicy.setHeightForWidth(self.label_clustering.sizePolicy().hasHeightForWidth())
        self.label_clustering.setSizePolicy(sizePolicy)
        self.label_clustering.setMinimumSize(QSize(30, 0))

        self.gridLayout_2.addWidget(self.label_clustering, 5, 2, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.radio_2d = QRadioButton(self.verticalLayoutWidget_3)
        self.buttonGroup_2 = QButtonGroup(sortingGui)
        self.buttonGroup_2.setObjectName(u"buttonGroup_2")
        self.buttonGroup_2.addButton(self.radio_2d)
        self.radio_2d.setObjectName(u"radio_2d")
        self.radio_2d.setChecked(True)

        self.horizontalLayout_2.addWidget(self.radio_2d)

        self.radio_3d = QRadioButton(self.verticalLayoutWidget_3)
        self.buttonGroup_2.addButton(self.radio_3d)
        self.radio_3d.setObjectName(u"radio_3d")

        self.horizontalLayout_2.addWidget(self.radio_3d)

        self.combo_clustering_method = QComboBox(self.verticalLayoutWidget_3)
        self.combo_clustering_method.addItem("")
        self.combo_clustering_method.addItem("")
        self.combo_clustering_method.addItem("")
        self.combo_clustering_method.addItem("")
        self.combo_clustering_method.setObjectName(u"combo_clustering_method")

        self.horizontalLayout_2.addWidget(self.combo_clustering_method)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 6, 1, 1, 1)

        self.Button_Start_User_Feedback = QPushButton(self.verticalLayoutWidget_3)
        self.Button_Start_User_Feedback.setObjectName(u"Button_Start_User_Feedback")

        self.gridLayout_2.addWidget(self.Button_Start_User_Feedback, 3, 0, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_2)

        self.label_status = QLabel(self.verticalLayoutWidget_3)
        self.label_status.setObjectName(u"label_status")
        font1 = QFont()
        font1.setPointSize(10)
        font1.setBold(True)
        font1.setItalic(False)
        font1.setStrikeOut(False)
        font1.setKerning(True)
        self.label_status.setFont(font1)
        self.label_status.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_status)


        self.verticalLayout_3.addLayout(self.verticalLayout)

        self.label_pca_cluster = QLabel(self.verticalLayoutWidget_3)
        self.label_pca_cluster.setObjectName(u"label_pca_cluster")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_pca_cluster.sizePolicy().hasHeightForWidth())
        self.label_pca_cluster.setSizePolicy(sizePolicy1)
        font2 = QFont()
        font2.setPointSize(10)
        font2.setBold(True)
        self.label_pca_cluster.setFont(font2)
        self.label_pca_cluster.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_pca_cluster)

        self.image_pca_cluster = QLabel(self.verticalLayoutWidget_3)
        self.image_pca_cluster.setObjectName(u"image_pca_cluster")
        sizePolicy2 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.image_pca_cluster.sizePolicy().hasHeightForWidth())
        self.image_pca_cluster.setSizePolicy(sizePolicy2)
        self.image_pca_cluster.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.image_pca_cluster)

        self.verticalLayoutWidget_2 = QWidget(sortingGui)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(720, 20, 911, 831))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.combo_cluster = QComboBox(self.verticalLayoutWidget_2)
        self.combo_cluster.setObjectName(u"combo_cluster")

        self.verticalLayout_2.addWidget(self.combo_cluster)

        self.label_cluster_examples = QLabel(self.verticalLayoutWidget_2)
        self.label_cluster_examples.setObjectName(u"label_cluster_examples")
        sizePolicy1.setHeightForWidth(self.label_cluster_examples.sizePolicy().hasHeightForWidth())
        self.label_cluster_examples.setSizePolicy(sizePolicy1)
        self.label_cluster_examples.setFont(font2)
        self.label_cluster_examples.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_cluster_examples)

        self.image_cluster_examples = QLabel(self.verticalLayoutWidget_2)
        self.image_cluster_examples.setObjectName(u"image_cluster_examples")
        self.image_cluster_examples.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.image_cluster_examples)

        self.label_live_conveyor = QLabel(self.verticalLayoutWidget_2)
        self.label_live_conveyor.setObjectName(u"label_live_conveyor")
        sizePolicy1.setHeightForWidth(self.label_live_conveyor.sizePolicy().hasHeightForWidth())
        self.label_live_conveyor.setSizePolicy(sizePolicy1)
        self.label_live_conveyor.setFont(font2)
        self.label_live_conveyor.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_live_conveyor)

        self.image_live_conveyor = QLabel(self.verticalLayoutWidget_2)
        self.image_live_conveyor.setObjectName(u"image_live_conveyor")
        self.image_live_conveyor.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.image_live_conveyor)


        self.retranslateUi(sortingGui)

        QMetaObject.connectSlotsByName(sortingGui)
    # setupUi

    def retranslateUi(self, sortingGui):
        sortingGui.setWindowTitle(QCoreApplication.translate("sortingGui", u"Autonomous Sorting", None))
        self.label_connection.setText(QCoreApplication.translate("sortingGui", u"Connection", None))
        self.label_hardware_status.setText(QCoreApplication.translate("sortingGui", u"Status:", None))
        self.label_status_roboter.setText(QCoreApplication.translate("sortingGui", u"Roboter:", None))
        self.button_dobot_homing.setText(QCoreApplication.translate("sortingGui", u"Homing Robot", None))
        self.button_dobot_standby.setText(QCoreApplication.translate("sortingGui", u"Standby", None))
        self.button_connect_Hardware.setText(QCoreApplication.translate("sortingGui", u"Connect Hardware", None))
        self.label_object_detection_method.setText(QCoreApplication.translate("sortingGui", u"Object Detection Method:", None))
        self.radio_classic.setText(QCoreApplication.translate("sortingGui", u"Classic", None))
        self.radio_yoloV7.setText(QCoreApplication.translate("sortingGui", u"YoloV7", None))
        self.label_status_camera.setText(QCoreApplication.translate("sortingGui", u"Camera:", None))
        self.label_seperator_connection.setText("")
        self.label_camera_connection.setText("")
        self.label_dobot_connection.setText("")
        self.label_conveyor_connection.setText("")
        self.label_status_conveyor.setText(QCoreApplication.translate("sortingGui", u"Conveyor Belt:", None))
        self.label_status_seperator.setText(QCoreApplication.translate("sortingGui", u"Seperator:", None))
        self.label_contour_method.setText(QCoreApplication.translate("sortingGui", u"Cut Contour", None))
        self.radio_contour_cut.setText(QCoreApplication.translate("sortingGui", u"True", None))
        self.label_phases.setText(QCoreApplication.translate("sortingGui", u"Phases", None))
        self.label_sorting.setText("")
        self.button_data_collection.setText(QCoreApplication.translate("sortingGui", u"Data Collection Phase", None))
        self.SortingType.setItemText(0, QCoreApplication.translate("sortingGui", u"Manually Select Data", None))
        self.SortingType.setItemText(1, QCoreApplication.translate("sortingGui", u"Autoencoder", None))
        self.SortingType.setItemText(2, QCoreApplication.translate("sortingGui", u"Transformer", None))

        self.label_data_loading.setText("")
        self.button_load_and_cluster_data.setText(QCoreApplication.translate("sortingGui", u"Load and Cluster Data", None))
        self.button_stop.setText(QCoreApplication.translate("sortingGui", u"Stop", None))
        self.button_sorting.setText(QCoreApplication.translate("sortingGui", u"Start Sorting", None))
        self.label_data_collection.setText("")
        self.check_feature_area.setText(QCoreApplication.translate("sortingGui", u"Area", None))
        self.check_feature_aspect.setText(QCoreApplication.translate("sortingGui", u"Aspect", None))
        self.check_feature_color.setText(QCoreApplication.translate("sortingGui", u"Color", None))
        self.check_feature_length.setText(QCoreApplication.translate("sortingGui", u"Length", None))
        self.check_feature_hu.setText(QCoreApplication.translate("sortingGui", u"Hu", None))
        self.check_feature_extent.setText(QCoreApplication.translate("sortingGui", u"Extent", None))
        self.check_feature_solidity.setText(QCoreApplication.translate("sortingGui", u"Solidity", None))
        self.label_clustering.setText("")
        self.radio_2d.setText(QCoreApplication.translate("sortingGui", u"2D", None))
        self.radio_3d.setText(QCoreApplication.translate("sortingGui", u"3D", None))
        self.combo_clustering_method.setItemText(0, QCoreApplication.translate("sortingGui", u"KMeans", None))
        self.combo_clustering_method.setItemText(1, QCoreApplication.translate("sortingGui", u"DBSCAN", None))
        self.combo_clustering_method.setItemText(2, QCoreApplication.translate("sortingGui", u"MeanShift", None))
        self.combo_clustering_method.setItemText(3, QCoreApplication.translate("sortingGui", u"Agglomerative", None))

        self.Button_Start_User_Feedback.setText(QCoreApplication.translate("sortingGui", u"Start User Feedback", None))
        self.label_status.setText(QCoreApplication.translate("sortingGui", u"Current Status", None))
        self.label_pca_cluster.setText(QCoreApplication.translate("sortingGui", u"PCA Cluster Preview", None))
        self.image_pca_cluster.setText(QCoreApplication.translate("sortingGui", u"Live PCA Cluster Image", None))
        self.label_cluster_examples.setText(QCoreApplication.translate("sortingGui", u"Cluster Image Preview", None))
        self.image_cluster_examples.setText(QCoreApplication.translate("sortingGui", u"Live Cluster Examples", None))
        self.label_live_conveyor.setText(QCoreApplication.translate("sortingGui", u"Live Conveyor Image", None))
        self.image_live_conveyor.setText(QCoreApplication.translate("sortingGui", u"Live Conveyor Belt Image", None))
    # retranslateUi

