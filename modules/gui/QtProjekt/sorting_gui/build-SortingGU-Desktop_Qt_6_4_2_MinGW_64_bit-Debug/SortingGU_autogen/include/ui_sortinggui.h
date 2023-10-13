/********************************************************************************
** Form generated from reading UI file 'sortinggui.ui'
**
** Created by: Qt User Interface Compiler version 6.4.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SORTINGGUI_H
#define UI_SORTINGGUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SortingGUI
{
public:
    QWidget *centralwidget;
    QLabel *label;
    QLabel *label_2;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QLabel *label_connection;
    QGridLayout *gridLayout;
    QPushButton *button_connect_conveyor;
    QLabel *label_seperator_connection;
    QPushButton *button_connect_seperator;
    QLabel *label_dobot_connection;
    QLabel *label_conveyor_connection;
    QSpacerItem *verticalSpacer;
    QPushButton *button_connect_dobot;
    QPushButton *button_connect_camera;
    QLabel *label_camera_connection;
    QPushButton *button_dobot_homing;
    QSpacerItem *verticalSpacer_3;
    QLabel *label_phases;
    QGridLayout *gridLayout_2;
    QSpacerItem *verticalSpacer_2;
    QPushButton *button_sorting;
    QHBoxLayout *horizontalLayout_2;
    QRadioButton *radioButton;
    QRadioButton *radioButton_2;
    QPushButton *button_data_collection;
    QPushButton *button_stop;
    QPushButton *button_clustering;
    QHBoxLayout *horizontalLayout;
    QCheckBox *check_feature_area;
    QCheckBox *check_feature_color;
    QCheckBox *check_feature_aspect;
    QCheckBox *check_feature_length;
    QCheckBox *check_feature_hu;
    QPushButton *button_load_data;
    QLabel *label_clustering;
    QLabel *label_data_collection;
    QLabel *label_sorting;
    QLabel *label_5;
    QLabel *label_status;
    QWidget *verticalLayoutWidget_2;
    QVBoxLayout *verticalLayout_2;
    QComboBox *combo_cluster;
    QLabel *image_cluster_examples;
    QLabel *image_live_conveyor;
    QLabel *image_pca_cluster;
    QLabel *label_3;
    QLabel *label_4;
    QStatusBar *statusbar;
    QMenuBar *menubar;

    void setupUi(QMainWindow *SortingGUI)
    {
        if (SortingGUI->objectName().isEmpty())
            SortingGUI->setObjectName("SortingGUI");
        SortingGUI->resize(1494, 910);
        SortingGUI->setMinimumSize(QSize(1280, 720));
        SortingGUI->setMaximumSize(QSize(1920, 1080));
        centralwidget = new QWidget(SortingGUI);
        centralwidget->setObjectName("centralwidget");
        label = new QLabel(centralwidget);
        label->setObjectName("label");
        label->setGeometry(QRect(280, 10, 161, 71));
        label->setPixmap(QPixmap(QString::fromUtf8("C:/Users/Drumm/OneDrive/Desktop/70f3d6f3daf5b559-83a5ca4c96b0-HHN_Logo_D_RGB_300.png")));
        label->setScaledContents(true);
        label_2 = new QLabel(centralwidget);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(40, 20, 221, 41));
        label_2->setPixmap(QPixmap(QString::fromUtf8("C:/Users/Drumm/OneDrive/Desktop/c30941a52d7c68ed-86a573a49101-Zentrum-f-r-Maschinelles-Lernen-ZML.png")));
        label_2->setScaledContents(true);
        verticalLayoutWidget = new QWidget(centralwidget);
        verticalLayoutWidget->setObjectName("verticalLayoutWidget");
        verticalLayoutWidget->setGeometry(QRect(10, 80, 501, 378));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        label_connection = new QLabel(verticalLayoutWidget);
        label_connection->setObjectName("label_connection");
        label_connection->setEnabled(true);
        QFont font;
        font.setBold(true);
        label_connection->setFont(font);
        label_connection->setTextFormat(Qt::AutoText);
        label_connection->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label_connection);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName("gridLayout");
        button_connect_conveyor = new QPushButton(verticalLayoutWidget);
        button_connect_conveyor->setObjectName("button_connect_conveyor");

        gridLayout->addWidget(button_connect_conveyor, 1, 0, 1, 1);

        label_seperator_connection = new QLabel(verticalLayoutWidget);
        label_seperator_connection->setObjectName("label_seperator_connection");

        gridLayout->addWidget(label_seperator_connection, 2, 2, 1, 1);

        button_connect_seperator = new QPushButton(verticalLayoutWidget);
        button_connect_seperator->setObjectName("button_connect_seperator");

        gridLayout->addWidget(button_connect_seperator, 2, 0, 1, 1);

        label_dobot_connection = new QLabel(verticalLayoutWidget);
        label_dobot_connection->setObjectName("label_dobot_connection");

        gridLayout->addWidget(label_dobot_connection, 0, 2, 1, 1);

        label_conveyor_connection = new QLabel(verticalLayoutWidget);
        label_conveyor_connection->setObjectName("label_conveyor_connection");

        gridLayout->addWidget(label_conveyor_connection, 1, 2, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 4, 0, 1, 1);

        button_connect_dobot = new QPushButton(verticalLayoutWidget);
        button_connect_dobot->setObjectName("button_connect_dobot");

        gridLayout->addWidget(button_connect_dobot, 0, 0, 1, 1);

        button_connect_camera = new QPushButton(verticalLayoutWidget);
        button_connect_camera->setObjectName("button_connect_camera");

        gridLayout->addWidget(button_connect_camera, 3, 0, 1, 1);

        label_camera_connection = new QLabel(verticalLayoutWidget);
        label_camera_connection->setObjectName("label_camera_connection");
        label_camera_connection->setMaximumSize(QSize(30, 16777215));

        gridLayout->addWidget(label_camera_connection, 3, 2, 1, 1);

        button_dobot_homing = new QPushButton(verticalLayoutWidget);
        button_dobot_homing->setObjectName("button_dobot_homing");

        gridLayout->addWidget(button_dobot_homing, 0, 1, 1, 1);


        verticalLayout->addLayout(gridLayout);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_3);

        label_phases = new QLabel(verticalLayoutWidget);
        label_phases->setObjectName("label_phases");
        label_phases->setFont(font);
        label_phases->setTextFormat(Qt::AutoText);
        label_phases->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label_phases);

        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName("gridLayout_2");
        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer_2, 8, 0, 1, 1);

        button_sorting = new QPushButton(verticalLayoutWidget);
        button_sorting->setObjectName("button_sorting");

        gridLayout_2->addWidget(button_sorting, 6, 0, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        radioButton = new QRadioButton(verticalLayoutWidget);
        radioButton->setObjectName("radioButton");

        horizontalLayout_2->addWidget(radioButton);

        radioButton_2 = new QRadioButton(verticalLayoutWidget);
        radioButton_2->setObjectName("radioButton_2");

        horizontalLayout_2->addWidget(radioButton_2);


        gridLayout_2->addLayout(horizontalLayout_2, 5, 1, 1, 1);

        button_data_collection = new QPushButton(verticalLayoutWidget);
        button_data_collection->setObjectName("button_data_collection");

        gridLayout_2->addWidget(button_data_collection, 1, 0, 1, 1);

        button_stop = new QPushButton(verticalLayoutWidget);
        button_stop->setObjectName("button_stop");

        gridLayout_2->addWidget(button_stop, 7, 0, 1, 1);

        button_clustering = new QPushButton(verticalLayoutWidget);
        button_clustering->setObjectName("button_clustering");

        gridLayout_2->addWidget(button_clustering, 5, 0, 1, 1);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName("horizontalLayout");
        check_feature_area = new QCheckBox(verticalLayoutWidget);
        check_feature_area->setObjectName("check_feature_area");

        horizontalLayout->addWidget(check_feature_area);

        check_feature_color = new QCheckBox(verticalLayoutWidget);
        check_feature_color->setObjectName("check_feature_color");

        horizontalLayout->addWidget(check_feature_color);

        check_feature_aspect = new QCheckBox(verticalLayoutWidget);
        check_feature_aspect->setObjectName("check_feature_aspect");

        horizontalLayout->addWidget(check_feature_aspect);

        check_feature_length = new QCheckBox(verticalLayoutWidget);
        check_feature_length->setObjectName("check_feature_length");

        horizontalLayout->addWidget(check_feature_length);

        check_feature_hu = new QCheckBox(verticalLayoutWidget);
        check_feature_hu->setObjectName("check_feature_hu");

        horizontalLayout->addWidget(check_feature_hu);


        gridLayout_2->addLayout(horizontalLayout, 3, 1, 1, 1);

        button_load_data = new QPushButton(verticalLayoutWidget);
        button_load_data->setObjectName("button_load_data");

        gridLayout_2->addWidget(button_load_data, 3, 0, 1, 1);

        label_clustering = new QLabel(verticalLayoutWidget);
        label_clustering->setObjectName("label_clustering");

        gridLayout_2->addWidget(label_clustering, 5, 2, 1, 1);

        label_data_collection = new QLabel(verticalLayoutWidget);
        label_data_collection->setObjectName("label_data_collection");

        gridLayout_2->addWidget(label_data_collection, 1, 2, 1, 1);

        label_sorting = new QLabel(verticalLayoutWidget);
        label_sorting->setObjectName("label_sorting");

        gridLayout_2->addWidget(label_sorting, 6, 2, 1, 1);

        label_5 = new QLabel(verticalLayoutWidget);
        label_5->setObjectName("label_5");

        gridLayout_2->addWidget(label_5, 3, 2, 1, 1);


        verticalLayout->addLayout(gridLayout_2);

        label_status = new QLabel(verticalLayoutWidget);
        label_status->setObjectName("label_status");
        QFont font1;
        font1.setPointSize(10);
        font1.setBold(true);
        font1.setItalic(false);
        font1.setStrikeOut(false);
        font1.setKerning(true);
        label_status->setFont(font1);
        label_status->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label_status);

        verticalLayoutWidget_2 = new QWidget(centralwidget);
        verticalLayoutWidget_2->setObjectName("verticalLayoutWidget_2");
        verticalLayoutWidget_2->setGeometry(QRect(640, 120, 711, 651));
        verticalLayout_2 = new QVBoxLayout(verticalLayoutWidget_2);
        verticalLayout_2->setObjectName("verticalLayout_2");
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        combo_cluster = new QComboBox(verticalLayoutWidget_2);
        combo_cluster->setObjectName("combo_cluster");

        verticalLayout_2->addWidget(combo_cluster);

        image_cluster_examples = new QLabel(verticalLayoutWidget_2);
        image_cluster_examples->setObjectName("image_cluster_examples");
        image_cluster_examples->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(image_cluster_examples);

        image_live_conveyor = new QLabel(verticalLayoutWidget_2);
        image_live_conveyor->setObjectName("image_live_conveyor");
        image_live_conveyor->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(image_live_conveyor);

        image_pca_cluster = new QLabel(centralwidget);
        image_pca_cluster->setObjectName("image_pca_cluster");
        image_pca_cluster->setGeometry(QRect(40, 470, 471, 291));
        image_pca_cluster->setAlignment(Qt::AlignCenter);
        label_3 = new QLabel(centralwidget);
        label_3->setObjectName("label_3");
        label_3->setGeometry(QRect(600, 20, 261, 61));
        label_4 = new QLabel(centralwidget);
        label_4->setObjectName("label_4");
        label_4->setGeometry(QRect(1140, 810, 241, 51));
        SortingGUI->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(SortingGUI);
        statusbar->setObjectName("statusbar");
        SortingGUI->setStatusBar(statusbar);
        menubar = new QMenuBar(SortingGUI);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 1494, 22));
        SortingGUI->setMenuBar(menubar);

        retranslateUi(SortingGUI);

        QMetaObject::connectSlotsByName(SortingGUI);
    } // setupUi

    void retranslateUi(QMainWindow *SortingGUI)
    {
        SortingGUI->setWindowTitle(QCoreApplication::translate("SortingGUI", "SortingGUI", nullptr));
        label->setText(QString());
        label_2->setText(QString());
        label_connection->setText(QCoreApplication::translate("SortingGUI", "Connection", nullptr));
        button_connect_conveyor->setText(QCoreApplication::translate("SortingGUI", "Connect Conveyor Belt", nullptr));
        label_seperator_connection->setText(QString());
        button_connect_seperator->setText(QCoreApplication::translate("SortingGUI", "Connect Separator", nullptr));
        label_dobot_connection->setText(QString());
        label_conveyor_connection->setText(QString());
        button_connect_dobot->setText(QCoreApplication::translate("SortingGUI", "Connect DoBot", nullptr));
        button_connect_camera->setText(QCoreApplication::translate("SortingGUI", "Connect Camera", nullptr));
        label_camera_connection->setText(QString());
        button_dobot_homing->setText(QCoreApplication::translate("SortingGUI", "Homing", nullptr));
        label_phases->setText(QCoreApplication::translate("SortingGUI", "Phases", nullptr));
        button_sorting->setText(QCoreApplication::translate("SortingGUI", "Sorting Phase", nullptr));
        radioButton->setText(QCoreApplication::translate("SortingGUI", "2D", nullptr));
        radioButton_2->setText(QCoreApplication::translate("SortingGUI", "3D", nullptr));
        button_data_collection->setText(QCoreApplication::translate("SortingGUI", "Data Collection Phase", nullptr));
        button_stop->setText(QCoreApplication::translate("SortingGUI", "Stop", nullptr));
        button_clustering->setText(QCoreApplication::translate("SortingGUI", "Clustering Phase", nullptr));
        check_feature_area->setText(QCoreApplication::translate("SortingGUI", "Area", nullptr));
        check_feature_color->setText(QCoreApplication::translate("SortingGUI", "Color", nullptr));
        check_feature_aspect->setText(QCoreApplication::translate("SortingGUI", "Aspect", nullptr));
        check_feature_length->setText(QCoreApplication::translate("SortingGUI", "Length", nullptr));
        check_feature_hu->setText(QCoreApplication::translate("SortingGUI", "Hu", nullptr));
        button_load_data->setText(QCoreApplication::translate("SortingGUI", "Load Data and Select Features", nullptr));
        label_clustering->setText(QCoreApplication::translate("SortingGUI", "x", nullptr));
        label_data_collection->setText(QCoreApplication::translate("SortingGUI", "x", nullptr));
        label_sorting->setText(QCoreApplication::translate("SortingGUI", "x", nullptr));
        label_5->setText(QCoreApplication::translate("SortingGUI", "x", nullptr));
        label_status->setText(QCoreApplication::translate("SortingGUI", "Current Status", nullptr));
        image_cluster_examples->setText(QCoreApplication::translate("SortingGUI", "Live Cluster Examples", nullptr));
        image_live_conveyor->setText(QCoreApplication::translate("SortingGUI", "Live Conveyor Belt Image", nullptr));
        image_pca_cluster->setText(QCoreApplication::translate("SortingGUI", "Live PCA Cluster Image", nullptr));
        label_3->setText(QCoreApplication::translate("SortingGUI", "Headline", nullptr));
        label_4->setText(QCoreApplication::translate("SortingGUI", "Footer", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SortingGUI: public Ui_SortingGUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SORTINGGUI_H
