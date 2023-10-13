#include "sortinggui.h"
#include "./ui_sortinggui.h"

SortingGUI::SortingGUI(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::SortingGUI)
{
    ui->setupUi(this);
}

SortingGUI::~SortingGUI()
{
    delete ui;
}

