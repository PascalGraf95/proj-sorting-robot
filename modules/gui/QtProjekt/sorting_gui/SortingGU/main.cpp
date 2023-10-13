#include "sortinggui.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    SortingGUI w;
    w.show();
    return a.exec();
}
