#ifndef SORTINGGUI_H
#define SORTINGGUI_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class SortingGUI; }
QT_END_NAMESPACE

class SortingGUI : public QMainWindow
{
    Q_OBJECT

public:
    SortingGUI(QWidget *parent = nullptr);
    ~SortingGUI();

private:
    Ui::SortingGUI *ui;
};
#endif // SORTINGGUI_H
