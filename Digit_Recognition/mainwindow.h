#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_table_cellClicked(int row, int column);

    void on_table_cellPressed(int row, int column);

    void on_pushButton_3_clicked();

    void on_pushButton_2_clicked();

    void on_table_cellActivated(int row, int column);

    void on_table_cellEntered(int row, int column);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
