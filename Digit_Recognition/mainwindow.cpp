#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "mnistrec.h"
#include <random>
#include <Eigen/Eigen>

Eigen::VectorXf values(784);

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->table->setRowCount(28);
    ui->table->setColumnCount(28);
    ui->table->verticalHeader()->setVisible(false);
    ui->table->horizontalHeader()->setVisible(false);
    ui->table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->table->setSelectionMode(QAbstractItemView::NoSelection);
    for (int i = 0; i < 28; i++ ) {
       ui->table->setColumnWidth(i,20);
       ui->table->setRowHeight(i,20);
    }
    for (int i = 0; i < 28; i++ ) {
        for (int j = 0; j < 28; j++) {
             ui->table->setItem(i, j, new QTableWidgetItem(""));
             ui->table->item(i, j)->setBackground(QColor(255, 255,255  ,255));
             values[i*28+j] = 0;
          //   count++;
        }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    using namespace MLP;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(1, 10000);
    int rand = distr(eng);
    std::cout<<"Rand: "<<rand<<"\n";
    MLP::Result res = MLP::test(rand);
   ui->label->setText("Prediction: "+QString::number(res.predict) + "\n\n"+"R/A Score: \n" + QString::number(res.pred_pers)+"\nW/A Score: \n" + QString::number(-res.error));
   int count = 0;

   for (int i = 0; i < 28; i++ ) {
       for (int j = 0; j < 28; j++) {
            ui->table->setItem(i, j, new QTableWidgetItem(""));
            ui->table->item(i, j)->setBackground(QColor(0, 0, 0,(int)(255*res.image[count])));
            values[count] = res.image[count];
            count++;
       }
   }
}

void MainWindow::on_table_cellClicked(int row, int column)
{
    values[28*row+column] = 1;
    ui->table->item(row, column)->setBackground(QColor(0, 0, 0,255));
}

void MainWindow::on_table_cellPressed(int row, int column)
{

}

void MainWindow::on_pushButton_3_clicked()
{

    for (int i = 0; i < 28; i++ ) {
        for (int j = 0; j < 28; j++) {
             ui->table->setItem(i, j, new QTableWidgetItem(""));
             ui->table->item(i, j)->setBackground(QColor(255, 255,255  ,255));
             values[i*28+j] = 0;
          //   count++;
        }
    }

}

void MainWindow::on_pushButton_2_clicked()
{
    using namespace MLP;

    MLP::Result res = MLP::test_custom(values);
   ui->label->setText("Prediction: "+QString::number(res.predict) + "\n\n"+"R/A Score: \n" + QString::number(res.pred_pers)+"\nW/A Score: \n" + QString::number(-res.error));
   int count = 0;

   for (int i = 0; i < 28; i++ ) {
       for (int j = 0; j < 28; j++) {
            ui->table->setItem(i, j, new QTableWidgetItem(""));
            ui->table->item(i, j)->setBackground(QColor(0, 0, 0,(int)(255*res.image[count])));
            values[count] = res.image[count];
            count++;
       }
   }
}

void MainWindow::on_table_cellActivated(int row, int column)
{
    values[28*row+column] = 1;
    ui->table->item(row, column)->setBackground(QColor(0, 0, 0,255));

}

void MainWindow::on_table_cellEntered(int row, int column)
{
    values[28*row+column] = 1;
    ui->table->item(row, column)->setBackground(QColor(0, 0, 0,255));
}
