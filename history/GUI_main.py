import GUI2 as GUI
import convLSTM
import similarityRUL
import arima
import arima_example
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
import sys
import re


class My_MainWindow(GUI.Ui_MainWindow):
    def __init__(self, MainWindow):
        super(My_MainWindow, self).__init__()
        super().setupUi(MainWindow)
        self.model = QtWidgets.QFileSystemModel()
        self.load_dir()
        self.slot_init()
        self.qss()
        self.name = -1
        self.result = ""
        self.path = ""
        self.algorithm = 'LSTM'
        self.data = '训练数据'
        self.circle = '周期1'
        self.checkBox_list = [self.checkBox_1, self.checkBox_2, self.checkBox_3, self.checkBox_4, self.checkBox_5
                              , self.checkBox_6, self.checkBox_7, self.checkBox_8, self.checkBox_9, self.checkBox_10
                              , self.checkBox_11, self.checkBox_12, self.checkBox_13, self.checkBox_14, self.checkBox_15
                              , self.checkBox_16, self.checkBox_17, self.checkBox_18, self.checkBox_19, self.checkBox_20
                              , self.checkBox_21]
        self.sensor_list = []

    def slot_init(self):
        self.Tree.doubleClicked.connect(self.tree_click)
        self.operate.clicked.connect(self.operate_algorithm)
        self.confirm_sensor.clicked.connect(self.arguments_convey)
        self.operate_testdata.clicked.connect(self.arima_test)

    def qss(self):
        self.Tree.setColumnWidth(0, 230)
        self.output_figure.setScaledContents(True)
        self.output_figure2.setScaledContents(True)
        self.output_figure.setStyleSheet('''QLabel{border: 1px solid black;}''')
        self.output_figure2.setStyleSheet('''QLabel{border: 1px solid black;}''')

    def load_dir(self):
        self.model.setRootPath("./datafolder/")
        self.Tree.setModel(self.model)
        self.Tree.setRootIndex(self.model.index(r'./datafolder/'))

    def tree_click(self, Qmodelidx):
        if ('.csv' in self.model.fileName(Qmodelidx)) or ('.txt' in self.model.fileName(Qmodelidx)):
            self.DataFileName = self.model.filePath(Qmodelidx)
            print(self.DataFileName)
            self.path = self.DataFileName
            self.name = int(self.DataFileName[-5])

    def operate_algorithm(self):
        self.algorithm = self.choose_algorithm.currentText()
        self.data = self.choose_data.currentText()


        if(self.algorithm != "相似模型" and self.data == "训练数据"):
            if(self.name == -1):
                self.error_output.clear()
                self.show_error("请先选择文件，再点击运行")
            elif(self.name != 1 and self.name != 2 and self.name != 3 and self.name != 4):
                self.error_output.clear()
                self.show_error("文件名字有误，无法判断数据形式")
            else:
                self.error_output.clear()
                self.output_widget.clear()
                self.show_message("testing...")
                lstm = convLSTM.ConvLSTM()
                if (self.algorithm[-1] == 'D'):
                    self.result = lstm.test(self.name, self.algorithm[:-3], 1)
                else:
                    self.result = lstm.test(self.name, self.algorithm, 0)
                self.output_widget.clear()
                self.show_message(self.result)
                if (self.algorithm[-1] == 'D'):
                    lstm.draw_image(self.name, self.algorithm[:-3], 1)
                else:
                    lstm.draw_image(self.name, self.algorithm, 0)
                self.output_figure.setPixmap(QPixmap("./FD00"+str(self.name)+"_"+self.algorithm+".png"))

        elif (self.algorithm != "相似模型" and self.data == "真实数据"):
            if (self.name == -1):
                self.error_output.clear()
                self.show_error("请先选择文件，再点击运行")
            elif (self.name != 1 and self.name != 2 and self.name != 3 and self.name != 4):
                self.error_output.clear()
                self.show_error("文件名字有误，无法判断数据形式")
            else:
                self.error_output.clear()
                self.output_widget.clear()
                self.show_message("predicting...")
                lstm = convLSTM.ConvLSTM()
                if (self.algorithm[-1] == 'D'):
                    self.result = lstm.real(self.name, self.algorithm[:-3], 1)
                else:
                    self.result = lstm.real(self.name, self.algorithm, 0)
                self.output_widget.clear()
                self.show_message(self.result)

        elif(self.algorithm == "相似模型" and self.data == "训练数据"):
            if (self.name == -1):
                self.error_output.clear()
                self.show_error("请先选择文件，再点击运行")
            elif (self.name != 1 and self.name != 2 and self.name != 3 and self.name != 4):
                self.error_output.clear()
                self.show_error("文件名字有误，无法判断数据形式")
            elif (self.name == 3 or self.name == 4):
                self.error_output.clear()
                self.show_error("该数据集不适合用相似匹配法进行预测！")
            else:
                self.error_output.clear()
                self.output_widget.clear()
                self.show_message("testing...")
                similar = similarityRUL.similarity()
                if(self.name == 1):
                    n_flights = 100
                else:
                    n_flights = 259
                test_set = self.path
                RUL_set = "./datafolder/RUL_FD00" + str(self.name) + ".txt"
                self.result = similar.test(n_flights, test_set, RUL_set)
                self.output_widget.clear()
                self.show_message(self.result)
                self.output_figure.setPixmap(QPixmap("./similarity.png"))

        elif (self.algorithm == "相似模型" and self.data == "真实数据"):
            if (self.name == -1):
                self.error_output.clear()
                self.show_error("请先选择文件，再点击运行")
            elif (self.name != 1 and self.name != 2 and self.name != 3 and self.name != 4):
                self.error_output.clear()
                self.show_error("文件名字有误，无法判断数据形式")
            elif(self.name == 3 or self.name == 4):
                self.error_output.clear()
                self.show_error("该数据集不适合用相似匹配法进行预测！")
            else:
                self.error_output.clear()
                self.output_widget.clear()
                self.show_message("predicting...")
                similar = similarityRUL.similarity()
                if (self.name == 1):
                    n_flights = 100
                else:
                    n_flights = 259
                real_set = self.path
                self.result = similar.real(n_flights, real_set)
                self.output_widget.clear()
                self.show_message(self.result)

        else:
            self.error_output.clear()
            self.show_error("选择错误")

    def show_message(self, mystr):
        self.output_widget.append(mystr)
        self.cursor = self.output_widget.textCursor()
        self.output_widget.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

    def show_message2(self, mystr):
        self.output_widget2.append(mystr)
        self.cursor = self.output_widget2.textCursor()
        self.output_widget2.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

    def show_error(self, mystr):
        self.error_output.append(mystr)
        self.cursor = self.error_output.textCursor()
        self.error_output.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

    def show_error2(self, mystr):
        self.error_output2.append(mystr)
        self.cursor = self.error_output2.textCursor()
        self.error_output2.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

    def arguments_convey(self):
        # print("hello")
        self.circle = self.select_circle.currentText()
        self.sensor_list = []
        for item in self.checkBox_list:
            if item.isChecked():
                self.sensor_list.append(int(re.findall(r"\d+",item.text())[0]))
        # print(self.circle)
        # print(self.sensor_list)
        if(self.sensor_list == []):
            self.error_output2.clear()
            self.output_widget2.clear()
            self.show_message2("请先选择传感器")

        else:
            self.error_output2.clear()
            self.output_widget2.clear()
            self.show_message2("predicting...")
            ari = arima.arima()
            output, error, predict, available = ari.predict(self.sensor_list, int(self.circle[-1]), self.path)
            ari.draw_image(self.name, available, predict, int(self.circle[-1]))
            self.output_widget2.clear()
            self.show_message2(output)
            self.show_error2(error)
            self.output_figure2.setPixmap(QPixmap("./arima.png"))

    def arima_test(self):
        self.error_output2.clear()
        self.output_widget2.clear()
        self.show_message2("predicting...")
        message = arima_example.example()
        self.output_widget2.clear()
        self.show_message2(message)
        self.output_figure2.setPixmap(QPixmap("./arima_example.png"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = My_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())