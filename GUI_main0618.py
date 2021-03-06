from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import GUI0629 as GUI
import convLSTM
import arima_example
import similarityRUL
import arima
import StressExperience
import nn
import sys
import re
import math
import numpy as np


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
        self.tab3operate.clicked.connect(self.Stress_expe)

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
                self.output_figure.setPixmap(QPixmap("./result_img/FD00"+str(self.name)+"_"+self.algorithm+".png"))

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
                # similar = similarityRUL.similarity()
                if(self.name == 1):
                    n_flights = 100
                else:
                    n_flights = 259
                test_set = self.path
                RUL_set = "./datafolder/RUL_FD00" + str(self.name) + ".txt"
                # self.result = similar.test(n_flights, test_set, RUL_set)
                self.SimilarTrain = SimilarTrainThread(n_flights, test_set, RUL_set)
                self.SimilarTrain.result_signal.connect(self.SimilarTrainOutput)
                self.SimilarTrain.start()
                # print(type(self.result))


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

    def SimilarTrainOutput(self, result):
        self.output_widget.clear()
        self.show_message(result)
        self.output_figure.setPixmap(QPixmap("./result_img/similarity.png"))

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
            self.output_figure2.setPixmap(QPixmap("./result_img/arima.png"))

    def arima_test(self):
        self.error_output2.clear()
        self.output_widget2.clear()
        self.show_message2("predicting...")
        message = arima_example.example()
        self.output_widget2.clear()
        self.show_message2(message)
        self.output_figure2.setPixmap(QPixmap("./result_img/arima_example.png"))

    def Stress_expe(self):
        self.temperture = int(self.temperaturetextEdit.document().toPlainText())
        # print(self.temperture)
        reliability = self.ReliabilitytextEdit.document().toPlainText()
        if reliability == '1/e':
            reliability = 1/math.e
        else:
            reliability = float(reliability)
        print(reliability)
        if self.choose_model.currentText() == '阿伦尼斯模型':
            model = 1
        elif self.choose_model.currentText() == '逆幂率模型':
            model = 2
        elif self.choose_model.currentText() == '神经网络模型':
            model = 3
        self.experience = ExperienceThread(self.path, self.temperture, reliability, model)
        self.experience.result_signal.connect(self.experience_output)
        self.experience.nn_result_signal.connect(self.nn_experience_output)
        self.experience.start()
        # pred, result_img_path = StressExperience.load_data(self.path, temperture, reliability, model)


    def experience_output(self, pred, result_img_path):
        self.tab3_output_widget.append('在' + str(self.temperture) + '摄氏度的工作环境下')
        self.tab3_output_widget.append('陀螺仪寿命为' + str(pred) + '小时')
        self.tab3_output_figure.setPixmap(QPixmap(result_img_path))

    def nn_experience_output(self, pred):
        self.tab3_output_widget.append('在' + str(self.temperture) + '摄氏度的工作环境下')
        self.tab3_output_widget.append('陀螺仪寿命为' + str(pred) + '小时')


#相似模型训练线程
class SimilarTrainThread(QThread):
    result_signal = pyqtSignal(str)
    def __init__(self, n_flights, test_set, RUL_set):
        super(SimilarTrainThread, self).__init__()
        self.n_flights = n_flights
        self.test_set = test_set
        self.RUL_set = RUL_set

    def run(self):
        similar = similarityRUL.similarity()
        result = similar.test(self.n_flights, self.test_set, self.RUL_set)
        self.result_signal.emit(result)

#步降应力试验线程
class ExperienceThread(QThread):
    result_signal = pyqtSignal(float, str)
    nn_result_signal = pyqtSignal(float)
    def __init__(self, path, temperture, reliability, model):
        super(ExperienceThread, self).__init__()
        self.path = path
        self.temperture = temperture
        self.reliability = reliability
        self.model = model

    def run(self):
        if self.model ==1 or self.model ==2:
            pred, result_img_path = StressExperience.load_data(self.path, self.temperture, self.reliability, self.model)
            self.result_signal.emit(pred, result_img_path)
        elif self.model == 3:
            train = False
            inputdata = [[self.temperture, self.reliability]]
            inputdata = np.array(inputdata)
            maxdata = np.array([199, 0.9])
            prediction = nn.gyro_nn(train, (inputdata / maxdata).reshape((-1, 2)), [[1]])
            nn_output = prediction[0][0]*1000.0
            self.nn_result_signal.emit(nn_output)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = My_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())