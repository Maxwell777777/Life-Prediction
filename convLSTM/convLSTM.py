import tensorflow as tf
from train_models import *
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize # import figsize

class ConvLSTM():
    def __init__(self, dbop=None, node=None):
        #super(ConvLSTM, self).__init__(dbop, node)
        self.testdata = {"FD001": [15, 30, 100, False, 2000, True],
                "FD002": [18, 21, 259, True, 30000, False],
                "FD003": [15, 30, 100, False, 20000, False],
                "FD004": [18, 19, 248, True, 30000, False]}  # [num_inputs,time_steps,test_size,f_modes]
        self.model = {"JANET": [[0, 2], [1, 10]],
                  "LSTM": [[0, 2], [1, 10]]}  # archetype,filter_chanels
        self.learning_rate = 0.001
        self.batch_size = 1024
        self.display_step = 100

        # Network Parameters

        self.num_hidden = 100  # num hidden units
        self.num_outputs = 1  # Only one output
        self.veces = 20

    def getdata(self, dataNum, modelName, archiType, if_test=True):
        '''
        :param dataNum: There are 4 sets of data, so dataNum equals 1, 2, 3, 4
        :param modelName: There are 2 types of models, so modelName = "JANET" or "LSTM"
        :param archiType: There are 2 types of architecture: ordinary and encoder-decoder. So the archiType = 0 or 1
        :param if_test: There are 2 modes: test and predict. So the if_test = True or False
        :return: model
        '''
        a = "FD00" + str(dataNum)
        model_path = "./PRUEBA_FINAL/" + a + "/"
        name = "Model"
        filter_channels = self.model[modelName][archiType][1]
        arq = self.model[modelName][archiType][0]

        if (dataNum == 1):
            X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a = get_data(
                window=self.testdata[a][1], f_modes=self.testdata[a][3], data1=True, if_test=if_test)  # only use dataset1
            if(if_test):
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_a, Y_test_a,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], self.testdata[a][2],
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                               self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name, save=True)
            else:
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_a, Y_test_a,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], 1,
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                               self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name, save=True)
        elif (dataNum == 2):
            X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a, X_test_b1, Y_test_b1, X_test_b2, Y_test_b2 = get_data(
                window=self.testdata[a][1], f_modes=self.testdata[a][3], if_test=if_test)  # use four datasets, test2, test4
            if(if_test):
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_b1, Y_test_b1,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], self.testdata[a][2],
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                               self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name)
            else:
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_b1, Y_test_b1,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], 1,
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                               self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name)
        elif (dataNum == 3):
            X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a, X_test_b1, Y_test_b1, X_test_b2, Y_test_b2 = get_data(
                window=self.testdata[a][1], f_modes=self.testdata[a][3], if_test=if_test)  # only use data1 and data3, test1, test3
            if(if_test):
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_b2, Y_test_b2,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], self.testdata[a][2],
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                              self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name)
            else:
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_b2, Y_test_b2,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], 1,
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                               self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name)
        else:
            X_train, Y_train, X_crossval, Y_crossval, X_test_a, Y_test_a, X_test_b1, Y_test_b1, X_test_b2, Y_test_b2 = get_data(
                window=self.testdata[a][1], f_modes=self.testdata[a][3], if_test=if_test)  # use four datasets, test2, test4
            if(if_test):
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_b2, Y_test_b2,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], self.testdata[a][2],
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                               self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name)
            else:
                modelx = model(X_train, Y_train,
                               X_crossval, Y_crossval,
                               X_test_a, Y_test_a,
                               X_test_b2, Y_test_b2,
                               model_path,
                               self.learning_rate, self.testdata[a][4],
                               1024, X_test_a.shape[0], 1,
                               self.display_step,
                               self.testdata[a][0], self.testdata[a][1],
                               self.num_hidden, self.num_outputs,
                               filter_channels, arq,
                               kind=modelName, name=name)

        return modelx

    def test(self, dataNum, modelName, archiType):
        '''
        :param dataNum: There are 4 sets of data, so dataNum equals 1, 2, 3, 4
        :param modelName: There are 2 types of models, so modelName = "JANET" or "LSTM"
        :param archiType: There are 2 types of architecture: ordinary and encoder-decoder. So the archiType = 0 or 1
        :return: None
        '''
        modelx = self.getdata(dataNum, modelName, archiType)
        modelx.test_b()
        a = "FD00" + str(dataNum)
        if(archiType == 0):
            path = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '/prediction.txt'
        else:
            path = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '_ED/prediction.txt'
        result = pd.read_csv(path, delim_whitespace=True, header=None)
        result = np.array(result)
        error = result[:, 0] - result[:, 1]
        print('MAE:', np.sum(np.abs(error)) / len(error))

        if (archiType == 0):
            path2 = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '/RMSE.txt'
        else:
            path2 = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '_ED/RMSE.txt'
        result2 = pd.read_csv(path2, delim_whitespace=True, header=None)
        result2 = np.array(result2)[-20:]
        print('RMSE:', np.sum(result2) / len(result2))

        message = 'MAE:'+ str(np.sum(np.abs(error)) / len(error)) + "\n" + 'RMSE:'+ str(np.sum(result2) / len(result2))

        return message


    def real(self, dataNum, modelName, archiType):
        '''
        :param dataNum: There are 4 sets of data, so dataNum equals 1, 2, 3, 4
        :param modelName: There are 2 types of models, so modelName = "JANET" or "LSTM"
        :param archiType: There are 2 types of architecture: ordinary and encoder-decoder. So the archiType = 0 or 1
        :return: None
        '''
        modelx = self.getdata(dataNum, modelName, archiType, False)
        modelx.test_b()
        a = "FD00" + str(dataNum)
        if(archiType == 0):
            path = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '/prediction.txt'
        else:
            path = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '_ED/prediction.txt'
        result = pd.read_csv(path, delim_whitespace=True, header=None)
        result = np.array(result)
        print("rul:",result[0][0])
        print("预测结果：",result[0][1])

        message = '预测结果:'+ str(result[0][1])

        return message


    def train(self, dataNum, modelName, archiType):
        '''
        :param dataNum: There are 4 sets of data, so dataNum equals 1, 2, 3, 4
        :param modelName: There are 2 types of models, so modelName = "JANET" or "LSTM"
        :param archiType: There are 2 types of architecture: ordinary and encoder-decoder. So the archiType = 0 or 1
        :return: None
        '''
        modelx = self.getdata(dataNum, modelName, archiType)
        for i in range(self.veces):
            modelx.train()
        return None

    def draw_image(self, n_set, modelName, archiType):
        if(n_set == 1 or n_set == 3):
            n_flights = 100
        elif(n_set == 2):
            n_flights = 259
        else:
            n_flights = 248
        set_images = './datafolder/test/test_FD00' + str(n_set) + '.txt'
        # set_rul = './datafolder/RUL_FD00' + str(n_set) + '.txt'
        testdata_list = []
        Test_data = pd.read_csv(set_images, delim_whitespace=True, header=None)
        Test_data.columns = ['Unit Number', 'Cycles', 'Operational Setting 1', 'Operational Setting 2',
                             'Operational Setting 3',
                             'Sensor Measurement  1', 'Sensor Measurement  2', 'Sensor Measurement  3',
                             'Sensor Measurement  4',
                             'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  7',
                             'Sensor Measurement  8',
                             'Sensor Measurement  9', 'Sensor Measurement  10', 'Sensor Measurement  11',
                             'Sensor Measurement  12',
                             'Sensor Measurement  13', 'Sensor Measurement  14', 'Sensor Measurement  15',
                             'Sensor Measurement  16',
                             'Sensor Measurement  17', 'Sensor Measurement  18', 'Sensor Measurement  19',
                             'Sensor Measurement  20',
                             'Sensor Measurement  21']
        for i in range(n_flights+1):
            flight = Test_data[Test_data['Unit Number'] == i]
            testdata_list.append(flight.shape[0])

        a = "FD00" + str(n_set)
        if (archiType == 0):
            path = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '/prediction.txt'
        else:
            path = "./PRUEBA_FINAL/" + a + "/" + 'Conv' + modelName + '_ED/prediction.txt'
        result = pd.read_csv(path, delim_whitespace=True, header=None)
        result = np.array(result)


        realdata_list = result[:,0]
        predict_list = result[:,1]

        ruldata_list = realdata_list.tolist()
        predict_list = predict_list.tolist()

        # print(len(ruldata_list))
        test_real1 = list(zip(testdata_list[1:],ruldata_list))
        test_real2 = list(zip(testdata_list[1:], predict_list))
        test_real1.sort()
        test_real2.sort()
        testdata_list_sorted = [y for y, x in test_real1]
        realdata_list_sorted = [x for y, x in test_real1]
        predict_list_sorted = [z for y, z in test_real2]
        print(test_real1)
        print(test_real2)
        print(testdata_list_sorted)
        print(realdata_list_sorted)
        print(predict_list_sorted)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(121)
        type1 = ax.scatter(testdata_list_sorted, realdata_list_sorted, color = 'red', alpha = 0.7, s = 15)
        type2 = ax.scatter(testdata_list_sorted, predict_list_sorted, color='blue', alpha=0.7, s=15)
        plt.xlabel("Decreasing RUL")
        plt.ylabel("RUL")
        ax.legend((type1, type2), ("real data", "predict data"), loc=0)
        ax2 = fig.add_subplot(122)
        type3 = ax2.scatter(realdata_list_sorted, predict_list_sorted, color='blue', alpha=0.7, s=15)
        plt.plot([0,125],[0,125], color = 'red')
        plt.xlabel("real")
        plt.ylabel("predict")
        if(archiType == 0):
            plt.savefig("./result_img/" + a + "_" + modelName + ".png")
        else:
            plt.savefig("./result_img/" + a + "_" + modelName + "_ED.png")
        # plt.savefig(a + "_" + modelName + ".png")
        # plt.show()
        plt.close('all')




if __name__ == "__main__":
    test = ConvLSTM()
    test.test(2, "JANET", 0)

    # test.test(2,'JANET',0)
    # test.real(2,'JANET',0)
    # test.draw_image(1, "JANET", 0)
    # test.draw_image(2, "JANET", 0)
    # test.draw_image(3, "JANET", 0)
    test.draw_image(2, "JANET", 0)





