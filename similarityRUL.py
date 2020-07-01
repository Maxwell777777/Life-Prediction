import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings

class similarity():
    def __init__(self,n_flights=100, set = './model/data/train_FD001.txt'):
        #warnings.filterwarnings("ignore")
        # self.data_num = 1
        # self.n_flights = 100
        # self.train_set = './datafolder/train_FD00' + str(self.data_num) + '.txt'
        # self.test_set = './datafolder/test_FD00' + str(self.data_num) + '.txt'
        # self.RUL_set = './datafolder/RUL_FD00' + str(self.data_num) + '.txt'
        # self.real_set = './datafolder/real_FD00' + str(self.data_num) + '.txt'
        #self.real_set = "D:/mjtshanghai/LabProject/HealthQT/life prediction/datafolder/real_FD001.txt"
        pass

    def norm(self,x):
        M = max(x)
        m = min(x)
        r = M - m
        return [(i - m) / r for i in x]

    def fun(self,x, a, b):
        return a * (1 - np.exp(-b * x))

    def readTrainData(self,n_flights, set):
        data = pd.read_csv(set, delim_whitespace=True, header=None)
        values = np.linspace(1, n_flights, n_flights)
        data.columns = ['Unit Number', 'Cycles', 'Operational Setting 1', 'Operational Setting 2',
                        'Operational Setting 3',
                        'Sensor Measurement  1', 'Sensor Measurement  2', 'Sensor Measurement  3',
                        'Sensor Measurement  4',
                        'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  7',
                        'Sensor Measurement  8',
                        'Sensor Measurement  9', 'Sensor Measurement  10', 'Sensor Measurement  11',
                        'Sensor Measurement  12', 'Sensor Measurement  13', 'Sensor Measurement  14',
                        'Sensor Measurement  15', 'Sensor Measurement  16', 'Sensor Measurement  17',
                        'Sensor Measurement  18', 'Sensor Measurement  19', 'Sensor Measurement  20',
                        'Sensor Measurement  21']
        data['RUL'] = pd.Series(np.zeros(len(data)), index=data.index)
        for value in values:
            a = data[data['Unit Number'] == value]
            time = len(a)
            RUL = np.linspace(time - 1, 0, time)
            # for n in range(0, len(RUL)):
            #     if RUL[n] > 125:  # RUL edition for 99.99% reliability
            #         RUL[n] = 125
            data.loc[a.index, 'RUL'] = pd.Series(RUL, index=a.index)
        #data = np.array(data)
        # print(data.shape) [20631,27]
        #L = data.shape[0]
        return data

    def readTestData(self, set):
        data = pd.read_csv(set, delim_whitespace=True, header=None)
        data.columns = ['Unit Number', 'Cycles', 'Operational Setting 1', 'Operational Setting 2',
                        'Operational Setting 3',
                        'Sensor Measurement  1', 'Sensor Measurement  2', 'Sensor Measurement  3',
                        'Sensor Measurement  4',
                        'Sensor Measurement  5', 'Sensor Measurement  6', 'Sensor Measurement  7',
                        'Sensor Measurement  8',
                        'Sensor Measurement  9', 'Sensor Measurement  10', 'Sensor Measurement  11',
                        'Sensor Measurement  12', 'Sensor Measurement  13', 'Sensor Measurement  14',
                        'Sensor Measurement  15', 'Sensor Measurement  16', 'Sensor Measurement  17',
                        'Sensor Measurement  18', 'Sensor Measurement  19', 'Sensor Measurement  20',
                        'Sensor Measurement  21']
        return data

    def RUL_finding(self,sen, S, M, Ti, r, n_flghts, coef = [[0.49893257, 0.5686767, -0.50950882]], intercept = [0.11699473]):
        y = S @ coef.reshape(len(sen), 1) + intercept  # intercept_截距 coef_回归系数
        #     print(len(y))# 115,114,110...158
        RUL_pred = np.zeros((1, n_flghts), dtype=float)[0]
        D = np.zeros((1, n_flghts), dtype=float)[0]  # 100个0的一维数组

        for i in range(n_flghts):
            d = 0
            for j in range(r):
                #             def fun(x,a,b):
                #                 return a * (1- np.exp(-b * x))
                d += (y[j] - self.fun(-Ti[i] + j, M[i, 0], M[i, 1])) ** 2

            RUL_pred[i] = Ti[i] - r + 1
            D[i] = d

        # D_pos = D[np.where(RUL_pred>0)]
        # RUL_pos_pred= RUL_pred[np.where(RUL_pred>0)]

        # Outlier removal
        D_pos = []
        RUL_pos_pred = []

        for i in range(len(RUL_pred)):

            if RUL_pred[i] > 0 and RUL_pred[i] + r > 150 and RUL_pred[i] < 190:
                D_pos.append(D[i])
                RUL_pos_pred.append(RUL_pred[i])

        temp = sorted(zip(D_pos, RUL_pos_pred))
        #     print(len(temp))
        #     print(temp)
        D_pos, RUL_pos_pred = map(list, zip(*temp))
        #     print(D_pos)
        #     print(RUL_pos_pred)
        RUL_final = 4 / 5 * RUL_pos_pred[0] + 1 / 5 * RUL_pos_pred[-1]

        return RUL_final

    def train(self, n_flights, set):
        data = self.readTrainData(n_flights, set)
        unit = data['Unit Number'].values
        unit = unit[~np.isnan(unit)]
        L = len(unit)
        S = np.ndarray(shape=(L, 0))
        data_num = int(set[-5])
        if(data_num == 1):
            sen = [7, 12, 15]
        elif(data_num == 2):
            sen = [11,14]
        elif(data_num == 3):
            sen = [3,4,11]
        else:
            sen = [3,11]
        num_sen = len(sen)
        for i in sen:
            temp = data['Sensor Measurement  ' + str(i)].values
            temp = temp[~np.isnan(temp)].reshape(L,1)
            temp = self.norm(temp)
            S = np.hstack((S, temp))
        print(S.shape)  # 将sen中的sensor数据归一化，拼接在一起  [20631,3]
        X = np.ndarray(shape=(0, num_sen))
        Ti = []
        y = np.ndarray(shape=(0, 1))
        M = np.ndarray(shape=(n_flights, 2))  # store 100 as and bs
        for i in range(1, n_flights+1):
            ind = np.where(unit == i)[0]  # idlist where unit==i
            Ti.append(len(ind))  # To be used later in the testing phase [192 unit=1,...]

            temp = np.linspace(1, 0, len(ind), dtype=float).reshape(len(ind), 1)
            y = np.vstack((y, temp))  # at last the dimension is [len(unit),1] 1~0

            X = np.vstack((X, S[ind, :]))  # X is S

            # Exponential curve fitting
            C_adj = np.arange(-len(ind), 0, 1) + 1  # C_adj is from -num(idlist)+1 to 0
            popt, _ = curve_fit(self.fun, C_adj, temp.flatten())  # fit fun = a(1-e^(-bx))

            M[i - 1, 0] = popt[0]
            M[i - 1, 1] = popt[1]
        lm = LinearRegression()
        print('training...')
        lm.fit(X, y)
        print(lm.coef_, lm.intercept_)
        print('done!')
        #print(M)
        return lm.coef_,lm.intercept_,M,Ti

    def test(self, n_flights, set, real):
        data = self.readTestData(set)
        unit = data['Unit Number'].values
        unit = unit[~np.isnan(unit)]
        L = len(unit)
        data_num = int(set[-5])
        if (data_num == 1):
            sen = [7, 12, 15]
        elif (data_num == 2):
            sen = [11, 14]
        elif (data_num == 3):
            sen = [3,4,11]
        else:
            sen = [3,11]
        coef, intercept, M ,Ti = self.train(n_flights, "./datafolder/train/train_FD00" + set[-5] + ".txt")
        S = np.ndarray(shape=(L, 0))
        predictions = []
        for i in sen:
            temp = data['Sensor Measurement  ' + str(i)].values
            temp = temp[~np.isnan(temp)].reshape(L,1)
            temp = self.norm(temp)
            S = np.hstack((S, temp))
        for i in range(n_flights):
            ind = np.where(unit == i+1)[0]
            r = len(ind)
            testdata = S[ind, :]
            #prediction = RUL_finding(sen, testdata, M, Ti, r, [[ 0.49893257,0.5686767,-0.50950882]], [0.11699473])
            prediction = self.RUL_finding(sen, testdata, M, Ti, r, n_flights, coef, intercept)
            predictions.append(prediction)
        y = pd.read_csv(real, delim_whitespace=True, header=None)
        y = np.array(y)
        # realdata_list = y,tolist()
        y = y.flatten()
        predictions_array = np.array(predictions)
        #print('y:',y)
        error = y-predictions_array
        self.draw_image(data_num, y.tolist(), predictions_array.tolist())
        print('MAE:',np.sum(np.abs(error))/n_flights)
        print('RMSE:', np.sqrt(np.sum(np.power(error,2)) / n_flights))
        return  'MAE:'+ str(np.sum(np.abs(error))/n_flights) + "\n" + 'RMSE:'+ str(np.sqrt(np.sum(np.power(error,2)) / n_flights))

    def draw_image(self, n_set, ruldata_list, predict_list):
        if (n_set == 1 or n_set == 3):
            n_flights = 100
        elif (n_set == 2):
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
        for i in range(n_flights + 1):
            flight = Test_data[Test_data['Unit Number'] == i]
            testdata_list.append(flight.shape[0])

        test_real1 = list(zip(testdata_list[1:], ruldata_list))
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
        type1 = ax.scatter(testdata_list_sorted, realdata_list_sorted, color='red', alpha=0.7, s=15)
        type2 = ax.scatter(testdata_list_sorted, predict_list_sorted, color='blue', alpha=0.7, s=15)
        plt.xlabel("Decreasing RUL")
        plt.ylabel("RUL")
        ax.legend((type1, type2), ("real data", "predict data"), loc=0)
        ax2 = fig.add_subplot(122)
        type3 = ax2.scatter(realdata_list_sorted, predict_list_sorted, color='blue', alpha=0.7, s=15)
        plt.plot([0, 125], [0, 125], color='red')
        plt.xlabel("real")
        plt.ylabel("predict")
        plt.savefig("./result_img/similarity.png")
        # plt.show()
        plt.close('all')

    def real(self,n_flights, set):
        data = self.readTestData(set)
        unit = data['Unit Number'].values
        unit = unit[~np.isnan(unit)]
        L = len(unit)
        data_num = int(set[-5])
        if (data_num == 1):
            sen = [7, 12, 15]
        elif (data_num == 2):
            sen = [11, 14]
        elif (data_num == 3):
            sen = [3,4,11]
        else:
            sen = [3,11]
        coef, intercept, M, Ti = self.train(n_flights, "./datafolder/train/train_FD00" + set[-5] + ".txt")
        S = np.ndarray(shape=(L, 0))
        predictions = []
        for i in sen:
            temp = data['Sensor Measurement  ' + str(i)].values
            temp = temp[~np.isnan(temp)].reshape(L, 1)
            temp = self.norm(temp)
            S = np.hstack((S, temp))


        r = len(S)
        testdata = S[:, :]
        prediction = self.RUL_finding(sen, testdata, M, Ti, r, n_flights, coef, intercept)
        predictions.append(prediction)

        predictions_array = np.array(predictions)
        print(predictions_array)

        return '预测结果:'+ str(predictions_array[0])


if(__name__ == "__main__"):
    test = similarity()
    # predictions = test.test(test.n_flights, test.test_set, test.RUL_set)
    # predictions = test.real(test.n_flights, test.real_set)
    predictions = test.test(100, "./datafolder/test/test_FD001.txt", "./datafolder/RUL_FD001.txt")
    predictions = test.real(100, "./datafolder/real/real_FD001.txt")
    #print(len(predictions))
    #print(predictions)
