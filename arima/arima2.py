import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class arima2():
    def __init__(self):
        pass

    def readTestData(self, set='./datafolder/test_FD001.txt'):
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

    def ari_model(self, testdata):
        y = testdata.flatten()
        print('y:',y)
        y1 = np.diff(y)
        # 对模型进行定阶
        from statsmodels.tsa.arima_model import ARIMA

        pmax = int(len(y1) / 10)  # 一般阶数不超过 length /10
        qmax = int(len(y1) / 10)
        bic_matrix = []
        for p in range(pmax + 1):
            temp = []
            for q in range(qmax + 1):
                try:
                    temp.append(ARIMA(y, (p, 1, q)).fit().bic)
                except:
                    temp.append(None)
                bic_matrix.append(temp)

        bic_matrix = pd.DataFrame(bic_matrix)  # 将其转换成Dataframe 数据结构
        p, q = bic_matrix.stack().astype("float64").idxmin()  # 先使用stack 展平， 然后使用 idxmin 找出最小值的位置
        print(u'BIC 最小的p值 和 q 值：%s,%s' % (p, q))  # BIC 最小的p值 和 q 值：0,1
        # 所以可以建立ARIMA 模型，ARIMA(0,1,1)
        model = ARIMA(y, (p, 1, q)).fit()
        # model.summary2()  # 生成一份模型报告
        y_pred = model.forecast(5)  # 为未来5天进行预测， 返回预测结果， 标准误差， 和置信区间
        return y_pred[0]

    def predict(self, n_flights=100, set='./datafolder/test_FD001.txt'):
        data = self.readTestData(set)
        unit = data['Unit Number'].values
        unit = unit[~np.isnan(unit)]
        L = len(unit)
        sen = [7, 12, 15]
        S = np.ndarray(shape=(L, 0))
        predictions = []
        for i in sen:
            temp = data['Sensor Measurement  ' + str(i)].values
            temp = temp[~np.isnan(temp)].reshape(L, 1)
            # temp = self.norm(temp)
            S = np.hstack((S, temp))
        for i in range(1):
            ind = np.where(unit == i + 1)[0]
            testdata = S[ind, :]
            # print(testdata.shape)[31,3]
            prediction = []
            for j in range(len(sen)):
                temp = testdata[:,j]
                # temp = temp.flatten()
                pred = self.ari_model(temp)
                prediction.append(pred)
            predictions.append(prediction)

        return predictions

if __name__ == "__main__":
    test = arima2()
    pred = test.predict()
    print(pred)
    print('pred shape:',np.array(pred).shape)