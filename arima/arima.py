import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings


class arima():
    def __init__(self):
        #warnings.filterwarnings("ignore")
        pass

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

    def ari_model(self, testdata, num):
        y = testdata.flatten()
        #print('y:',y)
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
        #print(u'BIC 最小的p值 和 q 值：%s,%s' % (p, q))  # BIC 最小的p值 和 q 值：0,1
        # 所以可以建立ARIMA 模型，ARIMA(0,1,1)
        model = ARIMA(y, (p, 1, q)).fit()
        # model.summary2()  # 生成一份模型报告
        y_pred = model.forecast(num)  # 为未来5天进行预测， 返回预测结果， 标准误差， 和置信区间
        return y_pred[0]

    def predict(self, sen, num, set='./datafolder/real/real_FD001.txt'):
        data = self.readTestData(set)
        unit = data['Unit Number'].values
        unit = unit[~np.isnan(unit)]
        L = len(unit)
        #sen = [5,7,12]
        S = np.ndarray(shape=(L, 0))
        predictions = []
        for i in sen:
            temp = data['Sensor Measurement  ' + str(i)].values
            temp = temp[~np.isnan(temp)].reshape(L, 1)
            # temp = self.norm(temp)
            S = np.hstack((S, temp))

        testdata = S[:, :]
        prediction = []
        output = ""
        errorput = ""
        avilible = []
        for j in range(len(sen)):
            temp = testdata[:,j]
            # temp = temp.flatten()
            try:
                pred = self.ari_model(temp, num)
            except ValueError:
                print(str(sen[j])+'序列无法进行arima预测')
                errorput = errorput+str(sen[j])+'序列无法进行arima预测'+"\n"
                continue
            pred = pred.tolist()
            prediction.append(pred)
            pred_new = [str(x) for x in pred]
            pred_new = ",".join(pred_new)
            output = output+str(sen[j])+"序列的预测为"+pred_new+"\n"
            avilible.append(sen[j])
        predictions.append(prediction)
        print(predictions[0])

        return output, errorput, predictions[0], avilible

    def draw_image(self, n_set, list_num, predict, circle):
        set_images = './datafolder/real/real_FD00' + str(n_set) + '.txt'
        Real_data = pd.read_csv(set_images, delim_whitespace=True, header=None)

        data = np.array(Real_data)
        color_list = ['r','y','g','b','m','pink','black',
                      'brown','orange','lawngreen','cyan','purple','orchid','slategray',
                      'lightcoral','chocolate','palegreen','navy','blueviolet','crimson','gold']

        temp = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in list_num:
            realdata_list = data[:, i+4]
            time = data[:, 1]
            time = time.tolist()
            realdata_list = realdata_list.tolist()
            time.append(time[-1]+1)
            realdata_list.append(predict[temp][0])
            ax.plot(time,realdata_list,color = color_list[i-1], linestyle = '--',label='sensor'+str(i))
            circle_list = []
            for j in range(circle):
                circle_list.append(time[-2]+j+1)
            ax.plot(circle_list,predict[temp],color = color_list[i-1], linestyle = '-', marker = 'o')
            temp+=1
        plt.xlabel("time")
        plt.ylabel("data")
        plt.legend()
        plt.savefig("./result_img/arima.png")
        # plt.show()
        plt.close('all')

if __name__ == "__main__":
    test = arima()
    output, error, predict, available = test.predict([1,7,12],2,"D:/mjtshanghai/LabProject/HealthQT/life prediction/datafolder/real_FD001.txt")
    #
    # print("hello")
    print('output:',predict)
    # print(error)
    test.draw_image(1,available, predict,2)