from sympy import *
import numpy as np
import matplotlib.pyplot as plt

class StressExperiment():
    def __init__(self, step, n, R, T):
        self.step = step #实验步数
        self.n = n #进行测试的样本总数
        self.R = R #每一步失效停止数
        self.T = T #每一步的失效时间
        self.X = T
        self.M = [0.0]*step
        self.Y = [0.0]*step

    # 逆矩估计
    def inverse_moment_estimation(self):
        r = self.R[-1] #步降实验最高应力失效数
        t = self.T[-1] #步降实验最高应力失效时间
        W = [0.0] * r
        u = [0.0] * r
        m = Symbol('m') #待求的最高应力下的形状参数
        for i in range(r):
            if i == 0:
                W[i] = n * ((t[i]) ** m)
                u[i] = W[i]
            else:
                W[i] = (n - i) * (t[i] ** m - t[i - 1] ** m)
                u[i] = u[i - 1] + W[i]
        sum_lnu = 0
        for i in range(r):
            sum_lnu += log(u[-1] / u[i])
        print(nsolve(sum_lnu - r + 1, 0))
        m = nsolve(sum_lnu - r + 1, 0)
        sum_t = 0
        for i in range(r):
            sum_t += t[i] ** m
        #最高应力下的寿命特征参数
        y = ((sum_t + (n - r) * t[-1] ** m) / r) ** (1 / m)
        print(y)
        self.M[-1] = m
        self.Y[-1] = y
        self.step -= 1
        # self.Y[self.step - 1] = y
        # self.M[self.step - 1] = m
        return m, y

    #特征寿命参数的迭代
    def iteration_y(self, y0, T):
        step = self.step
        ni = len(T[step]) + len(T[step - 1]) #截止本步结束失效数
        ni0 = len(T[step]) #上一步失效数
        X = [0.0] * ni
        iteration_time = 0
        m = self.M[self.step]
        y = y0
        # y0 = self.Y[self.step]
        # print(y, y0)

        while (True):
            temp = y
            for i in range(len(X)):
                if i < ni0:
                    X[i] = T[step][i] * (y / y0)
                else:
                    X[i] = T[step - 1][i - ni0] + X[ni0 - 1]
            # print('X:',X)
            u_j = 0
            for j in range(ni):
                u_j += X[j] ** m
            u = u_j + (self.n - ni) * (X[ni - 1] ** m)

            y = (u / ni) ** (1.0 / m)
            print('y:', y)
            iteration_time += 1
            if (abs(y - temp) < 0.0007):
                break

        for i in range(len(X)):
            if i < ni0:
                X[i] = T[step][i] * (y / y0)
            else:
                X[i] = T[step - 1][i - ni0] + X[ni0 - 1]

        print('迭代了%d次' % iteration_time)
        return y, X

    #形状参数迭代
    def iteration_m(self, m, X):
        ni = sum(self.R)
        for i in range(self.step - 1):
            ni -= self.R[i]  # 截止本步结束失效数
        u = [0.0] * ni
        deta = np.linspace(0.5, 0, 1000)
        iteration_time = 0
        while True:
            for j in range(len(X)):
                u_j = 0
                for k in range(j):
                    u_j += X[k] ** m
                u[j] = u_j + (self.n - j) * X[j] ** m
            sum_lnu = 0
            for i in range(ni-1):
                sum_lnu += log(u[i])
            ni_new = 1 + (ni - 1) * log(u[-1]) - sum_lnu
            # print(ni)
            if ni_new - ni > 0.00001 and iteration_time < 1000:
                m = m - deta[iteration_time]
                # print(m)
            elif ni_new - ni < -0.00001 and iteration_time < 1000:
                m = m + deta[iteration_time]
                # print(m)
            else:
                print('停了')
                print(m)
                break
            iteration_time += 1
        return m

    #数据折算及参数计算
    def data_conversion(self):
        self.inverse_moment_estimation()
        while self.step > 0:
            while True:
                y, X = self.iteration_y(self.Y[self.step], self.X)
                m = self.iteration_m(self.M[self.step], X)
                if abs(y - self.Y[self.step - 1]) < 0.001 and abs(m - self.M[self.step - 1]) < 0.001:
                    self.Y[self.step - 1] = y
                    self.M[self.step - 1] = m
                    print(m,y)
                    break
                else:
                    self.Y[self.step - 1] = y
                    self.M[self.step - 1] = m

            self.X[self.step - 1] = X
            self.step -= 1
            # self.Y[self.step - 1] = y
        print(self.M, self.Y, self.X)
        return self.M, self.Y

    def coherence_m(self, m, n):
        num = sum(n)
        sum_mn = 0
        for i, j in zip(m, n):
            sum_mn += i * j
        return sum_mn / num

    def fun1(self, x):
        x = log(x)
        return x

    def fun2(self, x):
        x = 1 / x
        return x

    def Arrhenius(self, Y, T):
        y = list(map(self.fun1, Y))
        t = list(map(self.fun2, T))
        y = np.array(y, dtype=float)
        t = np.array(t, dtype=float)
        a = np.polyfit(t, y, 1)
        return a

    def Inverse_power(self, Y, V):
        y = list(map(self.fun1, Y))
        v = list(map(self.fun1, V))
        y = np.array(y, dtype=float)
        v = np.array(v, dtype=float)
        a = np.polyfit(v, y, 1)
        return a




if __name__ == "__main__":
    '''
    step:降应力步数
    n：样本数
    r：每一组失效的个数
    t：每一组失效时间
    M: 存放每一步的形状参数
    Y：存放每一步的特征寿命参数
    '''
    # step = 4
    # n = 40
    # R = [5, 5, 5, 20]
    # T = [[1.271, 2.392, 13.946, 20.725, 35.743],
    #      [0.209, 0.396, 0.806, 8.971, 9.325],
    #      [1.427, 2.419, 3.331, 5.757, 7.092],
    #      [5.755, 5.943, 6.476, 8.150, 9.348, 9.446, 9.581, 9.640, 9.819, 9.898, 10.646,
    #      10.887, 11.426, 11.554, 11.578, 11.659, 12.692, 13.072, 13.143, 13.342]]
    # # M = [0.0]*step
    # # Y = [0.0]*step
    # n_list = [35, 30, 25, 20]
    # V_list = [250, 270, 287, 300]
    # normal = 220
    # experiment = StressExperiment(step, n, R, T)
    # M, Y = experiment.data_conversion()
    # result = experiment.coherence_m(M, n_list)
    #
    # a = experiment.Inverse_power(Y, V_list)  # 逆幂模型：适用于应力失效机理，如：机械疲劳，机械磨损，电压击穿，绝缘击穿
    # a1 = experiment.Arrhenius(Y, V_list)  # Arrhenius模型：适用于电子产品，与温度有关的失效
    # pred = exp(a[1] + a[0] * log(normal))
    # # pred = exp(a1[1] + a1[0] * (1/normal))
    # r = np.linspace(0, 1, 1000)
    #
    #
    # def fun(x):
    #     x = pred * (log(1 / x)) ** (1 / result)
    #     return x
    #
    #
    # plt.plot(r, list(map(fun, r)))
    # plt.show()
    #
    # print(pred)

    step = 4
    n = 5
    R = [1, 1, 1, 2]
    # T = [[132],
    #      [264],
    #      [396],
    #      [168, 528]]
    T = [[1392],
         [364],
         [348],
         [168, 528]]
    n_list = [5, 4, 3, 2]
    T_list = [313.15, 326.15, 339.15, 353.15]
    normal = 298.15
    experiment = StressExperiment(step, n, R, T)
    M, Y = experiment.data_conversion()

    result = experiment.coherence_m(M, n_list)

    a1 = experiment.Arrhenius(Y, T_list)  # Arrhenius模型：适用于电子产品，与温度有关的失效
    print(a1)
    pred = exp(a1[1] + a1[0] * (1 / normal))
    r = np.linspace(0, 1, 1000)
    def fun(x):
        x = pred * (log(1 / x)) ** (1 / result)
        return x


    plt.plot(r, list(map(fun, r)))
    plt.show()

    print(pred)
    print(result)





