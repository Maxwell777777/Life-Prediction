import nn
import numpy as np
train = False
inputdata = [[198, 0.26]]
inputdata = np.array(inputdata)
maxdata = np.array([199, 0.9])
print(inputdata/maxdata)
prediction = nn.gyro_nn(train, (inputdata/maxdata).reshape((-1,2)), [[1]])
print('The prediction is: ',prediction[0][0]*1000.0)