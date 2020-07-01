import prepare
import importlib
import numpy as np

def example():
    path = './arimadata/business.xls'
    business = prepare.loadDataSet(path)
    business_copy = business.copy()
    business_copy['rentNumber'] = np.log(business_copy.rentNumber)
    message = prepare.stationarity_test(business_copy,7)
    message += prepare.whitenoise_test(business_copy,7)
    importlib.reload(prepare)
    pred, RMSE=prepare.arma_predict(business_copy,21)
    return message

if __name__ == '__main__':
    example()