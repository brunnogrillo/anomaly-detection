import glob
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import joblib
import cv2
import numpy as np
from tqdm import tqdm


class OCSVM(object):
    def __init__(self):
        self.model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        self.ss = StandardScaler()
        self.ocsvmclf = svm.OneClassSVM(gamma=0.001,
                                        kernel='rbf',
                                        nu=0.01)
        self.ifclf = IsolationForest(contamination=0.01,
                                     max_features=1.0,
                                     max_samples=1.0,
                                     n_estimators=100,
                                     verbose=1)
        self.pca = None

    def extractResnet(self, X):
        # X numpy array
        fe_array = self.model.predict(X)
        return fe_array

    def prepareData(self, path):
        datalist = glob.glob(path + '/*.jpg')
        felist = []
        for p in tqdm(datalist):
            img = cv2.imread(p)
            img = cv2.resize(img, (224, 224))
            # img = preprocess_input(img, mode='tf')
            img = np.expand_dims(img, axis=0)
            fe = self.extractResnet(img)
            felist.append(fe.reshape(1, -1))

        X_t = felist[0]
        for i in range(len(felist)):
            if i == 0:
                continue
            X_t = np.r_[X_t, felist[i]]

        return X_t

    def initPCA(self, X_train):
        self.pca = PCA(n_components=X_train.shape[0], whiten=True)

    def doSSFit(self, Xs):
        self.ss.fit(Xs)

    def doPCAFit(self, Xs):
        self.pca = self.pca.fit(Xs)
        return Xs

    def doSSTransform(self, Xs):
        Xs = self.ss.transform(Xs)
        return Xs

    def doPCATransform(self, Xs):
        Xs = self.pca.transform(Xs)
        return Xs

    def train(self, Xs):
        self.ocsvmclf.fit(Xs)
        self.ifclf.fit(Xs)

    def predictSVM(self, Xs):
        pred = self.ocsvmclf.predict(Xs)
        return pred

    def predictIf(self, Xs):
        pred = self.ifclf.predict(Xs)
        return pred


def trainSVM():
    f = OCSVM()
    X_train = f.prepareData('data/train')
    # do StandardScaler
    f.doSSFit(X_train)
    X_train = f.doSSTransform(X_train)
    # do pca
    f.initPCA(X_train)
    f.doPCAFit(X_train)
    X_train = f.doPCATransform(X_train)
    # train svm
    f.train(X_train)

    # save our models
    joblib.dump(f.ocsvmclf, 'ocsvmclf.model')
    joblib.dump(f.ifclf, 'ifclf.model')
    joblib.dump(f.pca, 'pca.model')
    joblib.dump(f.ss, 'ss.model')


def loadSVMAndPredict():
    f = OCSVM()
    # load models
    f.ocsvmclf = joblib.load('ocsvmclf.model')
    f.pca = joblib.load('pca.model')
    f.ss = joblib.load('ss.model')

    X_test = f.prepareData('data/test')
    # do test data ss
    X_test = f.doSSTransform(X_test)
    # do test data pca
    X_test = f.doPCATransform(X_test)

    # predict
    preds = f.predictSVM(X_test)
    print(f'{preds}')

def loadIfAndPredict():
    f = OCSVM()
    # load models
    f.ifclf = joblib.load('ifclf.model')
    f.pca = joblib.load('pca.model')
    f.ss = joblib.load('ss.model')

    X_test = f.prepareData('data/test')
    # do test data ss
    X_test = f.doSSTransform(X_test)
    # do test data pca
    X_test = f.doPCATransform(X_test)

    # predict
    preds = f.predictIf(X_test)
    print(f'{preds}')


if __name__ == '__main__':
    trainSVM()
    # loadSVMAndPredict()
    pass
