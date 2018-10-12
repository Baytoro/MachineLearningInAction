'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # print(stringArr[0])
    datArr = [list(map(float, line)) for line in stringArr]
    # print(datArr[0])
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=False)
    eigVals, eigVects = linalg.eig(mat(covMat))
    # print(type(eigVects))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    # print(eigValInd)
    eigValInd = eigValInd[-1:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    # reorganize eig vects largest to smallest
    # print(type(eigVects))
    # print(type(eigValInd))
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # values that are not NaN (a number)
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # set NaN values to mean
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


if __name__ == '__main__':
    ld = loadDataSet('testSet.txt')
    # print(shape(ld))
    l, r = pca(ld, 1)
    # print(shape(l))
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.scatter(ld[:, 0].flatten().A[0],
               ld[:, 1].flatten().A[0], marker='^', s=60)

    ax.scatter(r[:, 0].flatten().A[0],
               r[:, 1].flatten().A[0], marker='o', s=50, color='red')

    plt.show()
