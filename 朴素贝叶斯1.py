import numpy as np
from functools import reduce
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec
def setOfWords2Vec(myVocabList, postinDoc):
    returnVec = [0] * len(myVocabList)
    # print(returnVec)
    for word in postinDoc:
        if word in myVocabList:
            returnVec[myVocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def createVocabList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)
    # print(vocabSet)
    return list(vocabSet)
def trainNB(trainMatrix,trainCategory):
    # print(trainCategory)
    num = len(trainMatrix)
    numwords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(num)
    p0Num = np.ones(numwords)
    p1Num = np.ones(numwords)
    p0Denom = 2.0
    p1Denom = 2.0                            #分母初始化为0
    for i in range(num):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num +=trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print(p1Denom)
    print(p0Denom)
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    print(p1Vect)
    print(p0Vect)
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0



def testingNB():
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love','my','dalmation','quit','buying','worthless']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
    testEntry = ['quit', 'buying']  # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')




if __name__ == '__main__':
    # postingLIst, classVec = loadDataSet()
    # myVocaList = createVocabList(postingLIst)
    # print('myVocabList:\n',myVocaList)
    # trainMat=[]
    # for postinDoc in postingLIst:
    #     trainMat.append(setOfWords2Vec(myVocaList, postinDoc))
    # # print(len(trainMat[0]))
    # # print('trainMat:\n', trainMat)
    # p0V, p1V, pAb = trainNB(trainMat, classVec)
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # # print('classVec:\n', classVec)
    # # print('pAb:\n', pAb)
    testingNB()