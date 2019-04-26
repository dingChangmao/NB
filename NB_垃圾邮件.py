import re
import random
import numpy as np

def testParse(bigString):
    listoftoken = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listoftoken if len(tok) > 2]


def createVocabList(dataSet):
    # 创建一个空的不重复列表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)    # 取并集
    return list(vocabSet)

def setofwords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def spamTest():
    docList = []
    classList = []
    fullTest = []
    for i in range(1, 26):
        wordList = testParse(open('email/spam/%d.txt' % i, 'r', encoding='ISO-8859-1').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = testParse(open('email/ham/%d.txt' % i, 'r', encoding='ISO-8859-1').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)  # 标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setofwords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSam = trainNB(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setofwords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSam) != classList[docIndex]:
            errorCount+=1
            # print('分类错误的测试集：',docList[docIndex])
    print('错误率：%.2f%%'%(float(errorCount)/len(testSet)*100))

if __name__ == '__main__':
   spamTest()

