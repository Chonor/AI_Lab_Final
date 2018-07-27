# -*- coding:utf-8 -*-
import os
import csv
import sys
import math
import numpy as np
from collections import Counter
from os.path import join


def loadHash(Hashpath):
    reader = csv.reader(open(Hashpath, "r", encoding="utf-8"))
    Hash = []
    for row in reader:
        Hash = row
    return Hash


def loadIDF(IDFpath):
    reader = csv.reader(open(IDFpath, "r", encoding="utf-8"))
    IDF = []
    for row in reader:
        IDF = [float(x) for x in row]
    return IDF


def getIDF(trainpath, Hash):
    file = open(trainpath, "r", encoding="utf-8")
    reader = file.readlines()
    file.close()
    cnt = 0
    IDF = dict()
    print("getting idf...")
    for word in Hash:
        IDF[word] = 0
    for row in reader:
        cnt += 1
        sys.stdout.write("\rPercent: %f %%" % ((cnt)/62522 * 100))
        sys.stdout.flush()
        for word in Hash:
            if word in row:
                IDF[word] += 1
    for word in Hash:
        IDF[word] = math.log(62522/(1+IDF[word]))
    result = [str(IDF[x]) for x in Hash]
    return result
    print("")


def getTrainTFIDF(trainpath, outputdir, Hash, IDF):
    file = open(trainpath, "r", encoding="utf-8")
    reader = file.readlines()
    file.close()
    tfoutput = join(outputdir, "TF")
    tfidfoutput = join(outputdir, "TFIDF")
    if os.path.exists(tfoutput) is False:
        os.makedirs(tfoutput)
    if os.path.exists(tfidfoutput) is False:
        os.makedirs(tfidfoutput)
    writer1 = []
    writer2 = []
    for i in range(9):
        writer1.append(csv.writer(open(join(tfoutput, "train_" + str(i) + ".csv"), "w", encoding="utf-8", newline="")))
        writer2.append(csv.writer(open(join(tfidfoutput, "train_" + str(i) + ".csv"), "w", encoding="utf-8", newline="")))
    result1 = csv.writer(open(join(tfoutput, "result.csv"), "w", encoding="utf-8", newline=""))
    result2 = csv.writer(open(join(tfidfoutput, "result.csv"), "w", encoding="utf-8", newline=""))
    cnt = 0
    print("Dealing train data...")

    for row in reader:
        cnt += 1
        sys.stdout.write("\rPercent: %f %%" % ((cnt)/62522 * 100))
        sys.stdout.flush()
        row = row.split()
        label = row[0]
        row = row[1:]
        if "LOW" in label:
            result1.writerow(["1", "0", "0"])
            result2.writerow(["1", "0", "0"])
        elif "MID" in label:
            result1.writerow(["0", "1", "0"])
            result2.writerow(["0", "1", "0"])
        elif "HIG" in label:
            result1.writerow(["0", "0", "1"])
            result2.writerow(["0", "0", "1"])
        rowcnt = 0
        index = -1
        TF, TFIDF = [], []
        for word in Hash:
            index += 1
            if word in row:
                num = row.count(word)
                TF.append(num)
                TFIDF.append(num*IDF[index])
                rowcnt += num
            else:
                TF.append(0)
                TFIDF.append(0)
        if rowcnt != 0:
            TF = (np.array(TF) / rowcnt).tolist()
            TFIDF = (np.array(TFIDF) / rowcnt).tolist()
        for i in range(len(TF)):
            if TF[i] == 0.0:
                TF[i] = int(0)
            if TFIDF[i] == 0.0:
                TFIDF[i] = int(0)
        for i in range(9):
            j = i * 10000 + 10000
            j += 48 if i == 8 else 0
            writer1[i].writerow(TF[i*10000: j])
            writer2[i].writerow(TFIDF[i*10000: j])
    print("")


def getTestTFIDF(testpath, outputdir, Hash, IDF):
    file = open(testpath, "r", encoding="utf-8")
    reader = file.readlines()
    file.close()
    if os.path.exists(outputdir) is False:
        os.makedirs(outputdir)
    writer1 = csv.writer(open(join(outputdir, "TF.csv"), "w", encoding="utf-8", newline=""))
    writer2 = csv.writer(open(join(outputdir, "TFIDF.csv"), "w", encoding="utf-8", newline="")) 
    cnt = 0
    print("Dealing test data...")
    for row in reader:
        cnt += 1
        sys.stdout.write("\rPercent: %f %%" % ((cnt)/8671 * 100))
        sys.stdout.flush()
        row = row.split()[1:]
        rowcnt = 0
        index = -1
        TF, TFIDF = [], []
        for word in Hash:
            index += 1
            if word in row:
                num = row.count(word)
                TF.append(num)
                TFIDF.append(num*IDF[index])
                rowcnt += num
            else:
                TF.append(0)
                TFIDF.append(0)
        if rowcnt != 0:
            TF = (np.array(TF) / rowcnt).tolist()
            TFIDF = (np.array(TFIDF) / rowcnt).tolist()
        for i in range(len(TF)):
            if TF[i] == 0.0:
                TF[i] = int(0)
            if TFIDF[i] == 0.0:
                TFIDF[i] = int(0)
        writer1.writerow(TF)
        writer2.writerow(TFIDF)
    print("")


if __name__ == "__main__":
    currentpath = os.getcwd()
    trainpath = join(currentpath, "MulLabelTrain.ss")
    testpath = join(currentpath, "MulLabelTest.ss")
    Hashpath = join(currentpath, "Hash.csv")
    Hash = loadHash(Hashpath)
    IDF = getIDF(trainpath, Hash)
    # writer = csv.writer(open(join(currentpath, "IDF.csv"), "w", encoding="utf-8", newline=""))
    # writer.writerow(IDF)
    IDF = [float(x) for x in IDF]
    # IDF = loadIDF(join(currentpath, "IDF.csv"))
    trainoutputdir = join(currentpath+os.sep+"TFIDF")
    testoutputdir = join(currentpath+os.sep+"TFIDF", "test")
    getTrainTFIDF(trainpath, join(currentpath+os.sep+"TFIDF", "train"), Hash, IDF)
    getTestTFIDF(testpath, join(currentpath+os.sep+"TFIDF", "test"), Hash, IDF)