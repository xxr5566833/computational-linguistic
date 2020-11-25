import nltk
from config import *
import numpy as np
import simple_good_turing
from text_tackle import *
def computeZr(r_Nr):
    Zr = {}
    rs = [x[0] for x in r_Nr]
    ns = [x[1] for x in r_Nr]
    for i, r in enumerate(rs):
        if(r == 0):
            continue
        if(r == 1):
            q = 0
        else:
            q = rs[i - 1]
        if(i == (len(rs) - 1)):
            t = 2 * r - q
        else:
            t = rs[i + 1]
        Zr[r] = 2 * ns[i] / float(t - q)
    return Zr

def computeCoeff(rs, Zrs):
    logr = np.log10(np.array(list(rs)))
    logz = np.log10(np.array(list(Zrs)))
    A = np.vstack([logr, np.ones(len(logr))]).T
    a, b = np.linalg.lstsq(A, logz)[0]
    return a, b

def computeS(a, b, rs):
    return 10**(b + a * np.log10(rs))

def computeRstar(r_Nr, S):
    # 两个选择x, y
    # 如果flag为true 选择y ，否则选择x
    # flag只有在①r+1对应的N(r+1)不存在②x-y的abs超出阙值 时才会变成true
    rs = [x[0] for x in r_Nr]
    ns = [x[1] for x in r_Nr]
    rstar = []
    # 有的是1.96，在statistical NLP中是1.65 所以这里使用1.65
    # The 1.96 coefficient correspond to a 0.05 significance criterion,
    # some implementations can use a coefficient of 1.65 for a 0.1
    # significance criterion.
    coeff = 1.65
    flag = False
    for i in range(len(rs)):
        r = rs[i]
        nr = ns[i]
        # 注意S[0]记录着r=1时的值，所以r = r+1时对应S[r]
        y = float(r + 1) * S[r] / S[r - 1]
        # 如果使用了y，那么会一直使用y
        # 考虑N(r+1) 不存在的情况
        if(i + 1 >= len(rs) or rs[i + 1] != r + 1):
            flag = True
        if(flag == False):
            x = float(r + 1) * ns[i + 1] / nr
            std = np.sqrt((r + 1)**2 * ns[i + 1] / nr**2 * (1 + ns[i + 1] / nr))
            if(abs(y - x) <= coeff * std):
                flag = True
        if(flag):
            rstar.append(y)
        else:
            rstar.append(x)

    return rstar

def sgt(uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr):
    bi_r_Nr_list = sorted(bi_r_Nr.items(), key=lambda item:item[0])
    bi_unk_count = bi_r_Nr[0]
    # 计算r Zr时不考虑r = 0的情况
    bi_r_Nr_list.pop(0)
    print(bi_r_Nr_list[0])
    # print("修正前的总数")
    # print(freqdist.N())
    p0 = bi_r_Nr_list[0][1] / bi_freqdict.N()
    Zr = computeZr(bi_r_Nr_list)
    rs = Zr.keys()
    Zrs = Zr.values()
    a, b = computeCoeff(rs, Zrs)
    # print(max(rs) + 1)
    S = computeS(a, b, range(1, max(rs) + 2))
    r_star = computeRstar(bi_r_Nr_list, S)
    reestimate_r = {}
    reestimate_p = {}
    newN = 0.
    for i in range(len(r_star)):
        newN += (r_star[i] * bi_r_Nr_list[i][1])
    # reestimate_p存储频率为r的所有bigram的概率(平均分配概率)
    reestimate_p[0] = (p0 / bi_unk_count)
    # print("修正后除去unseen的总数，修正后的总数")
    # print(newN, newN + old_r_Nr[0][1])
    for i in range(len(r_star)):
        r = bi_r_Nr_list[i][0]
        reestimate_p[r] = (1 - p0) * (r_star[i] / newN)
    reestimate_r[0] = (bi_r_Nr_list[0][1] / bi_unk_count)
    for i in range(len(r_star)):
        r = bi_r_Nr_list[i][0]
        reestimate_r[r] = (reestimate_p[r] * bi_freqdict.N())
    print(reestimate_r)
    print(reestimate_p)
    # 输出为作业需要的格式
    bi_P = {}
    for bi in bi_freqdict.keys():
        bi_r = bi_freqdict.get(bi)
        bi_P[bi] = reestimate_p[bi_r]
    unigram_list = list(uni_freqdict.keys())
    for unigram in unigram_list:
        p = reestimate_p[0]
        bi_P[(unigram, UNK)] = p
        bi_P[(UNK, unigram)] = p
        # 重复更新了，懒得单独写
        bi_P[UNK, UNK] = p
    unigram_list.append(UNK)
    uni_P = {}
    # 这里设置好每个unigram对应（以这个unigram作为一个word）的bigram的概率
    bi_p_unigram_list = {}
    for bi in bi_P.keys():
        bi_p_list_bi = bi_p_unigram_list.get(bi[0])
        if (bi_p_list_bi == None):
            bi_p_list_bi = []
            bi_p_unigram_list[bi[0]] = bi_p_list_bi
        bi_p_unigram_list.get(bi[0]).append(bi_P[bi])
    for unigram in unigram_list:
        bi_p_list = bi_p_unigram_list.get(unigram)
        count = len(bi_p_list)
        V = len(unigram_list)
        p = reestimate_p[0]
        uni_P[unigram] = p * (V - count) + sum(bi_p_list)
    bi_P_cond = {}
    for bi in bi_P:
        bi_P_cond[bi] = bi_P[bi] / uni_P[bi[0]]


    return bi_P, bi_P_cond

def test():
    # train = getTrainSet()
    # train_texts = split_and_addBE(train)
    # bigrams = nltk.bigrams(train_texts)
    # freqdist = nltk.FreqDist(bigrams)
    # r_Nr = freqdist.r_Nr()
    # r_Nr.pop(0)
    # print(len(r_Nr))
    # testEst = simple_good_turing.Estimator(r_Nr)
    testEst = simple_good_turing.ChinesePluralsTest()
    testEst.test_unnorm_output()

if __name__ == "__main__":
    uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr = get_r_Nr(TRAIN_AND_VALID)
    bi_P, bi_P_cond = sgt(uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr)