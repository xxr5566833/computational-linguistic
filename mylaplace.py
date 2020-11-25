from config import *
from text_tackle import *
import math
def bi_laplace(uni_vocabulary, freqdict, V):
    P = {}
    N = freqdict.N()
    # 直接计算所有|V|²个bigram的太耗时间了，要算30亿（没数错的话）个概率，还是让<UNK, UNK>, <UNK, *>和<*, UNK>代替吧
    for bi in freqdict.keys():
        bip = float(freqdict[bi] + 1) / float(N + V)
        P[bi] = bip
        unnorm_r = freqdict[bi]
        normed_r = bip * N
        # print(unnorm_r, normed_r)
    for unigram in uni_vocabulary:
        p = 1.0 / float(N + V)
        bi = (unigram, UNK)
        P[bi] = p
        bi = (UNK, unigram)
        P[bi] = p
    # print(P[(UNK, UNK)])
    return P
def cond_laplace(uni_vocabulary, uni_freqdict, bi_freqdict):
    P = {}
    V = len(uni_vocabulary)
    for bi in bi_freqdict.keys():
        unnorm_count_bi = bi_freqdict.get(bi)
        w1 = bi[0]
        unnorm_count_uni = uni_freqdict.get(w1)
        bip_cond = float(unnorm_count_bi + 1) / float(V + unnorm_count_uni)
        P[bi] = bip_cond
    # 补充P(*|UNK) P(UNK|*) P(UNK|UNK)
    for unigram in uni_freqdict.keys():
        # P(*|UNK)
        # UNK对应的count是0
        p = 1.0 / float(V)
        bi = (UNK, unigram)
        P[bi] = p
        # P(UNK|*)
        p = 1.0 / float(V + uni_freqdict.get(unigram))
        bi = (unigram, UNK)
        P[bi] = p
    # P(UNK|UNK)
    p = 1.0 / float(V)
    bi = (UNK, UNK)
    P[bi] = p
    # 检验 需要接近1
    print(sum([P[(unigram, UNK)] for unigram in uni_vocabulary]))
    return P


def laplace(train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr):
    # unigram的集合 U {UNK} 才是unigram整个规模，然后平方就是bigram可能的种类数量
    # print(train_bi_freqdict.B())
    uni_keys = list(train_uni_freqdict.keys())
    uni_keys.append(UNK)
    # print(len(uni_keys))
    bi_V = (len(uni_keys))**2
    # 除了需要求bigram的概率以外，还需要求P(w2|w1)的概率，注意目前来看好像P(w2|w1)的概率不能通过P(w1 w2)和P(w1)和贝叶斯公式求出来，因为
    # P(w2|w1)的laplace smoothing公式是直接从 狄利克雷分布作为先验分布推导来的
    bi_P = bi_laplace(uni_keys, train_bi_freqdict, bi_V)
    bi_P_cond = cond_laplace(uni_keys, train_uni_freqdict, train_bi_freqdict)
    # 其他方法也想一想(UNK, UNK), (UNK, unigram), (unigram, UNK)
    return bi_P, bi_P_cond
if __name__ == "__main__":
    uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr = get_r_Nr(TRAIN_AND_VALID)
    bi_P, bi_P_cond = laplace(uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr)