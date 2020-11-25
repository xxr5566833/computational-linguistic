from config import *
from text_tackle import *
import nltk



def held_out(train_freqdict, train_r_Nr, valid_freqdict, valid_r_Nr):
    T = dict()
    r_list = list(train_r_Nr.keys())
    for r in r_list:
        if (r == 0):
            # 对于在valid中出现，但是在train中没有出现的，这时C1(bigram)肯定是0，那么求出相应的T0后面照常计算
            bigram_list = [big for big in valid_freqdict.keys() if train_freqdict.get(big) == None]
            Tr = sum([valid_freqdict[big] for big in bigram_list])

        else:
            bigram_list = [big for big, fre in train_freqdict.items() if fre == r]
            # 对于每一个bigram，统计它在valid里的次数，求和
            Tr = sum([valid_freqdict[big] for big in bigram_list if valid_freqdict.get(big) != None])
        T[r] = Tr
        # Tr_Nr = Tr / train_r_Nr[r]
        # if(abs(r - Tr_Nr) > 100):
        #     print(r, Tr_Nr)
    return T

# def get_average(T01, T10, freqdict_0, r_Nr_0, freqdict_1, r_Nr_1, freqdict):
#     average_P = dict()
#     N1 = freqdict_1.N()
#     N0 = freqdict_0.N()
#     N = N1 + N0
#     # 先求出ur r集合是T01和T10的r的并集，ur的r只需要求出train和valid 可能出现的r
#     ur_dict = {}
#     r_list = set(T01.keys()) | set(T10.keys())
#     for r in r_list:
#         if(T01.get(r) == None):
#             t0 = 0
#         else:
#             t0 = T01[r]
#         if(T10.get(r) == None):
#             t1 = 0
#         else:
#             t1 = T10[r]
#         ur = float(t0 + t1) / N
#         ur_dict[r] = ur
#     ur_dict[0] = 0
#     # 更新所有bigram的概率
#     all = freqdict_0.keys()
#     all = set(all) | set(freqdict_1.keys())
#     # 注意这里先把train和valid中出现的所有的bigram的概率求出来了，其他没出现的，一律分配 (1-已知) / 没出现的个数
#     for big in all:
#         # 不可能r0/r1都是0
#         if(freqdict_0.get(big) == None):
#             # 说明big在0数据集上没有出现，那么直接用1数据集的统计结果
#             r1 = freqdict_1[big]
#             N1_r1 = r_Nr_1[r1]
#             res = float(ur_dict[r1]) / float(N1_r1)
#         elif(freqdict_1.get(big) == None):
#             r0 = freqdict_0[big]
#             N0_r0 = r_Nr_0[r0]
#             res = float(ur_dict[r0]) / float(N0_r0)
#         else:
#             # 都存在那么取加权平均
#             r0 = freqdict_0[big]
#             r1 = freqdict_1[big]
#             N0_r0 = r_Nr_0[r0]
#             N1_r1 = r_Nr_1[r1]
#             res = float(ur_dict[r0] + ur_dict[r1]) / float(N0_r0 + N1_r1)
#         unnorm_r0 = freqdict_0[big]
#         unnorm_r1 = freqdict_1[big]
#         unnorm_r = freqdict[big]
#         normed_r = res * freqdict.N()
#         average_P[big] = res
#     p0 = 1.0 - sum([p for big, p in average_P.items()])
#     # p0分配给其他未知的bigram
#
#     return average_P

def get_average(T01, T10, freqdict_0, r_Nr_0, freqdict_1, r_Nr_1, bi_freqdict, bi_r_Nr, uni_freqdict):
    average_P = dict()
    N1 = freqdict_1.N()
    N0 = freqdict_0.N()
    # 先求出ur r集合是T01和T10的r的并集，ur的r只需要求出train和valid 可能出现的r
    # 更新所有bigram的概率
    all = freqdict_0.keys()
    all = set(all) | set(freqdict_1.keys())
    examples = [('增加', '资金'), ('国家', '控股'), ('，', '搞活'), ('劳动', '合作'), ('将', '出售'), ('自治', '制度')]
    for big in all:
        if(freqdict_0.get(big) == None):
            r0 = 0
        else:
            r0 = freqdict_0[big]
        if(freqdict_1.get(big) == None):
            r1 = 0
        else:
            r1 = freqdict_1[big]
        if(r0 == 0):
            res = float(T10[r1]) / float(N0 * r_Nr_1[r1])
        elif(r1 == 0):
            res = float(T01[r0]) / float(N1 * r_Nr_0[r0])
        else:
            res = (float(T01[r0] / r_Nr_0[r0]) + float(T10[r1] / r_Nr_1[r1])) / float(N0 + N1)
            # res1 = (float(T01[r0] / N1) + float(T10[r1] / N0)) / float(r_Nr_0[r0] + r_Nr_1[r1])
            # normed1_r = res1 * bi_freqdict.N()
            # oldres = float(T01[r0] + T10[r1]) / float(N1 * r_Nr_0[r0] + N0 * r_Nr_1[r1])
            # oldnormed_r = oldres * bi_freqdict.N()
        unnorm_r0 = freqdict_0[big]
        unnorm_r1 = freqdict_1[big]
        unnorm_r = bi_freqdict[big]
        normed_r0 = T01[r0] / r_Nr_0[r0]
        normed_r1 = T10[r1] / r_Nr_1[r1]
        normed_r = res * bi_freqdict.N()
        # print(unnorm_r0, unnorm_r1, unnorm_r, normed_r0, normed_r1, normed_r)
        average_P[big] = res
    p0 = 1.0 - sum([p for big, p in average_P.items()])
    # p0分配给其他未知的bigram
    # 所有出现的bigram已经被赋予了概率，剩下的没有出现的bigram的数量为
    remain_N = bi_r_Nr[0]
    p0 = p0 / remain_N
    # 同样的先对unk赋予概率
    uni_vocabulary = list(uni_freqdict.keys())
    uni_vocabulary.append(UNK)
    for unigram in uni_vocabulary:
        bi = (unigram, UNK)
        average_P[bi] = p0
        bi = (UNK, unigram)
        average_P[bi] = p0
    return average_P, p0

def deleted(train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr, valid_uni_freqdict, valid_uni_r_Nr,
            valid_bi_freqdict, valid_bi_r_Nr, train_valid_bi_freqdict, train_valid_bi_r_Nr, train_valid_uni_freqdict):
    T_bi_tv = held_out(train_bi_freqdict, train_bi_r_Nr, valid_bi_freqdict,
                                                  valid_bi_r_Nr)
    T_bi_vt = held_out(valid_bi_freqdict, valid_bi_r_Nr, train_bi_freqdict,
                                                  train_bi_r_Nr)
    # T_uni_tv = held_out(train_uni_freqdict, train_uni_r_Nr, valid_uni_freqdict,
    #                                               valid_uni_r_Nr)
    # T_uni_vt = held_out(valid_uni_freqdict, valid_uni_r_Nr, train_uni_freqdict,
    #                                               train_uni_r_Nr)
    bi_P, p0 = get_average(T_bi_tv, T_bi_vt, train_bi_freqdict, train_bi_r_Nr, valid_bi_freqdict, valid_bi_r_Nr,
                       train_valid_bi_freqdict, train_valid_bi_r_Nr, train_valid_uni_freqdict)
    # 更新(UNK, *) (*, UNK), (UNK, UNK)
    # 这些bigram在train和heldout中的r都是0！
    unigram_list = set(train_uni_freqdict.keys()) | set(valid_uni_freqdict.keys())
    # 这里使用bigram的概率，利用概率的加法原则求对应unigram的概率，然后求条件概率
    print(len(bi_P.keys()))
    unigram_list.add(UNK)
    uni_P = {}
    # 这里设置好每个unigram对应（以这个unigram作为一个word）的bigram的概率
    bi_p_unigram_list = {}
    for bi in bi_P.keys():
        bi_p_list_bi = bi_p_unigram_list.get(bi[0])
        if(bi_p_list_bi == None):
            bi_p_list_bi = []
            bi_p_unigram_list[bi[0]] = bi_p_list_bi
        bi_p_unigram_list.get(bi[0]).append(bi_P[bi])
    for unigram in unigram_list:
        bi_p_list = bi_p_unigram_list.get(unigram)
        count = len(bi_p_list)
        V = len(unigram_list)
        # 如果某个(unigram1, unigram2)没有在bi_P中出现，说明它在train和valid中的频率都是0，那么统一用上面p0的公式计算
        uni_P[unigram] = p0 * (V - count) + sum(bi_p_list)
    bi_p_cond = {}
    for bi in bi_P.keys():
        bi_p_cond[bi] = bi_P[bi] / uni_P[bi[0]]
    # uni_P = get_average(T_uni_tv, T_uni_vt, train_uni_freqdict, train_uni_r_Nr, valid_uni_freqdict, valid_uni_r_Nr)
    # p = float(T_uni_tv[0] + T_uni_vt[0]) / (
    #             valid_uni_freqdict.N() * train_uni_r_Nr[0] + train_uni_freqdict.N() * valid_uni_r_Nr[0])
    # uni_P[UNK] = p
    # # 接下来求P(w2|w1)条件概率，这里不像laplace有直接的条件概率计算公式
    # bi_p_cond = dict()
    # for bigram in bi_P.keys():
    #     bi_p = bi_P[bigram]
    #     uni_p = uni_P[bigram[0]]
    #     cond_p = bi_p / uni_p
    #     # 条件概率得小于1
    #     # assert(cond_p < 1)
    #     if(cond_p >= 1):
    #         print(bigram)
    #     bi_p_cond[bigram] = cond_p
    # print(sum([bi_p_cond[unigram, UNK] for unigram in uni_P.keys()]))
    return bi_P, bi_p_cond


if __name__ == "__main__":
    train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr = get_r_Nr(TRAIN)
    valid_uni_freqdict, valid_uni_r_Nr, valid_bi_freqdict, valid_bi_r_Nr = get_r_Nr(VALID)
    uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr = get_r_Nr(TRAIN_AND_VALID)
    deleted(train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr, valid_uni_freqdict, valid_uni_r_Nr,
            valid_bi_freqdict, valid_bi_r_Nr, bi_freqdict, bi_r_Nr, uni_freqdict)