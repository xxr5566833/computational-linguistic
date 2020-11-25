from text_tackle import *
from mysgt import sgt
from mydeleted import deleted
from mylaplace import laplace
from config import *
import os
def get_rank(P):
    rank = {}
    bi_p = list(P.items())
    bi_p.sort(key = lambda x:x[1], reverse = True)
    i = 0
    while(i < len(bi_p)):
        r = i + 1
        bi = bi_p[i][0]
        p = bi_p[i][1]
        # 检验bi后面的bi是否概率和它一样，如果有，那么需要平均rank，如果没有，那么设置rank[bi] = r
        # 注意rank好像要从1开始
        j = i + 1
        # 这里判断两个float是否相等，我相信python的==有这个判断能力
        while(j < len(bi_p) and bi_p[j][1] == p):
            j += 1
        if(j > i + 1):
            # 现在j存储着第一次遇到概率不是p的下标
            # 那么下标从i 到 j-1的位置的bi的rank是 (i + j + 1) / 2
            for k in range(i, j, 1):
                bi_k = bi_p[k][0]
                rank[bi_k] = (i + j + 1) / 2
        else:
            rank[bi] = r
        i = j
    return rank

def compute_spearman(rank1, rank2):
    assert(len(rank1.items()) == len(rank2.items()))
    n = len(rank1.items())
    bi_d2 = [(bi, (rank1[bi] - rank2[bi])**2) for bi in rank1.keys()]
    d2_sum = sum([d2 for bi, d2 in bi_d2])
    # 排序以便得到排序变化比较大的bigram
    bi_d2.sort(key=lambda x:x[1], reverse=True)
    p = 1.0 - (6.0 * d2_sum) / (n * (n**2 - 1))
    return bi_d2, p
def pretty_print(spearman_list, spearman_p, method1, method2, p1, p2, freqdict):
    head = 10
    print("%s 方法与 %s 方法的spearman rank 系数为 %f" % (method1, method2, spearman_p))
    print("%s 方法与 %s 方法的排序变化大的前 %d 个bigram:" % (method1, method2, head))
    print("bigram\t\t\t\t\t\td²\t\t\t\t\t%s方法的概率\t\t\t%s方法的概率\t\t\tunnormed r\t\t\t%s方法的normed r\t\t\t%s方法的normed r"
          % (method1, method2, method1, method2))
    for i in range(head):
        bi = spearman_list[i][0]
        d_2 = spearman_list[i][1]
        print(bi, "\t\t", d_2, "\t\t", p1[bi], "\t\t", p2[bi], "\t\t", freqdict[bi], "\t\t", p1[bi] * freqdict.N(), "\t\t",
              p2[bi] * freqdict.N())

def spearman(bi_freqdict, laplace_P, deletd_P, sgt_P):
    bi_seen_vocabulary = bi_freqdict.keys()
    laplace_seen_P = {bi : laplace_P[bi] for bi in bi_seen_vocabulary}
    deleted_seen_P = {bi : deletd_P[bi] for bi in bi_seen_vocabulary}
    sgt_seen_P = {bi : sgt_P[bi] for bi in bi_seen_vocabulary}
    # 得到每个bigram在不同的smoothing策略下的概率的rank
    laplace_seen_rank = get_rank(laplace_seen_P)
    deleted_seen_rank = get_rank(deleted_seen_P)
    sgt_seen_rank = get_rank(sgt_seen_P)
    # 计算spearman等级系数
    spearman_laplace_deleted_list,  spearman_laplace_deleted_p = compute_spearman(laplace_seen_rank, deleted_seen_rank)
    pretty_print(spearman_laplace_deleted_list, spearman_laplace_deleted_p, LAPLACE, DELETED, laplace_bi_P, deleted_bi_P, bi_freqdict)
    spearman_laplace_sgt_list, spearman_laplace_sgt_p = compute_spearman(laplace_seen_rank, sgt_seen_rank)
    pretty_print(spearman_laplace_sgt_list, spearman_laplace_sgt_p, LAPLACE, SGT, laplace_bi_P, sgt_bi_P, bi_freqdict)
    spearman_deleted_sgt_list, spearman_deleted_sgt_p = compute_spearman(deleted_seen_rank, sgt_seen_rank)
    pretty_print(spearman_deleted_sgt_list, spearman_deleted_sgt_p, DELETED, SGT, deleted_bi_P, sgt_bi_P, bi_freqdict)


if __name__ == "__main__":
    uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr = get_r_Nr(TRAIN_AND_VALID)
    train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr = get_r_Nr(TRAIN)
    valid_uni_freqdict, valid_uni_r_Nr, valid_bi_freqdict, valid_bi_r_Nr = get_r_Nr(VALID)
    if(not os.path.exists(data_file)):
        os.mkdir(data_file)
    if(os.path.exists(laplace_p_file) and os.path.exists(laplace_condp_file)):
        laplace_bi_P = load(laplace_p_file)
        laplace_bi_P_cond = load(laplace_condp_file)
    else:
        laplace_bi_P, laplace_bi_P_cond = laplace(uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr)
        dump(laplace_p_file, laplace_bi_P)
        dump(laplace_condp_file, laplace_bi_P_cond)
    if(os.path.exists(deleted_p_file) and os.path.exists(deleted_condp_file)):
        deleted_bi_P = load(deleted_p_file)
        deleted_bi_P_cond = load(deleted_condp_file)
    else:
        deleted_bi_P, deleted_bi_P_cond = deleted(train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr,
                                              valid_uni_freqdict, valid_uni_r_Nr,
                                              valid_bi_freqdict, valid_bi_r_Nr,
                                            bi_freqdict, bi_r_Nr, uni_freqdict)
        dump(deleted_p_file, deleted_bi_P)
        dump(deleted_condp_file, deleted_bi_P_cond)
    if(os.path.exists(sgt_p_file) and os.path.exists(sgt_condp_file)):
        sgt_bi_P = load(sgt_p_file)
        sgt_bi_P_cond = load(sgt_condp_file)
    else:
        sgt_bi_P, sgt_bi_P_cond = sgt(uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr)
        dump(sgt_p_file, sgt_bi_P)
        dump(sgt_condp_file, sgt_bi_P_cond)
    spearman(bi_freqdict, laplace_bi_P, deleted_bi_P, sgt_bi_P)







