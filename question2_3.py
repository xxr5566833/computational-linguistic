from config import *
from text_tackle import *
from mysgt import sgt
from mydeleted import deleted
from mylaplace import laplace
import math
# 因为是bigram模型，给定firstword，预测secondword的概率以及在这个位置上的概率最大的前10个word
def predict(firstword, secondword, bi_freqdict, uni_freqdict, cond_p):
    # 如果firstword没在uni_freqdict中出现，说明firstword根本没在训练集里出现过，所以此时firstword就相当于UNK
    old_firstword = firstword
    if(firstword not in uni_freqdict.keys()):
        firstword = UNK
    real_cond_p = {bi[1] : p for bi, p in cond_p.items() if bi[0] == firstword}
    # 因为之前计算cond_p把一些没出现的bigram的对应条件概率省略了
    # 对于laplace，计算了出现的bigram对应的条件概率, P(*|UNK), P(UNK|*)
    # 对于deleted，计算了出现的bigram对应的条件概率，P(*|UNK), P(UNK|*)
    # 对于sgt，同上
    # 那么现在在real_cond_p中加入没有在cond_p中出现的词的条件概率，这个概率与P(UNK|firstword)相同
    for unigram in uni_freqdict.keys():
        if(real_cond_p.get(unigram) == None):
            real_cond_p[unigram] = real_cond_p[UNK]
    # 然后排序
    real_cond_p_list = list(real_cond_p.items())
    real_cond_p_list.sort(key=lambda x:x[1], reverse=True)
    real_cond_p_rank = {x[0]:i+1 for i, x in enumerate(real_cond_p_list)}
    # print(uni_freqdict.get(secondword))
    if(uni_freqdict.get(secondword) == None):
        # 如果secondword没在训练+验证中出现过，那么它就是UNK
        p_secondword = real_cond_p[UNK]
        rank = real_cond_p_rank[UNK]
    else:
        p_secondword = real_cond_p[secondword]
        rank = real_cond_p_rank[secondword]
    # 取排序的前十个
    head = 10
    return real_cond_p_list[0:head], p_secondword, rank


def compute_test_perplexity(bi_P, test_freqdict, train_valid_bi_freqdict, train_valid_uni_freqdict):
    p = 1.0
    # 先求句子/语料库的概率，然后对它做-1/n次幂
    n = test_freqdict.N()
    logsum = 0.
    for bi in test_freqdict.keys():
        w1 = bi[0]
        w2 = bi[1]
        if(train_valid_bi_freqdict.get(bi) == None):
            if(train_valid_uni_freqdict.get(w1) == None):
                w1 = UNK
                w2 = UNK
                # 如果w1是UNK，那么说明w1没在训练+验证里出现过，那么P(*|UNK)概率都是一样的，把*设置为UNK即可
            else:
                # 如果w1不是UNK，那么w1出现过，但是w1 w2没出现过，说明P(w2|w1) = P(UNK|w1)
                w2 = UNK
        # 如果bi在训练集里出现过，那么条件概率肯定是经过计算的
        newbi = (w1, w2)
        logsum -= test_freqdict[bi] * math.log(bi_P[newbi], 2)
        mid = ((bi_P[newbi])**test_freqdict[bi])
        p = p * ((bi_P[newbi])**test_freqdict[bi])
    perplexity = math.pow(2, logsum / (n + 1))
    return p, perplexity

def getCandidate(train_valid_freqdict, test_freqdict):
    d = set(test_freqdict.keys()) - set(train_valid_freqdict.keys())
    print(d)

if __name__ == "__main__":
    uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr = get_r_Nr(TRAIN_AND_VALID)
    train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr = get_r_Nr(TRAIN)
    valid_uni_freqdict, valid_uni_r_Nr, valid_bi_freqdict, valid_bi_r_Nr = get_r_Nr(VALID)
    test_uni_freqdict, test_uni_r_Nr, test_bi_freqdict, test_bi_r_Nr = get_r_Nr(TEST)
    # getCandidate(uni_freqdict, test_uni_freqdict)
    if (not os.path.exists(data_file)):
        os.mkdir(data_file)
    if (os.path.exists(laplace_p_file) and os.path.exists(laplace_condp_file)):
        laplace_bi_P = load(laplace_p_file)
        laplace_bi_P_cond = load(laplace_condp_file)
    else:
        laplace_bi_P, laplace_bi_P_cond = laplace(uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr)
        dump(laplace_p_file, laplace_bi_P)
        dump(laplace_condp_file, laplace_bi_P_cond)
    if (os.path.exists(deleted_p_file) and os.path.exists(deleted_condp_file)):
        deleted_bi_P = load(deleted_p_file)
        deleted_bi_P_cond = load(deleted_condp_file)
    else:
        deleted_bi_P, deleted_bi_P_cond = deleted(train_uni_freqdict, train_uni_r_Nr, train_bi_freqdict, train_bi_r_Nr,
                                                  valid_uni_freqdict, valid_uni_r_Nr,
                                                  valid_bi_freqdict, valid_bi_r_Nr,
                                                  bi_freqdict, bi_r_Nr, uni_freqdict)
        dump(deleted_p_file, deleted_bi_P)
        dump(deleted_condp_file, deleted_bi_P_cond)
    if (os.path.exists(sgt_p_file) and os.path.exists(sgt_condp_file)):
        sgt_bi_P = load(sgt_p_file)
        sgt_bi_P_cond = load(sgt_condp_file)
    else:
        sgt_bi_P, sgt_bi_P_cond = sgt(uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr)
        dump(sgt_p_file, sgt_bi_P)
        dump(sgt_condp_file, sgt_bi_P_cond)
    print("question 2:")
    sents = ["年轻人 不 讲 武德", "扶贫 开发 工作 取得 很 大 成绩 。", "我 相信 我 就是 我"]
    word_index = [4, 6, 2]
    for i, sent in enumerate(sents):
        text, sent_uni_freqdict, sent_uni_r_Nr, sent_bi_freqdict, sent_bi_r_Nr = split_sentence(sent)
        laplace_sent_p, laplace_sent_per = compute_test_perplexity(laplace_bi_P_cond, sent_bi_freqdict, bi_freqdict, uni_freqdict)
        deleted_sent_p, deleted_sent_per = compute_test_perplexity(deleted_bi_P_cond, sent_bi_freqdict, bi_freqdict, uni_freqdict)
        sgt_sent_p, sgt_sent_per = compute_test_perplexity(sgt_bi_P_cond, sent_bi_freqdict, bi_freqdict,
                                                                   uni_freqdict)
        laplace_top, laplace_p_secondword, laplace_rank = predict(text[word_index[i] - 1], text[word_index[i]],
                                                                  bi_freqdict, uni_freqdict, laplace_bi_P_cond)
        deleted_top, deleted_p_secondword, deleted_rank = predict(text[word_index[i] - 1], text[word_index[i]],
                                                                  bi_freqdict, uni_freqdict, deleted_bi_P_cond)
        sgt_top, sgt_p_secondword, sgt_rank = predict(text[word_index[i] - 1], text[word_index[i]],
                                                                  bi_freqdict, uni_freqdict, sgt_bi_P_cond)
        print("句子 %d：%s" % (i + 1, sent))
        print("特定位置为 %s 所在位置" % text[word_index[i]])

        print("laplace方法：")
        print("句子概率：", laplace_sent_p, "困惑度为：", laplace_sent_per)
        print("%s 所在位置概率最高的10个词及其概率：" % text[word_index[i]])
        for unigram, top_p in laplace_top:
            print(unigram, top_p)
        print("真实出现的词的预测概率为:", laplace_p_secondword, "排序为：", laplace_rank)

        print("deleted方法：")
        print("句子概率：", deleted_sent_p, "困惑度为：", deleted_sent_per)
        print("%s 所在位置概率最高的10个词及其概率：" % text[word_index[i]])
        for unigram, top_p in deleted_top:
            print(unigram, top_p)
        print("真实出现的词的预测概率为:", deleted_p_secondword, "排序为：", deleted_rank)

        print("simple-good-turing方法：")
        print("句子概率：", sgt_sent_p, "困惑度为：", sgt_sent_per)
        print("%s 所在位置概率最高的10个词及其概率：" % text[word_index[i]])
        for unigram, top_p in sgt_top:
            print(unigram, top_p)
        print("真实出现的词的预测概率为:", sgt_p_secondword, "排序为：", sgt_rank)

    print("句子1选定词的位置排序>20; 句子2选定词的位置排序<=3; 句子3选定词的位置排序>3且<=20")
    print("question 3:计算测试集的困惑度：")
    laplace_s_p, laplace_s_per = compute_test_perplexity(laplace_bi_P_cond, test_bi_freqdict, bi_freqdict, uni_freqdict)
    deleted_s_p, deleted_s_per = compute_test_perplexity(deleted_bi_P_cond, test_bi_freqdict, bi_freqdict, uni_freqdict)
    sgt_s_p, sgt_s_per = compute_test_perplexity(sgt_bi_P_cond, test_bi_freqdict, bi_freqdict, uni_freqdict)
    print("laplace方法的句子概率：", laplace_s_p, "困惑度为：", laplace_s_per)
    print("deleted方法的句子概率：", deleted_s_p, "困惑度为：", deleted_s_per)
    print("simple-good-turing方法的句子概率：", sgt_s_p, "困惑度为：", sgt_s_per)

