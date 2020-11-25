from config import *
import nltk
import os
# 把filepath指定的文件内的全角符号（除了中文汉字）转化为半角，并保存到同名文件中
def file_full2half(filepath):
    s = ""
    with open(filepath, "r", encoding="utf-8") as fin:
        s = fin.read()
    s = full2half(s)
    with open(filepath, "w", encoding="utf-8") as fout:
        fout.write(s)
    print("%s 全角到半角转化完成" % filepath)
def full2half(s):
    new = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        new.append(num)
    return ''.join(new)

def getTrainSet():
    with open("train.txt", "r", encoding="utf-8") as f_train:
        s_train = f_train.read()
    return s_train

def getValidSet():
    with open("valid.txt", "r", encoding="utf-8") as f_valid:
        s_valid = f_valid.read()
    return s_valid

def getTrainAndValidSet():
    with open("train.txt", "r", encoding="utf-8") as f_train:
        s_train = f_train.read()
    with open("valid.txt", "r", encoding="utf-8") as f_valid:
        s_valid = f_valid.read()
    s = [s_train, s_valid]
    s = "\n".join(s)
    return s

def getTestSet():
    with open("test.txt", "r", encoding="utf-8") as f_train:
        s_test = f_train.read()
    return s_test

def split_and_addBE(s):
    texts = []
    sentences = s.split("\n")
    for sent in sentences:
        if(sent == ""):
            continue
        words = sent.split(" ")
        words = [w for w in words if w != ""]
        words.insert(0, BEGIN)
        words.append(END)
        texts.extend(words)
    return texts

def split_sentence(sent):
    texts = split_and_addBE(sent)
    uni_freqdict = nltk.FreqDist(texts)
    bi_freqdict = nltk.FreqDist(nltk.bigrams(texts))
    uni_r_Nr = uni_freqdict.r_Nr()
    bi_r_Nr = bi_freqdict.r_Nr()
    # unigram中出现次数为0的就是unk
    uni_r_Nr[0] = 1
    # bigram中出现次数为0的是(unk, *), (*, unk), (unk, unk)
    bi_r_Nr[0] = (uni_freqdict.B() + 1) ** 2 - bi_freqdict.B()
    return texts, uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr

def get_r_Nr(type):
    functions = {TRAIN : getTrainSet,
                 TEST : getTestSet,
                 VALID: getValidSet,
                 TRAIN_AND_VALID : getTrainAndValidSet}
    uni_freqdict_files = {TRAIN: train_uni_freqdict_file,
                         TEST: test_uni_freqdict_file,
                         VALID: valid_uni_freqdict_file,
                         TRAIN_AND_VALID: train_valid_uni_freqdict_file}
    uni_r_Nr_files = {TRAIN: train_uni_r_Nr_file,
                         TEST: test_uni_r_Nr_file,
                         VALID: valid_uni_r_Nr_file,
                         TRAIN_AND_VALID: train_valid_uni_r_Nr_file}
    bi_freqdict_files = {TRAIN: train_bi_freqdict_file,
                         TEST : test_bi_freqdict_file,
                         VALID: valid_bi_freqdict_file,
                         TRAIN_AND_VALID: train_valid_bi_freqdict_file}
    bi_r_Nr_files = {TRAIN: train_bi_r_Nr_file,
                      TEST: test_bi_r_Nr_file,
                      VALID: valid_bi_r_Nr_file,
                      TRAIN_AND_VALID: train_valid_bi_r_Nr_file}
    uni_freqdict_file = uni_freqdict_files[type]
    uni_r_Nr_file = uni_r_Nr_files[type]
    bi_freqdict_file = bi_freqdict_files[type]
    bi_r_Nr_file = bi_r_Nr_files[type]
    if (not os.path.exists(data_file)):
        os.mkdir(data_file)
    if (os.path.exists(uni_freqdict_file) and os.path.exists(uni_r_Nr_file) and os.path.exists(
            bi_freqdict_file)
            and os.path.exists(bi_r_Nr_file)):
        uni_freqdict = load(uni_freqdict_file)
        uni_r_Nr = load(uni_r_Nr_file)
        bi_freqdict = load(bi_freqdict_file)
        bi_r_Nr =  load(bi_r_Nr_file)
        return uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr
    string = functions[type]()
    texts = split_and_addBE(string)
    uni_freqdict = nltk.FreqDist(texts)
    bi_freqdict = nltk.FreqDist(nltk.bigrams(texts))
    uni_r_Nr = uni_freqdict.r_Nr()
    bi_r_Nr = bi_freqdict.r_Nr()
    # unigram中出现次数为0的就是unk
    uni_r_Nr[0] = 1
    # bigram中出现次数为0的是(unk, *), (*, unk), (unk, unk)
    bi_r_Nr[0] = (uni_freqdict.B() + 1)**2 - bi_freqdict.B()
    # 存储这些中间结果
    dump(uni_freqdict_file, uni_freqdict)
    dump(uni_r_Nr_file, uni_r_Nr)
    dump(bi_freqdict_file, bi_freqdict)
    dump(bi_r_Nr_file, bi_r_Nr)
    return uni_freqdict, uni_r_Nr, bi_freqdict, bi_r_Nr

def main():
    files = ["train.txt", "test.txt", "valid.txt"]
    for f in files:
        file_full2half(f)

if __name__ == "__main__":
    main()