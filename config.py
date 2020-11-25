import pickle
BEGIN = "<BOS>"
END = "<EOS>"
UNK = "UNK"

TRAIN = 0
TEST = 1
VALID = 2
TRAIN_AND_VALID = 3
data_file = "./data"
laplace_p_file = "data/laplace.p"
laplace_condp_file = "data/laplace_cond.p"
deleted_p_file = "data/deleted.p"
deleted_condp_file = "data/deleted_cond.p"
sgt_p_file = "data/sgt.p"
sgt_condp_file = "data/sgt_cond.p"

train_bi_freqdict_file = "data/train_bi.freqdict"
train_bi_r_Nr_file = "data/train_bi.rNr"
train_uni_freqdict_file = "data/train_uni.freqdict"
train_uni_r_Nr_file = "data/train_uni.rNr"

test_bi_freqdict_file = "data/test_bi.freqdict"
test_bi_r_Nr_file = "data/test_bi.rNr"
test_uni_freqdict_file = "data/test_uni.freqdict"
test_uni_r_Nr_file = "data/test_uni.rNr"

valid_bi_freqdict_file = "data/valid_bi.freqdict"
valid_bi_r_Nr_file = "data/valid_bi.rNr"
valid_uni_freqdict_file = "data/valid_uni.freqdict"
valid_uni_r_Nr_file = "data/valid_uni.rNr"

train_valid_bi_freqdict_file = "data/train_valid_bi.freqdict"
train_valid_bi_r_Nr_file = "data/train_valid_bi.rNr"
train_valid_uni_freqdict_file = "data/train_valid_uni.freqdict"
train_valid_uni_r_Nr_file = "data/train_valid_uni.rNr"


LAPLACE = "laplace"
DELETED = "deleted"
SGT = "simple-good-turing"

def load(filepath):
    with open(filepath, "rb") as p_file:
        P = pickle.load(p_file)
    print("%s load finish" % filepath)
    return P

def dump(filepath, P):
    with open(filepath, "wb") as Pfile:
        pickle.dump(P, Pfile)
    print("%s dump finish" % filepath)


