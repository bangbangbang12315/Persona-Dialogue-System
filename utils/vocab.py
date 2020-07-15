# raw_data : weibo_corpus.cutted format data
# data: post.train format data
import sys
import json
import random
import os
from collections import Counter
from tqdm import tqdm

def load_raw_data(fp):
    raw_data = []
    with open(fp, 'r') as f:
        for line in tqdm(f):
            obj = json.loads(line)
            raw_data.append(obj)
    return raw_data

def get_data_vocab(data, vocab_num=40000):
    # <unk> <s> </s> first
    # format data/src_vocab_file
    vocab_dic = {}
    for text in data:
        words = text.split(' ')
        for w in words:
            if not w: continue
            vocab_dic[w] = vocab_dic.get(w, 0) + 1
    vocab = [k for k, _ in sorted(vocab_dic.items(), key=lambda x: -x[1])]
    vocab = ['<unk>', '<s>', '</s>'] + vocab[:vocab_num - 3]
    return vocab

def get_userID_vocab(data, vocab_num=80000):
    vocab_dic = Counter(data)
    vocab = [k for k, _ in sorted(vocab_dic.items(), key=lambda x: -x[1])]
    vocab = ['<unk>'] + vocab[:vocab_num-1]
    return vocab

def data2model_data(raw_data):
    post, resp = [], []
    post_userID, resp_userID = [], []

    for line in tqdm(raw_data):
        p,p_id,p_time,r,r_id,r_time,his = line
        post.append(p)
        resp.append(r)
        post_userID.append(p_id)
        resp_userID.append(r_id)

    assert len(post) == len(resp) == len(post_userID) == len(resp_userID)
    return post, resp, post_userID, resp_userID


def save_data(fp, data):
    print(f"Save {fp}")
    with open(fp, 'w') as f:
        for text in tqdm(data):
            f.write(text + '\n')

if __name__ == "__main__":
    raw_fp, save_dir, phase, vocab_size, userID_vocab_size = sys.argv[1:]
    save_dir = save_dir.rstrip('/')
    vocab_size = int(vocab_size)
    userID_vocab_size = int(userID_vocab_size)
    raw_data = load_raw_data(raw_fp)
    post, resp, post_userID, resp_userID = data2model_data(raw_data)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # save_data(f"{save_dir}/post.{phase}", post)
    # save_data(f"{save_dir}/resp.{phase}", resp)
    # save_data(f"{save_dir}/post.userID.{phase}", post_userID)
    # save_data(f"{save_dir}/resp.userID.{phase}", resp_userID)

    if phase == "train":
        if vocab_size:
            post_vocab = get_data_vocab(post, vocab_num=vocab_size)
            resp_vocab = get_data_vocab(resp, vocab_num=vocab_size)
            save_data(f"{save_dir}/src_vocab_file", post_vocab)
            save_data(f"{save_dir}/tgt_vocab_file", resp_vocab)

        if userID_vocab_size:
            post_userID_vocab = get_userID_vocab(post_userID, vocab_num=userID_vocab_size)
            resp_userID_vocab = get_userID_vocab(resp_userID, vocab_num=userID_vocab_size)
            save_data(f"{save_dir}/src_userID_vocab_file", post_userID_vocab)
            save_data(f"{save_dir}/tgt_userID_vocab_file", resp_userID_vocab)

    # if phase == "test":
    #     post, resps = formatter_test_data(raw_data)
    #     save_data(f'{save_dir}/post.test.eval', post)
    #     save_data(f'{save_dir}/resps.test.eval', resps)