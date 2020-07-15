import os
import sys
import json
from tqdm import tqdm

# from sentence_similarity import similarity
from multi_process_wraper import Worker, MultiProcessor

def get_history_rank(query, histories, model='cosine'):
    scores = dict()

    query_seg = query.split(' ')
    for key in histories:
        key_seg = key.split(' ')
        sim = similarity.ssim(query_seg, key_seg, model=model)
        try: 
            sim = 1.0 * sim
        except:
            sim = 0
        scores[key] = scores.get(key, 0) + sim

    return scores

def limit_history(post, resp, his, topK=15):
    # his_p, his_p_id, his_p_time, his_r, his_r_id, his_r_time = zip(*his)
    idx = {}
    his_post, his_resp = [], []
    for i, item in enumerate(his):
        h_p, h_r = item[0], item[3]
        idx[(h_p, h_r)] = i
        his_post.append(h_p)
        his_resp.append(h_r)

    model = 'cosine'
    sim_p_p = get_history_rank(post, his_post, model=model)
    sim_p_r = get_history_rank(post, his_resp, model=model)

    result = {}
    for his_p, his_r in zip(his_post, his_resp):
        result[(his_p, his_r)] = sim_p_p[his_p] + sim_p_r[his_r]
    top = sorted(result.items(),key=lambda x:x[1],reverse=True)
    result = top[:topK]
    new_his, sim_val = [], []
    for (h_p, h_r), v in result:
        new_his.append(his[idx[(h_p, h_r)]])
        sim_val.append(v)
    
    return new_his, sim_val

def parse_line_sim(line):
    def padded(seqs):
        new_seqs = []
        seqs_len = list(map(len, seqs))
        pad_len = max(seqs_len)
        for seq in seqs:
            new_seqs.append(seq + ['<\s>'] * (pad_len - len(seq)))
        assert len(new_seqs) == len(seqs_len)
        new_seqs = '\t'.join(map(lambda x: ' '.join(x), new_seqs))
        seqs_len = ' '.join(map(str, seqs_len))
        return new_seqs, seqs_len


    line = json.loads(line)
    p, p_id, p_time, r, r_id, r_time, his = line
    his = json.loads(his)
    his_num = len(his)
    if his:
        his, his_sim_val = limit_history(p, r, his)
        his_p, his_p_id, his_p_time, his_r, his_r_id, his_r_time = zip(*his)
        his_p = list(map(lambda x: x.strip().split(' '), his_p))
        his_r = list(map(lambda x: x.strip().split(' '), his_r))
        his_p, his_p_len = padded(his_p)
        his_r, his_r_len = padded(his_r)
        his_p_id = ' '.join(his_p_id)
        his_p_time = ' '.join(his_p_time)
        his_r_id = ' '.join(his_r_id)
        his_r_time = ' '.join(his_r_time)
        his_sim_val = ' '.join(map(lambda x: f"{x:.4f}", his_sim_val))
    else:
        his_p, his_p_len, his_r, his_r_len = "<\s>", "0", "<\s>", "0"
        his_p_id, his_p_time, his_r_id, his_r_time = "-1", "-1", "-1", "-1"
        his_sim_val = "0.0"

    new_line = [p, p_id, p_time, r, r_id, r_time, his_p, his_p_len, his_p_id, his_p_time, his_r, his_r_len, his_r_id, his_r_time, his_sim_val, his_num]
    return json.dumps(new_line, ensure_ascii=False)

def parse_line_time(line):
    def padded(seqs):
        new_seqs = []
        seqs_len = list(map(len, seqs))
        pad_len = max(seqs_len)
        for seq in seqs:
            new_seqs.append(seq + ['<\s>'] * (pad_len - len(seq)))
        assert len(new_seqs) == len(seqs_len)
        new_seqs = '\t'.join(map(lambda x: ' '.join(x), new_seqs))
        seqs_len = ' '.join(map(str, seqs_len))
        return new_seqs, seqs_len


    line = json.loads(line)
    p, p_id, p_time, r, r_id, r_time, his = line
    his = json.loads(his)
    his_num = len(his)
    if his:
        # his, his_sim_val = limit_history(p, r, his)
        his = his[max(0, len(his)-15):]
        his_p, his_p_id, his_p_time, his_r, his_r_id, his_r_time = zip(*his)
        his_p = list(map(lambda x: x.strip().split(' '), his_p))
        his_r = list(map(lambda x: x.strip().split(' '), his_r))
        his_p, his_p_len = padded(his_p)
        his_r, his_r_len = padded(his_r)
        his_p_id = ' '.join(his_p_id)
        his_p_time = ' '.join(his_p_time)
        his_r_id = ' '.join(his_r_id)
        his_r_time = ' '.join(his_r_time)
        his_sim_val = 0.0
    else:
        his_p, his_p_len, his_r, his_r_len = "<\s>", "0", "<\s>", "0"
        his_p_id, his_p_time, his_r_id, his_r_time = "-1", "-1", "-1", "-1"
        his_sim_val = "0.0"

    new_line = [p, p_id, p_time, r, r_id, r_time, his_p, his_p_len, his_p_id, his_p_time, his_r, his_r_len, his_r_id, his_r_time, his_sim_val, his_num]
    return new_line


def transform(src_fp, tgt_dir, tgt_postfix, mode="cosine"):
    os.system(f"mkdir {tgt_dir}")
    os.system(f"mkdir {tgt_dir}/extra")
    with open(src_fp, 'r') as f_src, \
        open(f"{tgt_dir}/post.{tgt_postfix}", 'w') as f_p, \
        open(f"{tgt_dir}/resp.{tgt_postfix}", 'w') as f_r, \
        open(f"{tgt_dir}/his_post_padded.{tgt_postfix}", 'w') as f_his_p, \
        open(f"{tgt_dir}/his_resp_padded.{tgt_postfix}", 'w') as f_his_r, \
        open(f"{tgt_dir}/his_post_len.{tgt_postfix}", 'w') as f_his_p_len, \
        open(f"{tgt_dir}/his_resp_len.{tgt_postfix}", 'w') as f_his_r_len, \
        open(f"{tgt_dir}/extra/post_time.{tgt_postfix}", 'w') as f_p_time, \
        open(f"{tgt_dir}/extra/resp_time.{tgt_postfix}", 'w') as f_r_time, \
        open(f"{tgt_dir}/extra/post_id.{tgt_postfix}", 'w') as f_p_id, \
        open(f"{tgt_dir}/extra/resp_id.{tgt_postfix}", 'w') as f_r_id, \
        open(f"{tgt_dir}/extra/his_post_time.{tgt_postfix}", 'w') as f_his_p_time, \
        open(f"{tgt_dir}/extra/his_resp_time.{tgt_postfix}", 'w') as f_his_r_time, \
        open(f"{tgt_dir}/extra/his_post_id.{tgt_postfix}", 'w') as f_his_p_id, \
        open(f"{tgt_dir}/extra/his_resp_id.{tgt_postfix}", 'w') as f_his_r_id, \
        open(f"{tgt_dir}/extra/his_num.{tgt_postfix}", 'w') as f_his_num, \
        open(f"{tgt_dir}/extra/his_sim_val.{tgt_postfix}", 'w') as f_his_sim_val:
        for line in tqdm(f_src, ncols=50):
            # p, p_id, p_time, r, r_id, r_time, his_p, his_p_len, his_p_id, his_p_time, his_r, his_r_len, his_r_id, his_r_time, his_sim_val, his_num = parse_line(line)
            if mode == "cosine":
                p, p_id, p_time, r, r_id, r_time, his_p, his_p_len, his_p_id, his_p_time, his_r, his_r_len, his_r_id, his_r_time, his_sim_val, his_num = json.loads(line)
            if mode == "time":
                p, p_id, p_time, r, r_id, r_time, his_p, his_p_len, his_p_id, his_p_time, his_r, his_r_len, his_r_id, his_r_time, his_sim_val, his_num = parse_line_time(line)
            
            if not his_p:
                his_p, his_p_len, his_r, his_r_len = "<\s>", "0", "<\s>", "0"
                his_p_id, his_p_time, his_r_id, his_r_time = "-1", "-1", "-1", "-1"
                his_sim_val = "0.0"

            # assert len(his_p.split('\t')) == len(his_p_len.split(' ')) == len(his_p_id.split(' ')) == len(his_p_time.split(' ')) == len(his_r.split('\t')) == len(his_r_len.split(' ')) == len(his_r_id.split(' ')) == len(his_r_time.split(' ')) == len(his_sim_val.split(' '))

            f_p.write(f"{p}\n")
            f_r.write(f"{r}\n")
            f_his_p.write(f"{his_p}\n")
            f_his_r.write(f"{his_r}\n")
            f_his_p_len.write(f"{his_p_len}\n")
            f_his_r_len.write(f"{his_r_len}\n")

            f_p_time.write(f"{p_time}\n")
            f_r_time.write(f"{r_time}\n")
            f_p_id.write(f"{p_id}\n")
            f_r_id.write(f"{r_id}\n")
            f_his_p_id.write(f"{his_p_id}\n")
            f_his_r_id.write(f"{his_r_id}\n")
            f_his_p_time.write(f"{his_p_time}\n")
            f_his_r_time.write(f"{his_r_time}\n")
            f_his_num.write(f"{his_num}\n")
            f_his_sim_val.write(f"{his_sim_val}\n")

def multi_trans(src_fp, tgt_fp):
    worker = Worker(src_fp, tgt_fp, parse_line_sim)
    mp = MultiProcessor(worker, 15)
    mp.run()
    print("All Processes Done.")
    worker.merge_result(keep_pid_file=False)

if __name__ == "__main__":
    # src_dir = "../raw"
    src_dir = "../data/small/raw"
    for phase in ['test', 'train', 'dev']:
        transform(f"{src_dir}/{phase}.raw", f"../data/small/{phase}", f"{phase}", mode="time")
    #     multi_trans(f"{src_dir}/{phase}.raw", f"{src_dir}/{phase}.limited")
    #     transform(f"{src_dir}/{phase}.limited", f"../clean_news_400W/cosine/{phase}", f"{phase}", mode="cosine")
        # transform(f"../raw/{phase}.limited", f"../cosine/{phase}", f"{phase}", mode="cosine")
        # transform(f"../raw/{phase}.raw", f"../time/{phase}", f"{phase}", mode="time")
    # transform(f"../raw/test.raw.shuf50", f"../test", f"test")
    # multi_trans(f"../raw/test.raw", f"../raw/test.limited")
    # transform(f"../raw/test.limited", f"../test", f"test")
    # transform(f"{src_dir}/dev.limited.40000", f"../clean_news_400W/cosine/dev", f"dev", mode="cosine")
    # transform(f"{src_dir}/test.limited.5000", f"../clean_news_400W/cosine/test", f"test", mode="cosine")
    # transform(f"{src_dir}/same_post.limited", f"../clean_news_400W/cosine/same_post", f"same_post", mode="cosine")


