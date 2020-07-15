import gensim
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from gensim import similarities
TaggededDocument = gensim.models.doc2vec.TaggedDocument
EMBEDDING_SIZE=100
def get_datasest():
    with open("../data/model_data/similarity_corpus.txt", 'r') as cf: 
        docs = cf.readlines()
    x_train = []
    for i, text in tqdm(enumerate(docs)):
        word_list = text.strip().split(' ')
        l = len(word_list)
        # word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train
 
def train(x_train, size=100, epoch_num=4): 
    model_dm = Doc2Vec(x_train, min_count=1, window=5, size=size, sample=1e-3, negative=5, workers=20)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epoch_num)
    model_dm.save('model/model_dm') ##模型保存的位置
 
    return model_dm
 
def preprocess_sentence(sentence):
    while sentence and sentence[-1] == "<\s>":
        sentence.pop()
    return sentence



def eval(src,tgt,model_dm):
    with open(src,'r') as f_src, open(tgt,'w') as f_tgt:
        for line in tqdm(f_src, ncols=100, desc=f"{src.split('/')[-1]}"):
            line = line.rstrip().split('\t')
            vec_list = []
            for sentence in line:
                sentence = preprocess_sentence(sentence.split(' '))
                if sentence:
                    sen_vec = model_dm.infer_vector(sentence)
                    vec_list.append(' '.join(map(lambda x: f"{x:.8f}", sen_vec.tolist())))
            history_num = len(vec_list)     
            if history_num == 0:
                vec_list.append(' '.join(['0.0'] * EMBEDDING_SIZE))
            f_tgt.write('\t'.join(vec_list) + "\n")

def eval_post(src,tgt,model_dm):
    with open(src,'r') as f_src, open(tgt,'w') as f_tgt:
        for line in tqdm(f_src, ncols=100, desc=f"{src.split('/')[-1]}"):
            sentence = line.rstrip()
            sentence = sentence.split(' ')
            if sentence:
                sen_vec = model_dm.infer_vector(sentence)
                vec = ' '.join(map(lambda x: f"{x:.8f}", sen_vec.tolist()))
            f_tgt.write(vec + "\n")

def test(model_dm,x_train,test_text):
    inferred_vector_dm = model_dm.infer_vector(test_text)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    for cnt,sim in sims:
        print(x_train[cnt],sim)


    
if __name__ == '__main__':
    # x_train = get_datasest()
    # model_dm = train(x_train)
    src_dir = "../data/model_data"
    model_dm = Doc2Vec.load("model/model_dm")


    # test_text = ['今天','打卡','故宫','！']
    # test(model_dm,x_train,test_text)
 
    # exit(0)
    for phase in ['train', 'dev','test']:
        eval_post(f"{src_dir}/{phase}/post.{phase}", f"../data/model_data/{phase}/post_vec.{phase}",model_dm)
 