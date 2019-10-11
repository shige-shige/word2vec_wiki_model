from gensim.models import word2vec
import logging
import MeCab
import numpy as np


def make_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('./wiki_wakati.txt')

    model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
    model.save('./wiki.model')


def discover_similar_word():
    model = word2vec.Word2Vec.load('./wiki.model')
    results = model.wv.most_similar(positive=['ヤクルトスワローズ'])
    for result in results:
        print(result)


mt = MeCab.Tagger('')
mt.parse('')
model = word2vec.Word2Vec.load('./wiki.model')


def get_vector(text):
    sum_vec = np.zeros(200)
    word_count = 0
    node = mt.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            sum_vec += model.wv[node.surface]
            word_count += 1
        node = node.next
    return sum_vec / word_count


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == '__main__':
    # make_model()
    # discover_similar_word()

    sample1 = get_vector('昨日、金子はパスタを食べた。')
    sample2 = get_vector('金子は一昨日カレーを食べた')
    print('score： {}'.format(cos_sim(sample1, sample2)))
