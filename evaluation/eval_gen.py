# -*- coding: utf-8 -*-
# @Author   : Just-silent Ygg
# @time     : 2020/3/25 9:16
import copy
import math
import six
import os
import re
import numpy as np
import time
import subprocess
import tempfile
import logging
import importlib
import types

from collections import defaultdict
from six.moves import xrange as range


def _strip(s):
    return s.strip()


class Bleu(object):
    """ Bleu score. """
    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"

    # special_reflen is used in oracle (proportional effective ref len for a node).

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        """ singular instance """
        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def bleu_precook(self, s, n=4, out=False):
        words = s.split()
        counts = defaultdict(int)
        for k in range(1, n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i + k])
                counts[ngram] += 1
        return (len(words), counts)

    def bleu_cook_refs(self, refs, eff=None, n=4):
        reflen = []
        maxcounts = {}
        for ref in refs:
            rl, counts = self.bleu_precook(ref, n)
            reflen.append(rl)
            for (ngram, count) in six.iteritems(counts):
                maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
        if eff == "shortest":
            reflen = min(reflen)
        elif eff == "average":
            reflen = float(sum(reflen)) / len(reflen)
        return (reflen, maxcounts)

    def bleu_cook_test(self, test, reflen_refmaxcounts, eff=None, n=4):
        reflen, refmaxcounts = reflen_refmaxcounts
        testlen, counts = self.bleu_precook(test, n, True)
        result = {}
        if eff == "closest":
            result["reflen"] = min((abs(l - testlen), l) for l in reflen)[1]
        else:  ## i.e., "average" or "shortest" or None
            result["reflen"] = reflen
        result["testlen"] = testlen
        result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]
        result['correct'] = [0] * n
        for (ngram, count) in six.iteritems(counts):
            result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)
        return result

    def copy(self):
        """ copy the refs. """
        new = Bleu(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def cook_append(self, test, refs):
        """ called by constructor and __iadd__ to avoid creating new instances."""
        if refs is not None:
            self.crefs.append(self.bleu_cook_refs(refs))
            if test is not None:
                cooked_test = self.bleu_cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test)  # N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match
        self._score = None  # need to recompute

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        """
        :param option:
        :return: (bleu, len_ratio) pair
        """
        return (self.fscore(option=option), self.ratio(option=option))

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(self.bleu_cook_test(t, rs))
        self._score = None

        return self

    def rescore(self, new_test):
        """ replace test(s) with new test(s), and returns the new score. """
        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        """ add an instance (e.g., from another sentence). """
        if type(other) is tuple:
            ## avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None  ## need to recompute
        return self

    def compatible(self, other):
        return isinstance(other, Bleu) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):
        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = min((abs(l - testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option
        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15  ## so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]
        if self._score is not None:
            return self._score
        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"
        self._testlen = 0
        self._reflen = 0
        totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}

        # for each sentence
        for comps in self.ctest:
            testlen = comps['testlen']
            self._testlen += testlen
            if self.special_reflen is None:  ## need computation
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen
            self._reflen += reflen
            for key in ['guess', 'correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        / (float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small)  ## N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) \
                    / (totalcomps['guess'][k] + small)
            bleus.append(bleu ** (1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen + small)  ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1 / ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._score = bleus
        return self._score, bleu_list


class LazyLoader(types.ModuleType):

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super(LazyLoader, self).__init__(name)

    def _load(self):
        """ Load the module and insert it into the parent's globals. """

        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


class MultiBleu(object):

    def __init__(self, multi_bleu_path=None):
        """ singular instance """
        self.multi_bleu_path = multi_bleu_path

    def get_moses_multi_bleu(self, hypotheses, references, lowercase=False):
        """Get the BLEU score using the moses `multi-bleu.perl` script.

        **Script:**
        https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

        Args:
            hypotheses (list of str): List of predicted values
            references (list of str): List of target values
            lowercase (bool): If true, pass the "-lc" flag to the `multi-bleu.perl` script

        Returns:
            (:class:`np.float32`) The BLEU score as a float32 value.

        Example:
            hypotheses = [
                "The brown fox jumps over the dog ",
                "The brown fox jumps over the dog 2 笑"
            ]
            references = [
                "The quick brown fox jumps over the lazy dog 笑",
                "The quick brown fox jumps over the lazy dog 笑"
             ]
            get_moses_multi_bleu(hypotheses, references, lowercase=True)
            :return: score=46.51
        """
        six = LazyLoader('six', globals(), 'six')
        logger = logging.getLogger(__name__)
        if isinstance(hypotheses, list):
            hypotheses = np.array(hypotheses)
        if isinstance(references, list):
            references = np.array(references)

        if np.size(hypotheses) == 0:
            return np.float32(0.0)

        # Get MOSES multi-bleu script
        try:
            multi_bleu_path = self.multi_bleu_path
            os.chmod(multi_bleu_path, 0o755)
        except:
            logger.warning("Unable to fetch multi-bleu.perl script")
            return None

        # Dump hypotheses and references to tempfiles
        hypothesis_file = tempfile.NamedTemporaryFile()
        hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
        hypothesis_file.write(b"\n")
        hypothesis_file.flush()
        reference_file = tempfile.NamedTemporaryFile()
        reference_file.write("\n".join(references).encode("utf-8"))
        reference_file.write(b"\n")
        reference_file.flush()

        # Calculate BLEU using multi-bleu script
        with open(hypothesis_file.name, "r") as read_pred:
            bleu_cmd = [multi_bleu_path]
            if lowercase:
                bleu_cmd += ["-lc"]
            bleu_cmd += [reference_file.name]
            try:
                bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
                bleu_out = bleu_out.decode("utf-8")
                bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
                bleu_score = float(bleu_score)
                bleu_score = np.float32(bleu_score)
            except subprocess.CalledProcessError as error:
                if error.output is not None:
                    logger.warning("multi-bleu.perl script returned non-zero exit code")
                    logger.warning(error.output)
                bleu_score = None

        # Close temp files
        hypothesis_file.close()
        reference_file.close()

        return bleu_score


class Cider(object):
    """Cider score"""

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def precook(s, n=4, out=False):
        words = s.split()
        counts = defaultdict(int)
        for k in range(1, n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i + k])
                counts[ngram] += 1
        return counts

    def cook_refs(self, refs, n=4):
        return [self.cider_precook(ref, n) for ref in refs]

    def copy(self):
        new = Cider(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def cider_cook_refs(self, refs, n=4):
        return [self.cider_precook(ref, n) for ref in refs]

    def cook_test(self, test, n=4):
        return self.cider_precook(test, n, True)

    def cider_precook(self, s, n=4, out=False):
        words = s.split()
        counts = defaultdict(int)
        for k in range(1, n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i + k])
                counts[ngram] += 1
        return counts

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(self.cider_cook_refs(refs))
            if test is not None:
                self.ctest.append(self.cook_test(test))  # N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in six.iteritems(ref)]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in six.iteritems(cnts):
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in six.iteritems(vec_hyp[n]):
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                assert (not math.isnan(val[n]))
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        self.ref_len = np.log(float(len(self.crefs)))
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        self.compute_doc_freq()
        assert (len(self.ctest) >= max(self.document_frequency.values()))
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)


class Rouge():
    def __init__(self):
        self.beta = 1.2

    def my_lcs(self, string, sub):
        if (len(string) < len(sub)):
            sub, string = string, sub
        lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]
        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if (string[i - 1] == sub[j - 1]):
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[len(string)][len(sub)]

    def calc_score(self, candidate, refs):
        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        prec = []
        rec = []
        token_c = candidate[0].split(" ")
        for reference in refs:
            token_r = reference.split(" ")
            lcs = self.my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))
        prec_max = max(prec)
        rec_max = max(rec)
        if (prec_max != 0 and rec_max != 0):
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        score = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            score.append(self.calc_score(hypo, ref))
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)


class VecLoder():
    def __init__(self):
        pass

    def conver_float(self, x):
        '''将词向量数据类型转换成可以计算的浮点类型'''
        float_str = x
        return [float(f) for f in float_str]

    def process_wordembe(self, path):
        '''
        将词向量文件中的所有词向量存放到一个列表lines里
        :param path: a path of english word embbeding file 'glove.840B.300d.txt'
        :return: a list, element is a 301 dimension word embbeding, it's form like this
                ['- 0.12332 ... -0.34542\n', ', 0.23421 ... -0.456733/n', ..., 'you 0.34521 0.78905 ... -0.23123/n']
        '''
        # f = open(path, 'r', encoding='utf-8')
        # embed_lines = f.readlines()
        list1 = []
        list2 = []
        with open('data/glove.840B.300d.txt', 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            for line in lines:
                list_line = line.split()
                list1.append(list_line[0])
                list2.append(list_line[1:])
        dict_word = dict(zip(list1, list2))
        # dict_word = KeyedVectors.load_word2vec_format('data/glove.840B.300d.word2vec.txt', binary=False)
        return dict_word

    def word2vec(self, x, lines):
        '''
        将一个字符串(这里指句子）中所有的词都向量化，并存放到一个列表里
        :param x: a sentence/sequence, type is string, for example 'hello, how are you ?'
        :return: a list, the form like [[word_vector1],...,[word_vectorn]], save per word embbeding of a sentence.
        '''
        x = x.split()[:-1]
        x_words = []
        for w in x:
            for line in lines:
                # print(line)
                if w == line.split()[0]:  # 将词向量按空格切分到一个列表里，将列表的第一个词与x的word比较
                    # print(w)
                    x_words.append(self.conver_float(line[:-1].split()[1:]))  # 若在词向量列表中找到对应的词向量，添加到x_words列表里
                    break
        return x_words


class EmbeddingAverage():
    """ Embedding Average cosine similarity """

    def __init__(self):
        self.vecloder = VecLoder()

    def sentence_embedding(self, x_words):
        '''
        上面的第一个公式：computing sentence embedding by computing average of all word embeddings of sentence.
        :param x: a sentence, type is string.
        :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
        :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
        :return: a scalar, it's value is in [0, 1]
        '''
        sen_embed = np.array([0 for _ in range(len(x_words[0]))])  # 存放句向量
        # print(len(sen_embed))
        for x_v in x_words:
            x_v = np.array(x_v).astype('float')
            # print(len(x_v))
            sen_embed = np.add(x_v, sen_embed)
        sen_embed = sen_embed / math.sqrt(sum(np.square(sen_embed)))
        return sen_embed

    def cosine_similarity(self, x, y, norm=False):
        """ 向量均值法EA:计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = np.array([0 for _ in range(len(x))])
        # print(zero_list)
        if x.all() == zero_list.all() or y.all() == zero_list.all():
            return float(1) if x == y else float(0)

        # method 1
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    def get_score_emb(self, x_words, y_words):
        x_emb = self.sentence_embedding(x_words)
        y_emb = self.sentence_embedding(y_words)
        embedding_average = self.cosine_similarity(x_emb, y_emb)
        return embedding_average


class VectorExtrema():
    """ Vector Extrema cosine similarity """

    def __init__(self, path):
        self.path = path
        self.vecloder = VecLoder()

    def conver_float_list(self, x):
        '''将词向量数据类型转换成可以计算的浮点类型'''
        list = []
        for float_str in x:
            list.append([float(f) for f in float_str])
        return list

    def vector_extrema(self, x_words):
        '''
        对应公式部分：computing vector extrema by compapring maximun value of all word embeddings in same dimension.
        :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
        :return: a 300 dimension list, vector extrema
        '''
        # print(np.array(x_words).shape)
        vec_extre = np.max(np.array(x_words), axis=0)
        return vec_extre

    def cosine_similarity(self, x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = np.array([0 for _ in range(len(x))])
        # print(zero_list)
        if x.all() == zero_list.all() or y.all() == zero_list.all():
            return float(1) if x == y else float(0)

        # method 1
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    def get_score_vec(self, x_words, y_words):
        x_words = self.conver_float_list(x_words)
        y_words = self.conver_float_list(y_words)
        vec_x = self.vector_extrema(x_words)
        vec_y = self.vector_extrema(y_words)
        similarity = self.cosine_similarity(vec_x, vec_y)
        return similarity


class GreedyMatching():
    def __init__(self, path):
        self.path = path
        self.vecloder = VecLoder()

    def conver_float_list(self, x):
        '''将词向量数据类型转换成可以计算的浮点类型'''
        list = []
        for float_str in x:
            list.append([float(f) for f in float_str])
        return list

    def cosine_similarity(self, x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)

        # method 1
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    def greedy(self, x, x_words, y_words):
        '''
        上面提到的第一个公式
        :param x: a sentence, type is string.
        :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
        :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
        :return: a scalar, it's value is in [0, 1]
        '''
        cosine = []  # 存放一个句子的一个词与另一个句子的所有词的余弦相似度
        sum_x = 0  # 存放最后得到的结果
        for x_v in x_words:
            for y_v in y_words:
                cosine.append(self.cosine_similarity(x_v, y_v))
            if cosine:
                sum_x += max(cosine)
                cosine = []
        sum_x = sum_x / len(x.split()[:-1])
        return sum_x

    def greedy_match(self, sum_x, sum_y):
        score = (sum_x + sum_y) / 2
        return score

    def get_score_gre(self, x, y, x_words, y_words):
        x_words = self.conver_float_list(x_words)
        y_words = self.conver_float_list(y_words)
        sum_x = self.greedy(x, x_words, y_words)
        sum_y = self.greedy(y, y_words, x_words)
        score = self.greedy_match(sum_x, sum_y)
        return score


class Evaluation_Gen():
    """ NLG-Evaluation"""

    def __init__(self, vector_path=None, hyp_path=None, ref_path=None, ref_paths=None, q_path=None, a_path=None,
                 n_gram=0, sigma=0, is_vec=False):
        self.vector_path = vector_path
        self.hyp_path = hyp_path
        self.ref_path = ref_path
        self.ref_paths = ref_paths
        self.q_path = q_path
        self.a_path = a_path
        self._n = n_gram
        self._sigma = sigma
        if is_vec:
            starttime = time.time()
            vecloder = VecLoder()
            self.ques, self.ans = self.get_vector_list()
            embed_lines = vecloder.process_wordembe(self.vector_path)
            endtime = time.time()
            print('加载时间:', endtime - starttime)
            starttime = time.time()
            self.x_embs = []
            self.y_embs = []
            for i in range(len(self.ques)):
                score_emb = 0
                que = self.ques[i][0]
                an = self.ans[i][0]
                x_words = vecloder.word2vec(que, embed_lines)
                y_words = vecloder.word2vec(an, embed_lines)
                self.x_embs.append(x_words)
                self.y_embs.append(y_words)
            endtime = time.time()
            print('转换vec时间:', endtime - starttime)

    def get_list(self):
        with open(self.hyp_path, 'r', encoding='UTF-8') as f:
            hyp_list = f.readlines()
        ref_list = []
        for iidx, reference in enumerate(self.ref_paths):
            with open(reference, 'r', encoding='UTF-8') as f:
                ref_list.append(f.readlines())
        ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
        assert len(refs) == len(hyps)
        return refs, hyps

    def get_bleu(self):
        gts, res = self.get_list()
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        bleu = Bleu(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)
            bleu += (hypo[0], ref)
        score, scores = bleu.compute_score(option='closest', verbose=0)
        names = ['BLEU_1:', 'BLEU_2:', 'BLEU_3:', 'BLEU_4:']
        for i, sco in enumerate(score):
            print('{}\t{}'.format(names[i], sco))

    def get_multi_bleu(self):
        multi_bleu = MultiBleu('data/multi-bleu.perl')
        with open(self.hyp_path, 'r', encoding='UTF-8') as f:
            hyp_list = f.readlines()
        with open(self.ref_path, 'r', encoding='UTF-8') as f:
            ref_list = f.readlines()
        assert len(hyp_list) == len(ref_list)
        hyps = [l.strip() for l in hyp_list]
        refs = [l.strip() for l in ref_list]
        # hypotheses = [
        #     "The brown fox jumps over the dog ",
        #     "The brown fox jumps over the dog 2 笑"
        # ]
        # references = [
        #     "The quick brown fox jumps over the lazy dog 笑",
        #     "The quick brown fox jumps over the lazy dog 笑"
        #  ]
        # output: 46.51
        print('Multi_BLEU:', multi_bleu.get_moses_multi_bleu(hyps, refs, lowercase=True))

    def get_rouge(self):
        gts, res = self.get_list()
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        rough = Rouge()
        score, scores = rough.compute_score(gts, res)
        print('ROUGE_L:{}'.format(score))

    def get_cider(self):
        gts, res = self.get_list()
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        cider = Cider(n=self._n, sigma=self._sigma)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

            cider += (hypo[0], ref)

        (score, scores) = cider.compute_score()
        print('CIDEr:\t{}'.format(score))

    def get_vector_list(self):
        ques = []
        ans = []
        with open(self.q_path) as q:
            for line in q.readlines():
                ques.append([line])
        with open(self.a_path) as a:
            for line in a.readlines():
                ans.append([line])
        return ques, ans

    def get_embedding_average_cosine_similarity(self):
        ques, ans = self.get_vector_list()
        scores_emb = 0
        emb = EmbeddingAverage()
        for i in range(len(ques)):
            score_emb = 0
            score_emb = emb.get_score_emb(self.x_embs[i], self.y_embs[i])
            scores_emb += score_emb
        print('Embedding Average Cosine Similarity:', scores_emb / len(ques))

    def get_vector_extrema_cosine_similarity(self):
        ques, ans = self.get_vector_list()
        scores_vec = 0
        vec = VectorExtrema(self.vector_path)
        for i in range(len(ques)):
            score_emb = 0
            score_vec = vec.get_score_vec(self.x_embs[i], self.y_embs[i])
            scores_vec += score_vec
        print('Vector Extrema Cosine Similarity:', scores_vec / len(ques))

    def get_greedy_matching(self):
        ques, ans = self.get_vector_list()
        scores_gre = 0
        gre = GreedyMatching(self.vector_path)
        for i in range(len(ques)):
            score_emb = 0
            score_gre = gre.get_score_gre(self.ans[i][0], self.ques[i][0], self.x_embs[i], self.y_embs[i])
            scores_gre += score_gre
        print('Greedy Matching:', scores_gre / len(ques))


if __name__ == '__main__':
    path = 'data/'
    hyp_path = path + 'hyp.txt'
    ref_path = path + 'ref1.txt'
    ref_paths = [path + 'ref1.txt', path + 'ref2.txt']
    q_path = path + 'ques.txt'
    a_path = path + 'ans.txt'
    vector_path = path + 'glove.840B.300d.txt'

    evaluation = Evaluation_Gen(vector_path, hyp_path, ref_path, ref_paths, q_path, a_path, n_gram=4, sigma=6.0,
                                is_vec=False)
    evaluation.get_bleu()
    # evaluation.get_multi_bleu()   #此方法需在linux环境下运行
    evaluation.get_rouge()
    evaluation.get_cider()
    # 以下三个方法需要预加载vector_path, is_vec=True
    # evaluation.get_embedding_average_cosine_similarity()
    # evaluation.get_vector_extrema_cosine_similarity()
    # evaluation.get_greedy_matching()
