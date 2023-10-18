import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm.autonotebook import tqdm, trange
from typing import Iterable, Tuple, Union, Generator
from .multiprocessing import ArgumentQueue

####################################################################################################
# Span Type:                                                                                       #
####################################################################################################

class SpanIterator:
    def __init__(self, target:'SpanCollection'):
        self.__target = target
        self.__i = -1

    def __next__(self) -> slice:
        self.__i += 1

        if self.__i >= len(self.__target):
            raise StopIteration

        return self.__target[self.__i]

class SpanCollection:
    def __init__(self, obj:Union[Iterable[Tuple[int, int]],Iterable[slice],None]=None, buffer_size:int=5) -> None:
        # get length:
        if obj is None: self.__len = 0
        else:           self.__len = len(obj)

        # init array:
        self.__limits = np.empty((self.__len + buffer_size, 2), dtype=int)

        # fill array:
        if obj is not None:
            for i, element in enumerate(obj):
                if isinstance(element, slice):
                    self.__limits[i,0] = element.start
                    self.__limits[i,1] = element.stop
                else:
                    self.__limits[i] = element

        # sort array:
        self.__sort_active = True
        self._sort()

    def __getitem__(self, idx:Union[Tuple[Union[Iterable[int],Iterable[bool],slice,int],Union[Iterable[int],Iterable[bool],slice,int]],slice,int]):# -> Union['SpanCollection', slice, Iterable[int], int]:
        if isinstance(idx, int):
            return slice(*self.__limits[:self.__len][idx])
        
        if isinstance(idx, tuple):
            return self.__limits[:self.__len][idx]
        
        return SpanCollection(self.__limits[:self.__len][idx])

    def __len__(self) -> int:
        return self.__len

    def __iter__(self) -> SpanIterator:
        return SpanIterator(self)

    def __repr__(self) -> str:
        return '|'.join([f'({start:d},{stop-1:d})' for start, stop in self.__limits[:self.__len]])

    def __add__(self, other:Union['SpanCollection', Iterable[Tuple[int, int]], Iterable[slice]]) -> 'SpanCollection':
        if isinstance(other, SpanCollection):
            # create new SpanCollection:
            result = SpanCollection(buffer_size=self.__len+other.__len + 5)

            # add spans in other:
            result.__limits[result.__len:result.__len + other.__len] = self.__limits[:other.__len]
            result.__len += other.__len

        else:
            # create new SpanCollection from Iterables:
            result = SpanCollection(other, buffer_size=self.__len + 5)

        # add spans in self:
        result.__limits[result.__len:result.__len + self.__len] = self.__limits[:self.__len]
        result.__len += self.__len

        # sort spans:
        result._sort()

        return result

    def __sub__(self, other:Union['SpanCollection', Iterable[Tuple[int, int]], Iterable[slice]]) -> 'SpanCollection':
        # normalize input:
        sorted = False
        if isinstance(other, SpanCollection):
            sorted = other.sort
            other = other.__limits[:other.__len]

        if not sorted:
            other = other[np.argsort(other, axis=0)[:,0]]

        # get first span of other:
        def get_element(index:int=0) -> Tuple[Union[int,float],Union[int,float]]:
            if index >= len(other):
                return float('inf'), float('inf')

            if isinstance(other[index], slice):
                return other[index].start, other[index].stop

            return other[index]
        other_start, other_stop = get_element()

        # iterate over spans in self:
        result = SpanCollection(buffer_size=self.__len + 10)
        result.sort = False
        j = 0
        for self_start, self_stop in self.__limits[:self.__len]:
            # find next potentially overlapping span in other:
            while self_start > other_stop:
                j += 1
                other_start, other_stop = get_element(j)

            # check if overlapping:
            if other_start >= self_stop:
                # other_stop > other_start >= self_stop > self_start
                result.append((self_start, self_stop))
                continue

            # other_stop > other_start > self_start
            if other_start > self_start:
                result.append((self_start, other_start))

            # self_stop > other_stop >= self_start
            if self_stop > other_stop:
                result.append((other_stop, self_stop))

        # sort result:
        result.sort = True

        return result
    
    def __rshift__(self, other:int) -> 'SpanCollection':
        return SpanCollection(self.__limits + other)
    
    def __lshift__(self, other:int) -> 'SpanCollection':
        return SpanCollection(self.__limits - other)
    
    def _malloc(self, n:int, buffer_size:int=5):
        # extend spans array if necessary:
        if (self.__len + n) > self.__limits.__len__():
            old_limits = self.__limits
            self.__limits = np.empty((self.__len + n + buffer_size, 2), dtype=int)
            self.__limits[:self.__len] = old_limits

    def _sort(self):
        if self.__sort_active:
            self.__limits[:self.__len] = self.__limits[np.argsort(self.__limits[:self.__len], axis=0)[:,0]]

    def _append(self, limits:np.ndarray, buffer_size:int=5) -> None:
        # extend spans array if necessary:
        n = len(limits)
        self._malloc(n, buffer_size)

        # add new spans:
        self.__limits[self.__len:self.__len + n] = limits
        self.__len += n

        # sort array:
        self._sort()

    @staticmethod
    def parse(text:str, buffer_size:int=5) -> 'SpanCollection':
        limits = []

        i = text.find('(')
        while i >= 0:
            j = text.find(')', i+1)
            if j >= 0:
                start, stop = tuple([int(v) for v in text[i+1:j].split(',')])
                assert start <= stop
                limits.append((start, stop+1))

            i = text.find('(', j+1)

        return SpanCollection(limits, buffer_size=buffer_size)
    
    @property
    def sort(self) -> bool:
        return self.__sort_active
    
    @sort.setter
    def sort(self, value:bool) -> None:
        if value and not self.__sort_active:
            self.__sort_active = True
            self._sort()

        else: self.__sort_active = False

    def append(self, span:Union[Tuple[int, int], slice], buffer_size:int=5) -> 'SpanCollection':
        if isinstance(span, slice):
            span = (int(span.start), int(span.stop))

        # extend spans array if necessary:
        self._malloc(1, buffer_size)

        # add new spans:
        self.__limits[self.__len] = span
        self.__len += 1

        # sort array:
        self._sort()

        return self
    
    def limit(self, start:int, stop:int) -> 'SpanCollection':
        return SpanCollection([(max(start, i), min(stop, j)) for i,j in self.__limits if i < stop and j > start])

####################################################################################################
# Character Replacement Lists:                                                                     #
####################################################################################################

char_stems = {
# a:
    'à': 'a',
    'á': 'a',
    'â': 'a',
    'ã': 'a',
    'ä': 'a',
    'å': 'a',
    'æ': 'a',
    'ā': 'a',
    'ă': 'a',
    'ą': 'a',

# e:
    'è': 'e',
    'é': 'e',
    'ê': 'e',
    'ë': 'e',
    'ē': 'e',
    'ĕ': 'e',
    'ė': 'e',
    'ę': 'e',
    'ě': 'e',

# i:
    'ì': 'i',
    'í': 'i',
    'î': 'i',
    'ï': 'i',
    'ĩ': 'i',
    'ī': 'i',
    'ĭ': 'i',
    'į': 'i',
    'ı': 'i',

    'ĳ': 'ij',

    'ĵ': 'j',

# o:
    'ó': 'o',
    'ô': 'o',
    'õ': 'o',
    'ö': 'o',
    'ō': 'o',
    'ŏ': 'o',
    'ő': 'o',
    'œ': 'o',

# u:    
    'ù': 'u',
    'ú': 'u',
    'û': 'u',
    'ü': 'u',
    'ũ': 'u',
    'ū': 'u',
    'ŭ': 'u',
    'ů': 'u',
    'ű': 'u',
    'ų': 'u',

# b:
    'ƀ': 'b',

# c:
    'ç': 'c',
    'ć': 'c',
    'ĉ': 'c',
    'ċ': 'c',
    'č': 'c',

# d:
    'ď': 'd',
    'đ': 'd',

# g:
    'ĝ': 'g',
    'ğ': 'g',
    'ġ': 'g',
    'ģ': 'g',

# h:
    'ĥ': 'h',
    'ħ': 'h',

# k:
    'ķ': 'k',

# l:
    'ĺ': 'l',
    'ļ': 'l',
    'ľ': 'l',
    'ŀ': 'l',
    'ł': 'l',

# n:
    'ñ': 'n',
    'ń': 'n',
    'ņ': 'n',
    'ň': 'n',
    'ŉ': 'n',
    'ŋ': 'n',

# r:
    'ŕ': 'r',
    'ŗ': 'r',
    'ř': 'r',

# s:
    'ś': 's',
    'ŝ': 's',
    'ş': 's',
    'š': 's',

# t:
    'ţ': 't',
    'ť': 't',
    'ŧ': 't',

# w:
    'ŵ': 'w',

# y:
    'ý': 'y',
    'ÿ': 'y',
    'ŷ': 'y',

# z:
    'ź': 'z',
    'ż': 'z',
    'ž': 'z',

# other:
    '–': '-',
    '—': '-',
    '_': ' ',
    '„': '"',
    '“': '"',
    '”': '"',
    '«': '"',
    '»': '"',
    '‚': "'",
    '‘': "'",
    '’': "'",
    '‹': "'",
    '›': "'",
    '\u200b': '',
    '\u200e': '',
    '®': ''
}

####################################################################################################
# Thread Functions:                                                                                #
####################################################################################################

def _calculate_tf(i, y, labels, tokens, str2col):
    # create list of all words:
    tokens = [tokens[j] for j in np.argwhere([y in label for label in labels]).flatten()]
    if len(tokens) > 0: tokens = np.concatenate(tokens)

    # count words:
    row = np.zeros(len(str2col), dtype=float)
    for token in np.unique(tokens):
        if token in str2col:
            row[str2col[token]] = (tokens == token).sum()
    return i, row

####################################################################################################
# SpanExtractor Class:                                                                             #
####################################################################################################

class SpanExtractor:
    def __init__(self, labels:Iterable[str], languages:Union[Iterable[str],None]=None, p_prefix:float=.66, p_suffix:float=.66, t_stopword:int=50) -> None:
        # create arrays of filtered out chars:
        self._separators = [
            ';', ':', ',', '.', '!', '?',
            '(', ')', '[', ']', '{', '}',
            '\t', '\n', '\r',
            '-', '+', '*', '/', '\\', '|', '&',
            '"', '\''
        ]

        # initiate metrics:
        self.resetMetrics()

        # init languages:
        if languages is None:
            languages = ['unk'] * len(labels)

        # get per-word-statistics on labels:
        words = {}
        for label, language in set(zip(labels, languages)):
            label, _ = self.split(label)
            for i, w in enumerate(label):
                w_positions = [(i+.5)/len(label)]
                w_languages = [language]
                for l in words:
                    if w in words[l]:
                        w_positions.extend(words[l][w])
                        w_languages.extend(l.split('|'))
                        words[l].pop(w)

                w_languages = list(set(w_languages))
                w_languages.sort()
                w_languages = '|'.join(w_languages)

                if w_languages not in words:
                    words[w_languages] = {}

                words[w_languages][w] = w_positions

        # calculate stopword-probability and average word position:
        max_words = max(len(words[l]) for l in words)
        words = {w:(
            min(len(words[l][w]) / (t_stopword * (0.5*(len(words[l]) / max_words)+0.5)), 1.),
            np.mean(words[l][w])
        ) for l in words for w in words[l]}

        self._stopwords = {w:words[w][0] for w in words}
        self._prefixes  = {
            w:(1-words[w][1])*self._stopwords[w]
            for w in self._stopwords
            if (1-words[w][1])*self._stopwords[w] > p_prefix
        }
        self._suffixes  = {
            w:words[w][1]*self._stopwords[w]
            for w in self._stopwords
            if words[w][1]*self._stopwords[w] > p_suffix
        }
        self._previous  = {}

    def __call__(self, texts:Iterable[str], labels:Iterable[str], manual:bool=True, verbose:bool=True, **kwargs) -> Iterable[SpanCollection]:
        output = []
        for text, label, spans in zip(texts, labels, self._extract(texts, labels, **kwargs)):
            # fall back to manual assessment if necessary:
            if len(spans) == 0 and manual:
                spans = self._extract_manually(text, label)

            # append result to output:
            output.append(spans)

            # get metrics:
            if len(spans) == 0:
                self._num_warnings += 1
                if verbose: print(f'WARNING #{self.num_warnings:d}: No spans found for "{label}"')
                continue
            self._seqs_per_text.append(len(spans))
            self._seq_accuracy.append(np.mean([text[span].find(label) >= 0 for span in spans]))
        return output

    def _extract(self, texts:str, labels:str) -> Generator[SpanCollection, None, None]:
        for text, label in tqdm(zip(texts, labels), total=min(len(texts), len(labels))):
            # tokenize text:
            label, _ = self.split(label)

            # split label in duplets and triplets:
            seqs  = []
            probs = []

            for w0 in label:
                seqs.append((w0,))
                probs.append(1-self.isStopword(w0))

            for w0, w1 in zip(label[:-1], label[1:]):
                seqs.append((w0, w1))
                probs.append(1-self.isStopword((w0, w1)))

            for w0, w1, w2 in zip(label[:-2], label[1:-1], label[2:]):
                seqs.append((w0, w1, w2))
                probs.append(1-self.isStopword((w0, w1, w2)))

            seqs  = np.array(seqs, dtype=object)
            probs = np.array(probs)
            seqs  = np.unique(seqs[probs >= min(probs.mean(), .5)])

            # find spans in text:
            yield self.find(seqs, text)
    
    def _extract_manually(self, text:str, label:str) -> SpanCollection:
        # print text:
        print('='*100)
        print(text)
        print('='*100)

        if label in self._previous:
            spans = self.find(self._previous[label], text)
            
            if len(spans) > 0:
                print('Extracted spans:')
                print('\n'.join(text[span] for span in spans))
                return spans


        # extract spans:
        spans = []
        while True:
            # prompt:
            span = input('Copy & Paste a span from the text or press enter to finish:')

            # break on empty span:
            if len(span) == 0:
                if label in self._previous: self._previous[label].extend(spans)
                else:                       self._previous[label] = spans

                # find spans in text:
                spans = self.find(spans, text)

                print('Extracted spans:')
                print('\n'.join(text[span] for span in spans))
                return spans

            # split and append span:
            spans.append(self.split(span)[0])

    def _combine_spans(self, word_idcs:Iterable[int], char_idcs:SpanCollection) -> Tuple[SpanCollection, SpanCollection]:
        word_spans = SpanCollection()
        char_spans = SpanCollection()

        # pause sorting for spans:
        word_spans.sort = False
        char_spans.sort = False

        if len(word_idcs) > 0:
            offset = word_idcs[0]
            for i, j in zip(word_idcs[:-1], word_idcs[1:]):
                if j-i > 1:
                    word_spans.append((offset, i+1))
                    char_spans.append((char_idcs[offset,0], char_idcs[i,1]))
                    offset = j
            word_spans.append((offset, word_idcs[-1]+1))
            char_spans.append((char_idcs[offset,0], char_idcs[word_idcs[-1],1]))

        # sort spans:
        word_spans.sort = True
        char_spans.sort = True

        return word_spans, char_spans

    @property
    def prefixes(self) -> Iterable[str]:
        return self._prefixes
    
    @property
    def suffixes(self) -> Iterable[str]:
        return self._suffixes
    
    @property
    def num_warnings(self) -> int:
        return self._num_warnings
    
    def isPrefix(self, text:str) -> bool:
        return text in self._prefixes
    
    def isSuffix(self, text:str) -> bool:
        return text in self._suffixes
    
    def isStopword(self, text:Union[str,Iterable[str]]) -> float:
        if isinstance(text, str):
            text = [text,]

        p = 1.
        for token in text:
            try: p *= self._stopwords[token]
            except KeyError: p = 0.

        return p
    
    def add_prefix(self, prefix):
        self._prefixes[self.split(prefix)[0][0]] = 1.
    
    def add_suffix(self, suffix):
        self._suffixes[self.split(suffix)[0][0]] = 1.

    def resetMetrics(self):
        self._num_warnings  = 0
        self._seqs_per_text = []
        self._seq_accuracy  = []

    def printMetrics(self):
        print(f'Metrics:')
        print(f'  # of texts with no match:{self._num_warnings:6d}')
        print(f'  # of matches per text:   {np.mean(self._seqs_per_text):8.1f}')
        print(f'  Accuracy of matches:     {np.mean(self._seq_accuracy)*100.:8.1f}%')

    def find(self, seqs:Iterable[Union[str,Iterable[str]]], text:str) -> SpanCollection:
        # tokenize text:
        text, spans = self.split(text)

        # extract word indices:
        words = []
        for seq in seqs:
            if isinstance(seq, str):
                seq, _ = self.split(seq)

            i = 0
            n = len(seq)
            while i+n <= len(text):
                # check for matching tuples:
                if all([l == text[i+j] for j, l in enumerate(seq)]):
                    # add indices to spans:
                    for j in range(i,i+n):
                        words.append(j)

                # check for matching tuples without separators:
                elif ''.join(seq) == text[i]:
                    words.append(i)

                else:
                    i += 1
                    continue

                # check for prefix:
                for j in range(i, 0, -1):
                    if self.isPrefix(text[j-1]):
                        words.append(j-1)
                    else: break

                i += n

                # check for suffix:
                for j in range(i, len(text)):
                    if self.isSuffix(text[j]):
                        words.append(j)
                        i += 1
                    else: break

        words = list(set(words))
        words.sort()

        # combine words:
        words, spans = self._combine_spans(words, spans)

        return spans

    def split(self, txt:str) -> Tuple[Iterable[str], SpanCollection]:
        words = []
        spans = SpanCollection()
        spans.sort = False

        def add_word(wrd, idx):
            # drop trailing char 's':
            if len(wrd) > 0:
                if wrd[-1] == 's':
                    wrd = wrd[:-1]

            if len(wrd) > 0:
                # find previous space:
                start = min(idx)
                while start > 0 and not txt[start-1].isspace():
                    start -= 1
                    break

                # find following space:
                stop = max(idx)
                while stop < (len(txt)-1) and not txt[stop+1].isspace():
                    stop += 1
                    break

                # add word:
                words.append(wrd)
                spans.append((start, stop+1))

        # mask xml:
        i = txt.find('<')
        j = txt.find('>', i+1)+1
        while i >= 0 and j > 0:
            txt = txt[:i] + ' '*(j-i) + txt[j:]

            i = txt.find('<', j)
            j = txt.find('>', i+1)+1

        # lemmatize & tokenize:
        current = None
        for i, c in enumerate(txt.lower()):
            # replace characters with ascii eqivalent:
            try: c = char_stems[c]
            except KeyError: pass

            # split at separators:
            if c.isspace() or c in self._separators:
                if current is not None:
                    add_word(*current)
                current = None

            else:
                if current is None: current = (c, [i])
                else:               current = (current[0] + c, current[1] + [i])

        # add last word:
        if current is not None:
            add_word(*current)

        # sort spans:
        spans.sort = True

        return np.array(words, dtype=object), spans
    
    def split_sentences(self, txt:str, sep:Iterable[str]=['.', '!', '?', ',', ':', ';', '\n']) -> Tuple[Iterable[str], SpanCollection, SpanCollection]:
        # split text:
        words, spans = self.split(txt)

        # unify separators: 
        for s in sep[1:]:
            txt = txt.replace(s, sep[0])
        sep = sep[0]

        # find sentences:
        sntncs = SpanCollection()
        sntncs.sort = False
        i, j = 0, txt.find(sep)
        while j >= 0:
            idcs = np.argwhere((spans[:,1] >= i) & (spans[:,1] < j)).flatten()
            if len(idcs) > 0: sntncs.append((min(idcs), max(idcs)+1))

            i = j + 1
            j = txt.find(sep, i)

        # sort spans:
        sntncs.sort = True

        return words, spans, sntncs


class SpanExtractorTfIdf(SpanExtractor):
    def _extract(self, texts:Iterable[str], labels:Iterable[str], folds:int=5, window_size:int=3, random_state:Union[int,None]=None) -> None:
        # tokenize texts:
        tokens    = []
        spans     = []
        scores    = []
        sentences = []
        for txt in tqdm(texts, desc='Tokenizing texts'):
            # split text:
            wrds, spns, sntncs = self.split_sentences(txt)

            # append to list:
            tokens.append(wrds)
            spans.append(spns)
            scores.append(np.zeros_like(wrds, dtype=float))
            sentences.append(sntncs)

        # create progress bars:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        tqdm_main = tqdm(skf.split(tokens, labels), total=folds, desc='Predicting word importance', ascii=True)
        tqdm_sec  = tqdm(total=100, unit='%', ascii=True)

        # split and filter labels:
        labels = [np.unique(self.split(label)[0]) for label in labels]
        for i, label in enumerate(labels):
            p = np.array([1.-self.isStopword(token) for token in label])
            labels[i] = label[p >= p.mean()]

        # get unique words:
        cols = np.unique([token for entry in tokens for token in entry])
        cols.sort()
        cols = {key:i for i, key in enumerate(cols)}
        n_cols = len(cols)

        # get unique labels:
#        rows = np.unique(labels)
        rows = np.unique(np.concatenate(labels))
        rows.sort()
        rows = {key:i for i, key in enumerate(rows)}
        n_rows = len(rows)

        for idx_trn, idx_prd in tqdm_main:
            # calculate term frequency per label:
            tf = np.empty([n_rows, n_cols], dtype=float)
            tqdm_sec.desc = 'Fitting tf-idf'
            queue = ArgumentQueue(_calculate_tf, list(enumerate(rows)), bar=tqdm_sec)
            for i, row in queue(
                    labels=[np.array(labels[i], dtype=object) for i in idx_trn],
                    tokens=[np.array(tokens[i], dtype=object) for i in idx_trn],
                    str2col=cols
                ): tf[i] = row

            # calculate inverse label frequency:
            idf = np.nan_to_num(-np.log((tf > 0).mean(axis=0, dtype=float)))

            # calculate tf-idf:
            tfidf = tf*idf

            # normalize per row:
            tfidf /= np.maximum(tfidf.max(axis=1), 1e-6).reshape((-1,1))

            # upvote tokens in the label:
            for label in rows:
                if label in cols:
                    tfidf[rows[label],cols[label]] = 1.

            # predict holdout scores:
            tqdm_sec.desc = 'Predicting tf-idf'
            tqdm_sec.reset()
            inc = 100./len(idx_prd)
            for i in idx_prd:
                tqdm_sec.update(inc)
                for j, word in enumerate(tokens[i]):
#                    scores[i][j] += tfidf[rows[labels[i]], cols[word]]
                    for label in labels[i]:
                        scores[i][j] += tfidf[rows[label], cols[word]]
            tqdm_sec.refresh()

        # close progress bar:
        tqdm_sec.close()

        for i in trange(len(scores), desc='Applying window'):
            # accumulate scores:
            for sentence in sentences[i]:
                orig = scores[i][sentence]
                scrs = orig.copy()

                n = min(window_size, len(scrs))
                for j in range(n):
                    scrs[:-(j+1)] += orig[(j+1):]
                    scrs[(j+1):] += orig[:-(j+1)]
                scores[i][sentence] = scrs / (2.*n)

            # combine spans:
            idx, spans[i] = self._combine_spans(
                np.argwhere(scores[i] > scores[i].mean() + scores[i].std()).flatten(), 
                spans[i]
            )
            scores[i] = np.array([scores[i][span].mean() for span in idx])

            # filter for better-than-average spans:
            spans[i] = spans[i][scores[i] >= scores[i].mean()]

        return spans