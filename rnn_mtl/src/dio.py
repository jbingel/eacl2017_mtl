from collections import defaultdict
import numpy as np
import sys
import random
from collections import Counter

PAD_ID = 0
GO_ID = 1
OOV_ID = 2
EOS_ID = 3

PAD_SYM = "<PAD>"
GO_SYM = "<GO>"
OOV_SYM = "<OOV>"
EOS_SYM = "<EOS>"

SPECIAL_SYMBOLS = {
    PAD_SYM: PAD_ID,
    GO_SYM: GO_ID,
    OOV_SYM: OOV_ID,
    EOS_SYM: EOS_ID
}

INDICES = "indices"
ONE_HOT = "one_hot"
VECTORS = "vectors"
OUTPUT_TYPES = [INDICES, ONE_HOT, VECTORS]

SEQ = 'seq'
LBL = 'lbl'
TASKTYPES = [SEQ, LBL]


def make_default_index_mapper(special_symbols=SPECIAL_SYMBOLS):
    mapper = {}
    if special_symbols:
        assert type(special_symbols) == dict, \
            "Need to provide dict as special symbols mapping."
    for _symbol, _id in special_symbols.items():
        mapper[_symbol] = _id
    return mapper


def make_default_vector_mapper(word_vectors, special_symbols=SPECIAL_SYMBOLS):
    mapper = {}
    oov = np.average(word_vectors, axis=0)
    if special_symbols:
        assert type(special_symbols) == dict, \
            "Need to provide dict as special symbols mapping."
    for _symbol in special_symbols.keys():
        mapper[_symbol] = oov
    return mapper


def pad(sequence, pad_len, pad_value, cut=True):
    seq_len = len(sequence)
    if seq_len < pad_len:
        sequence = sequence + ([pad_value] * (pad_len - seq_len))
    if cut:
        if seq_len > pad_len:
            return sequence[:pad_len]
    return sequence


def one_hot(seq, voc_size):
    seq_len = len(seq)
    out = np.zeros((seq_len, voc_size))
    for i in range(seq_len):
        idx = int(seq[i])
        out[i][idx] = 1
    return out


def one_hot_batch(batch, voc_size):
    seq_len = len(batch[0])
    batch_len = len(batch)
    out = np.zeros((batch_len, seq_len, voc_size))
    for b in range(batch_len):
        for i in range(seq_len):
            idx = int(batch[b][i])
            out[b][i][idx] = 1
    return out


def invert(mapper):
    """
    Inverts a dictionary, mapping values to lists of keys
    :param mapper: A dictionary
    :return: the inverted mapper
    """
    inverted_map = defaultdict(list)
    for key, val in mapper.items():
        inverted_map[val].append(key)
    return inverted_map


class SequenceVectorizer:

    def __init__(self, output_type):
        """
        :param output_type: must be one of 'indices', 'one_hot', 'vectors'
        """
        assert output_type in OUTPUT_TYPES, \
            "Error initializing SequenceVectorizer. Argument output_type" \
            "must be one of 'indices', 'one_hot', 'vectors'."
        self.output_type = output_type
        self.mapper = None
        self.reverse_mapper = None

    def set_mapper(self, mapper):
        """
        Sets a pre-defined mapper, e.g. based on embeddings or clusters
        :param mapper: A dictionary object, mapping labels to ints or arrays
        """
        # TODO assert that mapper matches with self.output_type
        self.mapper = mapper
        for sym in SPECIAL_SYMBOLS:
            assert sym in self.mapper.keys(), \
                "Did not find required symbol {} in mapper.".format(sym)

    def fit(self, input_data, warm_start=True):
        """
        Fits the mapper based on input data. Will assign new ID for each
        symbol found in input_data. Includes all special symbols.
        :param input_data: a list of list. Contains example sequences (e.g.
        sentences), which in turn contain items (e.g. words/labels).
        :param warm_start: if True, keep building up existing mapper for new
        data. If False, mapper is newly initialized.
        :return: the fitted mapper (dictionary)
        """
        assert self.output_type != VECTORS, \
            "Cannot fit mapper in a 'vectors' SequenceVectorizer"
        mapper = self.mapper
        if not warm_start:
            mapper = {}
        if not self.mapper:
            mapper = make_default_index_mapper(SPECIAL_SYMBOLS)
        for sequence in input_data:
            for item in sequence:
                if item not in mapper:
                    # increments to next free ID
                    mapper[item] = len(mapper)
        self.mapper = mapper
        return mapper

    def transform(self, input_data, pad_len=0, cut=True):
        """
        Transforms data to numpy array. Prepends a special GO symbol and appends
        a special EOS symbol after the sequence. Also replaces unknown items
        with a OOV symbol.
        :param input_data: a list of list. Contains example sequences (e.g.
        sentences), which in turn contain items (e.g. words/labels).
        :param pad_len: If > 0, pad all sequences to pad_len using a PAD value
        :param cut: If pad_len > 0 and cut == True, cut off all sequences
        to not exceed pad_len
        :return: vectorized input sequences
        """
        output = []
        for sequence in input_data:
            # prepend GO value
            ex_out = [self.mapper[GO_SYM]]
            for item in sequence:
                # append mapped item, OOV value if not found in voc
                ex_out.append(self.mapper.get(item, self.mapper[OOV_SYM]))
            # append EOS value
            ex_out.append(self.mapper[EOS_SYM])
            if type(pad_len) == int and pad_len > 0:
                ex_out = pad(ex_out, pad_len, self.mapper[PAD_SYM], cut=cut)
            output.append(ex_out)
        if self.output_type == ONE_HOT:
            return one_hot_batch(output, len(set(self.mapper.values())))
        return np.array(output)

    def untransform(self, input_data):
        """
        Reverts numerical representation of data to original symbols. NB: uses
        inverted mapper, which maps to lists of original keys!
        :param input_data: a list of list. Contains numerical representations
        of example sequences (e.g. sentences), which in turn contain items
        (e.g. words/labels).
        :return: the reverted data
        """
        assert self.output_type != VECTORS, \
            "Cannot revert vectors (I'm a 'vectors' SequenceVectorizer)"
        if not self.reverse_mapper:
            self.reverse_mapper = invert(self.mapper)

        reverted_data = []
        for sequence in input_data:
            reverted_sequence = []
            for item in sequence:
                reverted_sequence.append(self.reverse_mapper[item])
            reverted_data.append(reverted_sequence)
        return reverted_data


def load_clusters(mapfile):
    print("Loading clusters... ", end="")
    sys.stdout.flush()
    if not mapfile:
        print()
        sys.stderr.write("No clusters specified. Please add line "
                         "'clusters[path]' to data config file!\n")
        sys.exit(1)
    mapper = make_default_index_mapper(SPECIAL_SYMBOLS)
    blocked = len(mapper)
    for line in open(mapfile):
        try:
            clid, w = line.split()
        except:
            raise ValueError("Clusters file malformed. Expected format "
                             "<num> <word>. \nLine: {}".format(line))
        mapper[w] = int(clid) + blocked
    print("Done!")
    sys.stdout.flush()
    return mapper


def load_embeddings(embfile):
    print("Loading embeddings... ", end="")
    sys.stdout.flush()
    if not embfile:
        print()
        sys.stderr.write("No clusters specified. Please add line "
                         "'clusters[path]' to data config file!\n")
        sys.exit(1)
    f = (line.split(" ", 1)[1] for line in open(embfile))
    words = [line.split(" ", 1)[0].split("_")[0] for line in open(embfile)]
    w2id = {word: word_id for word_id, word in enumerate(words)}
    emb_matrix = np.loadtxt(f)
    sys.stdout.flush()
    for _sym in SPECIAL_SYMBOLS.keys():
        assert _sym in w2id.keys(), \
            "Required special symbol {} not found in provided vectors file {}"\
            .format(_sym, embfile)
    print("Done!")
    return w2id, emb_matrix


def read_two_cols_data(fname):
    inputs, outputs = [], []
    for line in open(fname):
        line = line.strip().lower()
        if not line:
            if inputs and outputs:
                yield inputs, outputs
            inputs, outputs = [], []
        else:
            try:
                w, lbl = line.rsplit(maxsplit=1)
                w = "_".join(w.split())
            except:
                print(line)
                raise
            inputs.append(w)
            outputs.append(lbl)
    if inputs and outputs:
        yield inputs, outputs


def read_parallel_data(fname):
    from nltk.tokenize import word_tokenize
    for line in open(fname):
        line = line.strip().lower().split(" ||| ")
        if not line:
            continue
        try:
            en, fr = line
            sent = ([w for w in word_tokenize(en)],
                    [w for w in word_tokenize(fr)])
        except ValueError:
            continue
        if sent:
            yield sent


def compute_class_weights(vectorizer, data):
    n_classes = len(set(vectorizer.mapper.values()))
    labels = [vectorizer.mapper.get(item) for sentence in data
              for item in sentence]
    counter = Counter(labels)
    total_count = sum(counter.values())
    labels_count = np.array([counter[i] for i in range(n_classes)])
    labels_count[:4] = np.zeros(4)  # SPECIAL SYMBOLS
    class_weights = 1 - (labels_count / total_count)
    return class_weights


class Data:
    def __init__(self):
        self.tasks = {}
        self.input_vectorizer = None
        self.output_vectorizers = {}
        self.corpora = defaultdict(dict)

    def add_task(self, task_cfg):
        task_cfg.vectorizer = self.output_vectorizers[task_cfg.task_id]
        self.tasks[task_cfg.task_id] = task_cfg

    def set_input_vectorizer(self, vectorizer):
        self.input_vectorizer = vectorizer

    def add_output_vectorizer(self, task, vectorizer):
        self.output_vectorizers[task] = vectorizer

    def add_corpus(self, task, role, inputs, outputs):
        if role not in self.corpora[task].keys():
            self.corpora[task][role] = {}
        self.corpora[task][role]['in'] = inputs
        self.corpora[task][role]['out'] = outputs

    def get_input_length(self):
        if self.input_vectorizer.output_type == VECTORS:
            # just transform any word to vector to find out vector length
            length = self.input_vectorizer.transform([['<GO>']]).shape[2]
        else:
            length = len(set(self.input_vectorizer.mapper.values()))
        print("\nInput vectors are of length {}".format(length))
        return length

    def get_batch(self, task, role, seq_ids, vectorize=True, pad_len=0):
        input_batch, output_batch = [], []
        seq_lens = []
        corpus = self.corpora[task][role]
        for i in seq_ids:
            input_batch.append(corpus['in'][i])
            output_batch.append(corpus['out'][i])
            seq_lens.append(len(corpus['in'][i])+2)  # +2 for GO and EOS
        if vectorize:
            input_batch = self.input_vectorizer.transform(
                input_batch, pad_len=pad_len)
            output_batch = self.output_vectorizers[task].transform(
                output_batch, pad_len=pad_len)
            if pad_len > 0:
                seq_lens = [min(sl, pad_len) for sl in seq_lens]
        return input_batch, output_batch, seq_lens

    def get_random_batch(self, task, role, batch_size=64,
                         vectorize=True, pad_len=0):
        n_data = len(self.corpora[task][role]['in'])
        sequence_ids = random.sample(range(n_data), batch_size)
        return self.get_batch(task, role, sequence_ids,
                              vectorize=vectorize,
                              pad_len=pad_len)


def read_data(cfgfile, tasks, input_type, input_mapping=None,
              output_mappings=None):
    print("Reading data... ", end="")
    sys.stdout.flush()
    # tasktype2vectorizer_type = {
    #     LBL: ONE_HOT,
    #     SEQ: INDICES
    # }
    data = Data()
    # Init and register input vectorizer
    in_vectorizer = SequenceVectorizer(input_type)
    if input_mapping:
        in_vectorizer.set_mapper(input_mapping)
    # Read data config file line by line and process
    for line in open(cfgfile):
        line = line.strip()
        if (not line) or line[0] == '#':
            continue
        role, task_id, source, tasktype = line.split()
        if task_id not in tasks:
            continue
        if tasktype == LBL:
            # read_*_data are generators, therefore the list()
            inputs, outputs = zip(*list(read_two_cols_data(source)))
        elif tasktype == SEQ:
            inputs, outputs = zip(*list(read_parallel_data(source)))
        else:
            print("\nSorry, task type {} cannot be handled.".format(tasktype))
            raise ValueError
        data.add_corpus(task_id, role, inputs, outputs)
        # Fit vectorizers
        if role == 'train':
            if not input_mapping:
                print("\nFitting input vectorizer with '{}' "
                      "training data...".format(task_id))
                in_vectorizer.fit(inputs, warm_start=True)
            # Set task-dependent output vectorizers
            # vectorizer_type = tasktype2vectorizer_type[tasktype]
            # if tasktype == SEQ and task_id not in output_mappings.keys():
            #     vectorizer_type = ONE_HOT
            # vectorizer_type = ONE_HOT
            vectorizer_type = INDICES
            vectorizer = SequenceVectorizer(vectorizer_type)
            # If mapping specified for this task, set this as the vectorizer map
            if type(output_mappings) == dict and task_id in output_mappings:
                vectorizer.set_mapper(output_mappings[task_id])
            # If not, fit vectorizer on based on all labels in training data
            else:
                vectorizer.fit(outputs)
            # Register vectorizer for task
            class_weights = compute_class_weights(vectorizer, outputs)
            tasks[task_id].class_weights = class_weights
            data.add_output_vectorizer(task_id, vectorizer)
            # Register task only now, needs to have vectorizer set for task
            data.add_task(tasks[task_id])
    data.set_input_vectorizer(in_vectorizer)
    print("Done!")
    sys.stdout.flush()
    return data
