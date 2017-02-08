# import kenlm  TODO uncomment when needing to use seq2seq decoder scoring
import os
import sys
import random
from collections import defaultdict

import dio
import mtl_model
import numpy as np
import tensorflow as tf
from docopt import docopt
from nltk import word_tokenize as tokenize
import sklearn as sk
import util
import nltk
from util import DEFAULT_ARGS


lm = None
CONTENT_POS = ["NOUN", "VERB", "ADV", "ADJ", "NUM", "X"]


def create_model(session, tasks_cfg, data, embed_matrix, args):
    """Create translation model and initialize or load parameters in session."""
    trdir = args['train_dir']
    model = mtl_model.SeqMtlModel(tasks_cfg, data.get_input_length(),
                                  args['n_layers'], args['n_hidden'],
                                  args['optmzr'], args['actfunc'],
                                  args['timesteps'], args['dropout_rate'],
                                  args['learning_rate'], args['batch_size'],
                                  args['embed_size'],
                                  pretrained_embeddings=embed_matrix)
    ckpt = tf.train.get_checkpoint_state(trdir)
    # Restore existing model
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        # hack because couldn't get this var to be restored when loading model
        model.global_step = int(ckpt.model_checkpoint_path.split("-")[-1])
        saved_dict = {}
        for x in tf.all_variables():
            saved_dict[x.name] = x
    # Create new model
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
        model.global_step = 0
    return model


def train(session, model, task_cfgs, data, args):
    # # # PREPARING TRAINING METRICS AND LOGGERS # # #
    task_ids = list(task_cfgs.keys())
    trdir = args['train_dir']
    iters = int(args['iters'])
    timesteps = int(args['timesteps'])
    batch_size = int(args['batch_size'])
    losslog_train = open(trdir+"/loss_tr.log", "a", 1)
    losslog_dev = open(trdir + "/loss_dv.log", "a", 1)
    checkpoint_path = os.path.join(trdir, "mtl.ckpt")
    start_step = model.global_step
    blockloss = defaultdict(list)
    _tasks = data.tasks.values()
    with tf.variable_scope("", reuse=True):
        before = tf.get_variable("output_{}/weights".format(task_ids[0])).eval()
    print("START STEP: {}".format(start_step))

    # # #  TRAINING # # #
    # TODO more principled training schedule, e.g. get full range of train sets
    # for all tasks, then permutate IDs into batches and print out how many
    # batch iters correspond to full train sets iteration. e.g. batch_size 10,
    # task1 has 100 ex, task2 has 50, schedule could be
    # [(t1, 31-40), (t2, 11-20, ...)]
    for t in range(1, iters+1):
        model.global_step += 1
        task_id = random.choice(task_ids)
        task = task_cfgs[task_id]
        print("\nIter {}/{}, task: {}"
              .format(t + start_step, iters + start_step, task_id))

        # # # TRAIN SET PERFORMANCE MEASURING # # #
        if t % 10 == 0 or t == iters:
            for _task in _tasks:
                _task_id = _task.task_id
                print("  Iter {}: Avg. batch training loss (task: {}): {}"
                      .format(t, _task_id, np.average(blockloss[_task])))
                losslog_train.write("Iter {}, task {}, loss {}\n".
                                    format(t + start_step, _task_id,
                                           np.average(blockloss[_task])))
            blockloss = defaultdict(list)
            with tf.variable_scope("", reuse=True):
                after = tf.get_variable("output_{}/weights".
                                        format(task_ids[0])).eval()
                diff = abs(before - after)
                print("  Updated params by {} on avg. for task {} (shape {})."
                      .format(np.mean(diff), task_ids[0], diff.shape))
                before = after.copy()
            sys.stdout.flush()  # else stdout redirect is buffered for very long

        # # # ACTUAL TRAINING (MODEL STEP) # # #
        # get input data and labels
        xs, ys, input_lengths = data.get_random_batch(
            task.task_id, "train", batch_size,
            vectorize=True, pad_len=timesteps)
        # swapaxes since input placeholder is timesteps * batch_size * input_len
        pred, loss = model.step(
            session, task, xs.swapaxes(0, 1), input_lengths, ys, mode="train")
        blockloss[task].append(np.average(loss))

        # # # DEV SET PERFORMANCE MEASURING # # #
        if t % 100 == 0 or t == iters:
            for eval_task in data.tasks.values():
                if not data.corpora[eval_task.task_id]['dev']:
                    continue
                devloss, acc, f1, p, r = evaluate(session, model, data,
                                                  eval_task, "dev", args,
                                                  pct_of_dataset=100)
                print("  Avg. dev loss on task {}: {}, (acc: {})"
                      .format(eval_task.task_id, devloss, acc))
                losslog_dev.write("Iter {}, task {}, loss {}, acc {:1.5f}, "
                                  "f1 {:1.5f}, p {:1.5f}, r {:1.5f}\n"
                                  .format(t + start_step, eval_task.task_id,
                                          devloss, acc, f1, p, r))

        # # # SAVE MODEL AT CERTAIN INTERVALS # # #
        if t % int(args['save_interval']) == 0 or t == iters:
            model.saver.save(session, checkpoint_path,
                             global_step=model.global_step)
    losslog_train.close()
    losslog_dev.close()


def evaluate(session, model, data, task, split, args, sample_output=True,
             pct_of_dataset=100):
    """

    :param session:
    :param model:
    :param data:
    :param task:
    :param split:
    :param args:
    :param sample_output:
    :param pct_of_dataset: how many percent of the dataset to use
    :return:
    """
    task_id = task.task_id
    timesteps = int(args['timesteps'])
    batch_size = int(args['batch_size'])
    pct_of_dataset = min(max(pct_of_dataset, 0), 100)  # limit to [0, 100]
    n_items = int(len(data.corpora[task_id][split]['in']) *
                  (pct_of_dataset / 100))
    batches = max(n_items // batch_size, 1)  # at least one batch
    loss = 0
    # Compute micro-averaged F1/prec/recall, disregarding negative class ('O')
    # and special symbols
    negative_class = task.vectorizer.mapper.get('o', None)
    positive_classes = [cls for cls in set(task.vectorizer.mapper.values()) if
                        cls > 3 and cls != negative_class]
    # Use flat lists to compute acc/f1/r/p
    preds_flat, targets_flat = [], []
    # iterate over all possible batches in dev data
    for batch_id in range(batches):
        start, end = batch_id * batch_size, (batch_id+1) * batch_size
        # src_batch shape: batch_size * timesteps * input_length
        # tgt_batch shape: batch_size * timesteps * n_classes
        src_batch, tgt_batch, input_lengths = \
            data.get_batch(task_id, split, list(range(start, end)),
                           vectorize=True, pad_len=timesteps)
        # batch_pred shape: batch_size * timesteps * n_classes
        # swapaxes b/c model placeholder is timesteps * batch_size * input_len
        batch_pred, batch_loss = model.step(session, task,
                                            src_batch.swapaxes(0, 1),
                                            input_lengths,
                                            tgt_batch, mode="eval")
        # reshape for easier iteration over timesteps
        batch_pred = batch_pred.reshape([batch_size, timesteps, -1])
        loss += batch_loss
        for b in range(batch_size):
            # get gold/pred labels from argmaxing class vectors
            targets = tgt_batch[b]
            preds = [np.argmax(ts) for ts in batch_pred[b]]
            #  masking evaluation with input lengthts (don't count GO and EOS)
            targets = targets[1:input_lengths[b]-1]
            preds = preds[1:input_lengths[b]-1]
            targets_flat += list(targets)
            preds_flat += list(preds)

        if batch_id == 0 and sample_output:
            # Write out sample output for model inspection
            _xs = np.reshape(src_batch, [batch_size, timesteps, -1])
            _ys = np.reshape(tgt_batch, [batch_size, timesteps, -1])
            _ps = batch_pred
            util.write_sample_output(args['train_dir']+"/example_output.log",
                                     _xs, _ys, _ps, task)

    acc = sk.metrics.accuracy_score(targets_flat, preds_flat)
    f1 = sk.metrics.f1_score(targets_flat, preds_flat,
                             labels=positive_classes,
			     pos_label=positive_classes[0],
                             average='micro')
    prec = sk.metrics.precision_score(targets_flat, preds_flat,
                                      labels=positive_classes,
			              pos_label=positive_classes[0],
                                      average='micro')
    rec = sk.metrics.recall_score(targets_flat, preds_flat,
                                  labels=positive_classes,
			     	  pos_label=positive_classes[0],
                                  average='micro')

    return loss/batches, acc, f1, prec, rec


def decode(session, model, inputseq, vectorizer, task, args, interactive=False):
    """
    Greedy decoder, just argmaxes logits
    :param session:
    :param model:
    :param inputseq:
    :param vectorizer:
    :param task:
    :param args:
    :param interactive:
    :return:
    """
    timesteps = int(args['timesteps'])
    batch_size = int(args['batch_size'])

    def get_prediction(inp):
        tokens = tokenize(inp)
        if len(tokens) > timesteps:
            import sys
            sys.stderr.write("Exceeding allowed input length. "
                             "Cutting off after {} tokens.".format(timesteps))
            tokens = tokens[:timesteps]  # cut off after max model timesteps
        vecs = vectorizer.transform([tokens], pad_len=timesteps)  # list of seqs
        vecs = vecs.repeat(batch_size, axis=0)
        input_lengths = np.array([len(tokens)] * batch_size)
        pred, _ = model.step(session, task, vecs.swapaxes(0, 1),
                             input_lengths, mode="decode")
        pred = pred.reshape([batch_size, timesteps, -1])
        s = []
        out_vectorizer = task.vectorizer
        for t in range(timesteps):
            max_activation = pred[0][t].argmax()
            # first wrap in two lists (expected by vectorizer), then unpack
            s.append(out_vectorizer.untransform([[max_activation]])[0][0])
        return s

    if not interactive:
        return get_prediction(inputseq)
    else:
        while True:
            inputseq = input("Input: ")
            print(get_prediction(inputseq))


def beam_decode(session, model, inputseq, vectorizer, task, args,
                interactive=False):
    """
    Decode with beam, using top k logits
    :param session:
    :param model:
    :param inputseq:
    :param vectorizer:
    :param task:
    :param interactive:
    :param args:
    :return:
    """
    timesteps = int(args['timesteps'])
    batch_size = int(args['batch_size'])
    k_clusters = int(args['k_clusters'])
    beam_width = int(args['beam_width'])
    k_best = int(args['k_best'])
    prune = bool(args['prune'])

    def get_prediction(inp):
        tokens = tokenize(inp)
        if len(tokens) > timesteps:
            import sys
            sys.stderr.write("Exceeding allowed input length. "
                             "Cutting off after {} tokens.".format(timesteps))
            tokens = tokens[:timesteps]  # cut off after max model timesteps
        vecs = vectorizer.transform([tokens])  # list of seqs
        vecs = vecs.repeat(batch_size, axis=0)
        input_lengths = np.array([len(tokens)] * batch_size)
        pred, _ = model.step(session, task, vecs, input_lengths, mode="decode")
        pred = pred.reshape([batch_size, timesteps, -1])
        # Find all candidate words per timestep (all words from top k clusters)
        candidates = {}
        for t in range(timesteps):
            if np.argmax(pred[0][t]) == dio.PAD_ID:
                break
            candidates[t] = set()
            # get the top k clusters
            # (http://stackoverflow.com/questions/6910641/)
            topclusters = np.argpartition(pred[0][t], -k_clusters)[-k_clusters:]
            for c in topclusters:
                candidates[t].update(task.i2l[c])  # expand with this cluster
        # Find the optimal sequence from the candidate words using beam search
        global lm
        if not lm:
            lm = kenlm.Model('data/lms/en-70k-0.2-pruned.lm')
        hypos = ["<s>"]
        for t in range(len(candidates)):
            t_hypos = []
            content_words = [w for w, t in nltk.pos_tag(tokens)
                             if t in CONTENT_POS]
            if prune:
                copies = candidates[t].intersection(set(content_words))
                if copies:
                    candidates[t] = copies
            for cand in candidates[t]:
                for h in hypos:
                    cand_t = h+" "+cand
                    score = lm.score(cand_t)  # get language model score
                    t_hypos.append((cand_t, score))
            # get beam_width highest scoring hypotheses
            hypos = [h for h, s in
                     sorted(t_hypos, key=lambda x: x[1])[-beam_width:]]
        return hypos[:-k_best:-1]  # k highest scoring hypos, revert list

    if not interactive:
        return get_prediction(inputseq)
    else:
        while True:
            output = get_prediction(input("Input: "))
            print("\n".join(output))


def main(args=DEFAULT_ARGS):
    trdir = args['train_dir']
    task_cfgs = util.read_tasks_config(args['task_cfg'])
    input_mapping = None
    output_mapping = {}
    embed_matrix = None
    input_type = dio.ONE_HOT  # tells vectorizer to transform IDs to one-hots
    if args['input_map']:
        input_mapping = dio.load_clusters(args['input_map'])
    elif args['embeddings']:
        input_mapping, embed_matrix = dio.load_embeddings(args['embeddings'])
        # provide indices to mtl_model, embed_matrix will init embedding layer
        input_type = dio.INDICES
    if args['embed_size'] > 0:
        input_type = dio.INDICES  # Only put in indices to network, no one-hots
    if args['output_map']:
        output_mapper = dio.load_clusters(args['output_map'])
        # {'sim': output_mapper, 'en_fr': output_mapper, ...}
        output_mapping = {task_id: output_mapper
                          for task_id, task in task_cfgs.items()
                          if task.task_type == dio.SEQ}
    data = dio.read_data(args['data'],
                         tasks=task_cfgs,
                         input_type=input_type,
                         input_mapping=input_mapping,
                         output_mappings=output_mapping)
    os.makedirs(trdir, exist_ok=True)
    for task_id in task_cfgs:
        assert task_id in data.tasks, "No data found for task {}. Check data" \
                                      "config file {}. Exiting".format(
                                       task_id, args['data'])

    with tf.Session() as session:
        model = create_model(session, task_cfgs, data, embed_matrix, args)
        if args["decode"]:
            decode_task = task_cfgs[args["decode_task"]]
            if decode_task.task_type == "seq":
                beam_decode(session, model, None, data.input_vectorizer,
                            decode_task, args, interactive=True)
            else:
                decode(session, model, None, data.input_vectorizer,
                       decode_task, args, interactive=True)
        else:
            # if model.global_step == 0:
            #     util.make_logfiles(trdir, args)
            train(session, model, task_cfgs, data, args)


if __name__ == "__main__":

    parsed_args = docopt("""
        Usage:
            experiment.py train --task_cfg=<file> --data=<file> --train_dir=<ofile> [options ]
            experiment.py decode --task_cfg=<file> --data=<file> --train_dir=<ofile> --decode_task=<> [options ]

        Options:
            --iters k    [default: 1000]
            --n_layers k  [default: 1]
            --n_hidden k  [default: 100]
            --timesteps k  [default: 30]
            --batch_size k  [default: 32]
            --embed_size k  [default: 0] 0 means no embedding layer, model expects vectors
            --embeddings FILE   pretrained embeddings
            --input_map FILE   input mappings
            --output_map FILE   output mappings
            --rnn type   [default: lstm]  other options: simple
            --actfunc type  [default: tanh] activation between RNN layers -- sigmoid, relu, tanh, none
            --optmzr type  [default: adadelta] optimizer for tasks -- rmsprop, adagrad, adadelta, momentum
            --save_interval k   [default: 1000] save model every k iterations
            --dropout_rate k  [default: 0.0] apply word-dropout during trainign with rate k (drop if random < k)
            --learning_rate k   [default: 0.01] learning rate
            --k_clusters k  [default: 5] top-k clusters to use when beam decoding
            --beam_width k  [default: 20] always keep k candidates for each timestep
            --k_best k  [default: 1] return top k hypotheses when decoding
            """)

    _args = DEFAULT_ARGS
    for key, val in parsed_args.items():
        clean_key = key if key in ['train', 'decode'] else key[2:]  # leading --
        if val is not None:
            if clean_key in _args.keys():
                if type(_args[clean_key]) == int:
                    val = int(val)
                elif type(_args[clean_key]) == float:
                    val = float(val)
                elif type(_args[clean_key]) == bool:
                    val = bool(val)
            _args[clean_key] = val
    main(_args)
