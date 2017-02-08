import sys
import os
import numpy as np
import shutil


class Task:
    def __init__(self, task_id, task_type, layer):
        self.layer = layer
        self.task_id = task_id
        self.task_type = task_type
        # self.i2l = None
        self.vectorizer = None
        self.classes = []
        self.class_weights = []

    def get_num_labels(self):
        return len(set(self.vectorizer.mapper.values()))

    def get_class_weights(self):
        return self.class_weights


def read_tasks_config(fname):
    tasks = {}
    with open(fname) as f:
        for line in f:
            if (not line) or line[0] == '#':
                continue
            spl = line.strip().split()
            try:
                layer, task, tasktype = spl
                layer = int(layer)
                if layer < 1: raise ValueError
                tasks[task] = Task(task, tasktype, layer)
            except ValueError:
                sys.stderr.write("Warning! Error reading line {}. Not formatted"
                                 " correctly? Format should be <layer> <width> "
                                 "<task>, one per line.".format(line))
    return tasks


def flatten_tensor(tensor):
    out = np.zeros([len(tensor), len(tensor[0]), 1])
    for b in range(len(tensor)):  # batches
        for i in range(len(tensor[b])):  # timesteps
            # for j in range(len(tensor[b][i])):
            if tensor[b][i] == 1:
                out[b] = i
    return out


def write_sample_output(path, xs, ys, ps, task):
    f = open(path, "w")
    f.write("Task: " + task.task_id + "\n\n")
    i = 0
    for x, y, p in zip(xs, ys, ps):
        i += 1
        f.write("\n\n+++ SENTENCE {} +++ \n".format(i))
        for ts in range(len(x)):
            # lbl = np.where(y[ts] == 1)
            lbl = y[ts]
            gold_score = p[ts][lbl]
            f.write("  Gold label: {} ['{}'], got assigned {} (rank {}).\n"
                    .format(np.array_str(lbl[0]),
                            task.vectorizer.untransform([[int(lbl[0])]])[0][0],
                            gold_score, (p[ts] > gold_score).sum()))
            f.write("  Mean {}, max {} ({}), mean diff {}.\n"
                    .format(np.array_str(p[ts].mean()),
                            np.array_str(p[ts].max()),
                            np.argmax(p[ts]),
                            np.mean(np.abs(np.subtract(y[ts], p[ts])))))
            f.write("\n")

    f.close()


def make_logfiles(trdir, args):
    cfgfile = open(trdir+"/config", "w")
    for k, v in args.items():
        cfgfile.write("{}\t{}\n".format(k, v))
    cfgfile.close()

    losslog_tr = open(trdir+"/loss_tr.log", "w")
    losslog_tr.close()

    losslog_dv = open(trdir+"/loss_dv.log", "w")
    losslog_dv.close()

    try:
        shutil.copy(args["--tasks"], trdir+"/tasks.cfg")
        shutil.copy(args["--data"], trdir+"/data.cfg")
    except shutil.SameFileError:  # happens when using resume script
        pass

    make_resumescript(args)
    make_decodescript(args)


def make_decodescript(args):
    cwd = os.getcwd()
    srcpath = os.path.dirname(os.path.realpath(__file__))
    scriptdir = cwd+os.path.sep+args["--train_dir"]
    script = open(args["--train_dir"] + "/decode.sh", "w")
    script.write("#!/bin/bash\n")
    script.write("\n")
    # script.write('scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" '
    #              '&& pwd )"\n')
    script.write("""
if [ "$#" -ne 1 ]; then \n
    echo "Missing argument: task id to decode from. Using default task (sim)\n"
    decode="sim"
else
    decode=$1
fi\n
    """)
    script.write("\n")
    script.write("~/anaconda3/bin/python {}/experiment.py DECODE ".format(
        srcpath))
    script.write("--data {}/data.cfg ".format(scriptdir))
    script.write("--tasks {}/tasks.cfg ".format(scriptdir))
    script.write("--train_dir {} ".format(scriptdir))
    script.write("--clusters_in {} ".format(
        cwd+os.path.sep+args["--clusters_in"]))
    script.write("--clusters_out {} ".format(
        cwd + os.path.sep + args["--clusters_out"]))
    script.write("--timesteps {} ".format(args["--timesteps"]))
    script.write("--optimizer {} ".format(args["--optimizer"]))
    script.write("--n_hidden {} ".format(args["--n_hidden"]))
    script.write("--n_layers {} ".format(args["--n_layers"]))
    script.write("--batch_size {} ".format(args["--batch_size"]))
    script.write("--actfunc {} ".format(args["--actfunc"]))
    script.write("--dropout {} ".format(args["--dropout"]))
    script.write("--k_clusters {} ".format(args["--k_clusters"]))
    script.write("--beam_width {} ".format(args["--beam_width"]))
    script.write("--decode $decode \n")
    script.close()
    

def make_resumescript(args):
    cwd = os.getcwd()
    srcpath = os.path.dirname(os.path.realpath(__file__))
    scriptdir = cwd + os.path.sep + args["--train_dir"]
    script = open(args["--train_dir"]+"/resume_training.sh", "w")
    script.write("#!/bin/bash\n")
    script.write("\n")
    # script.write('scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" '
    #              '&& pwd )"\n')
    script.write("""
if [ "$#" -ne 2 ]; then
    echo "No number of iters and learning rate provided. Using default values (iters=1000, lr=0.1)"
    iters=1000
    learning_rate=0.1
else
    iters=$1
    learning_rate=$2
fi\n
    """)
    script.write("\n")
    script.write("python {}/experiment.py ".format(srcpath))
    script.write("--data {}/data.cfg ".format(scriptdir))
    script.write("--tasks {}/tasks.cfg ".format(scriptdir))
    script.write("--train_dir {} ".format(scriptdir))
    script.write("--clusters_in {} ".format(
        cwd+os.path.sep+args["--clusters_in"]))
    script.write("--clusters_out {} ".format(
        cwd + os.path.sep + args["--clusters_out"]))
    script.write("--timesteps {} ".format(args["--timesteps"]))
    script.write("--optimizer {} ".format(args["--optimizer"]))
    script.write("--n_hidden {} ".format(args["--n_hidden"]))
    script.write("--n_layers {} ".format(args["--n_layers"]))
    script.write("--batch_size {} ".format(args["--batch_size"]))
    script.write("--actfunc {} ".format(args["--actfunc"]))
    script.write("--dropout {} ".format(args["--dropout"]))
    script.write("--save_interval {} ".format(args["--save_interval"]))
    script.write("--learning_rate $learning_rate ")
    script.write("--iters $iters\n")
    script.close()

DEFAULT_ARGS = {
    # Experiment related, must be filled
    'train': True,
    'decode': False,
    'train_dir': None,
    'data': None,
    'task_cfg': None,
    # Experiment related, optional
    'input_map': None,
    'output_map': None,
    'embeddings': None,

    # Model related
    'timesteps': 30,
    'batch_size': 50,
    'save_interval': 100,
    'n_layers': 1,
    'n_hidden': 50,
    'dropout_rate': 0.1,
    'embed_size': 0,  # 0 means no embedding layer, model expects vectors
    'optmzr': 'adadelta',
    'actfunc': 'tanh',

    # Training related
    'iters': 1000,
    'learning_rate': 0.01,

    # Decoding related
    'k_clusters': 5,
    'beam_width': 10,
    'k_best': 1,
    'prune': True,
    'decode_task': 'sim'
}
