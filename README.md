# This repository
This repository contains the code used for the EACL '17 paper "Beneficial task relations for multi-task learning in deep neural networks"

## Experiments

The results presented in the paper were obtained through the experiments in `experiments.ipynb`. Have a look at this notebook to reproduce our results. The training curve logs from the single- and multi-task models are place in the `logs` directory.

## RNN code

This repository also includes a copy of the (fairly documented) Tensorflow RNN code used for the single-task and multi-task networks, and is placed in the `rnn_mtl` folder, along with the scripts we used to run those experiments. NB: The code is being developed further in a different repository, please get in touch via email if you're interested in an up-to-date version.

To re-run the experiments, we provide a script called `run.sh`, or `run_batch.sh` to run several single- and MTL models at once. We ran our experiments on a GPU cluster, all in all they took about 24 hours to finish. 
