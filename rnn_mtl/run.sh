clusters=10240
timesteps=30
tasks=$1
layers=$2
iters=$3
traindir=exps/$tasks
dropout=0.1

learning_rate=0.1
#optimizer=momentum
optimizer=adadelta

vecs=data/embeddings/glove.6B.100d.txt
#vecs=data/embeddings/en-es.chandar.vecs

echo $traindir
mkdir -p $traindir 2> /dev/null
echo "Starting training..." >> $traindir/stdout.log 
date >> $traindir/stdout.log 
nohup ~/anaconda3/bin/python src/experiment.py train --data cfg/data.cfg --train_dir $traindir --embeddings $vecs --task_cfg cfg/task_$tasks.cfg --timesteps $timesteps --optmzr $optimizer --n_layers $layers --iters $iters --learning_rate $learning_rate --dropout $dropout --save_interval 10000 >> $traindir/stdout.log 2>&1 &
