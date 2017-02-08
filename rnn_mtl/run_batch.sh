for tasks in $(cat $1); do

    #traindir=exps/$tasks
    if [[ $tasks != \#* ]] ;
    then
    #mkdir $traindir 2> /dev/null
    #echo "Starting training..." >> $traindir/stdout.log 
    #date >> $traindir/stdout.log 
    #nohup ~/anaconda3/bin/python src/rnn-mtl.py --data cfg/data.cfg --train_dir $traindir --clusters_in data/clusters/w2c.$clusters.txt --clusters_out data/clusters/w2c.1000.txt  --tasks cfg/task_$tasks.cfg --timesteps $timesteps --optimizer adadelta --n_layers 1 --iters 50000 >> $traindir/stdout.log 2>&1 &
    bash run.sh $tasks 1 50000
    #echo $tasks
    fi

done
