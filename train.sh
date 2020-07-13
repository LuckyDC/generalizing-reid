#!/bin/bash

# generate a port number randomly
MASTER_PORT=$RANDOM
let MASTER_PORT+=10000

# check whether the port has been occupied
ret=$(netstat -tlpn 2>/dev/null | grep "\b${MASTER_PORT}\b" | awk '{print $0}')

while [[ "$ret" != "" ]]
do
    MASTER_PORT=$RANDOM
    let MASTER_PORT+=10000
    ret=$(netstat -tlpn 2>/dev/null | grep "\b${MASTER_PORT}\b" | awk '{print $0}')
done

# get the number of process
NUM_PROC=0
for _ in ${1//,/ }
do
    let NUM_PROC++
done

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# launch the distributed program
OMP_NUM_THREADS=$TOTAL_CORES \
KMP_AFFINITY=granularity=fine,compact,1,0 \
KMP_BLOCKTIME=1 \
CUDA_VISIBLE_DEVICES=$1 \
python3 -m torch.distributed.launch --nproc_per_node=${NUM_PROC} --master_port ${MASTER_PORT} train.py --cfg $2