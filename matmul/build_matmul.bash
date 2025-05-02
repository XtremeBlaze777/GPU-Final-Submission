#!/bin/bash
if [ -z $1 ]
then
	echo "give a level of sync"
	exit 1
fi

nvcc $1_matmul.cu -o $1_matmul
nsys profile --output=$1_matmul --force-overwrite true --stats=true -t cuda ./$1_matmul
ncu --set full --target-processes all --launch-count 1 --force-overwrite -o $1_matmul ./$1_matmul
