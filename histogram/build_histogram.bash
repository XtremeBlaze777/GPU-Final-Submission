#!/bin/bash
if [ -z $1 ]
then
	echo "give a level of sync"
	exit 1
fi

nvcc $1_histogram.cu -o $1_histogram
nsys profile --output=$1_histogram --force-overwrite true --stats=true -t cuda ./$1_histogram
ncu --set full --target-processes all --force-overwrite -o $1_histogram ./$1_histogram
