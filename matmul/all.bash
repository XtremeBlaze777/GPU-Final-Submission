#!/bin/bash

./build_matmul.bash block > /dev/null
# ./build_matmul.bash warp > /dev/null
./build_matmul.bash device > /dev/null

tar -cf matmul_2048.tar *matmul.nsys-rep *matmul.ncu-rep block_matmul.cu device_matmul.cu
chmod 777 matmul_2048.tar