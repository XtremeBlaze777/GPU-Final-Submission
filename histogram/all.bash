#!/bin/bash

./build_histogram.bash block > /dev/null
./build_histogram.bash warp > /dev/null
./build_histogram.bash device > /dev/null

tar -cf histogram_1000000.tar *histogram.nsys-rep *histogram.ncu-rep *_histogram.cu
chmod 777 histogram_1000000.tar
# cp -v histogram_1000000.tar /tmp