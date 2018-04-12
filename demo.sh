#!/bin/bash
if [ -d build ]; then
  rm -r build
fi
mkdir build
cd build
cmake ..
make -j $(nproc)
cd ..
if [ -d out ]; then
  rm -r out
fi
build/test_detect
build/test_verify
build/test_time
