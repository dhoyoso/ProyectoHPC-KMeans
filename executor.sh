#!/usr/bin/env bash
if ! [ -x /usr/bin/nproc ]; then
    echo "nproc is not installed. Please install it."
    exit 1
fi

mpiexec -np 4 python ./Paralel.py

