#!/bin/bash

sudo apt install build-essential parallel libboost-all-dev
g++ -std=c++17 mcgregor_subgraphs.cpp -o mcgregor_subgraphs -Ofast -march=native -flto
