#!/bin/bash

if [[ -x "$1" ]] || [[ -e "$2" ]]; then
    echo "Usage: $0 [[PROGRAM]] [[JOB]]"
    exit 1
fi

parallel --colsep '	' -a "$2" "$(dirname "$0")./run.bash" {1} {2} {3} {4} {5} {6} {7} {8} "$1"
