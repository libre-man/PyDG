#!/bin/bash

graph1="$7"
graph2="$8"
program="$9"

graph1="${graph1//\', \'/|}"
graph1="${graph1//\'/}"
graph1="${graph1//[/}"
graph1="${graph1//]/}"
graph2="${graph2//\', \'/|}"
graph2="${graph2//\'/}"
graph2="${graph2//[/}"
graph2="${graph2//]/}"

num="$("$program" "${graph1}|$graph2")"
echo -e "$1\\t$2\\t$3\\t$4\\t$5\\t$6\\t$num"
