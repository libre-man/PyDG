# Experiments
This subdirectory contains all experiments done for detection plagiarism using codegra.plag.

# Usage
To experiment with codegra.plag you have two steps:

1. Create a csv file with that represents a job.
2. Execute that job.

## Creating jobs
To create a job you can use the the `create_job.py` script, however you can also
create your own jobs by modifying this script. All jobs should have a specific
format.

A job should contain the following fields separated by tabs:
1. Base path of the first student;
2. Base path of the second student;
3. Base path plus file of the first student;
4. Base path plus file of the second student;
5. Name of function of first student;
5. Name of function of second student;
6. Graph of first student;
7. Graph of second student

The graphs should be the string representation of the following python list:
1. First all nodes in number form, in sequential order.
2. The edges between all the nodes in the form `$NODE_NUMBER1,$NODE_NUMBER2`.

You can find an example job in `example_job.csv`.

## Executing jobs
To execute jobs you should use `experiment.bash [[PROGRAM]] [[JOB]]`, where
`[[PROGRAM]]` is the program used to compare two graphs and `[[JOB]]` is the job
that is created in step 1.

The program used to analyze the graphs is called with a single argument: the two
graphs separated by a single `|`. The graph format is changed by removing all
comma's into pipes (`|`) and removing all apostrophes (`'`) and brackets (`[`
and `]`). The program should output something on standard out that doesn't
contain a newline (`\n`) or tab (`\t`). You can use `mcgreggor_subgraphs.cpp`
(which you should of course compile first) as analyzing program which outputs a
number that estimates the Maximum Common Subgraph (MCS).

This produces output in the fields separated by tabs:
1. Base path of the first student;
2. Base path of the second student;
3. Base path plus file of the first student;
4. Base path plus file of the second student;
5. Name of function of first student;
5. Name of function of second student;
6. The output of the program that analyzed the graphs.
