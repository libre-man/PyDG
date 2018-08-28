# CodeGra.plag
This repository contains CodeGra.plag that makes it possible to create PDGs for
Python code. This research was done as the bachelor thesis of Thomas Schaper,
and was presented at [CompSys2018](https://www.aanmelder.nl/101005) where it
received a Outstanding Contribution award.

# Usage
This can only be run by python versions >=3.6, the dependencies can be found in
requirements.txt.

The most important function in this project is `create_pdg` that creates a PDG
for a AST (created by the `ast` module). This function returns a graph (see
`graph.py`) and can be rendered to DOT by calling the `to_dot` method.

# Acknowledgments
- Dr. Ana Lucia Varbanescu, for the enormous help during the bachelor thesis.
- Olmo Kramer, for the discussion during and before the project.

# Authors
- Thomas Schaper

# License
This project is licensed under AGPL - see the LICENSE file for details
