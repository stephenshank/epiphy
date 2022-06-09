import json

from epiphy import *


all_parameters = load_parameters()
parameters = all_parameters[0]


rule simulate_tree:
  output:
    "data/simulate/tree.new"
  run:
    write_simulated_tree(output[0], parameters)

rule simulate_full_gtr_alignment:
  input:
    rules.simulate_tree.output[0]
  output:
    "data/simulate/gtr_full.fasta"
  run:
    write_gtr_simulation(input[0], output[0], parameters)

rule filter_out_interior_gtr_nodes:
  input:
    rules.simulate_full_gtr_alignment.output[0]
  output:
    "data/simulate/gtr.fasta"
  run:
    filter_simulated_alignment(input[0], output[0])

rule simulate_full_mg94_alignment:
  input:
    rules.simulate_tree.output[0]
  output:
    "data/simulate/mg94_full.fasta"
  run:
    write_mg94_simulation(input[0], output[0], parameters)

rule filter_out_interior_mg94_nodes:
  input:
    rules.simulate_full_mg94_alignment.output[0]
  output:
    "data/simulate/mg94.fasta"
  run:
    filter_simulated_alignment(input[0], output[0])

rule fit_gtr:
  input:
    alignment=rules.simulate_full_gtr_alignment.output[0],
    tree=rules.simulate_tree.output[0]
  output:
    "data/simulate/gtr.json"
  run:
    write_fit_gtr(input.alignment, input.tree, output[0])
