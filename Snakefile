import json

from epiphy import *


parameters = load_parameters()


def params_from_wc(wc):
  return parameters[int(wc)]


rule all:
  input:
    expand("data/simulate-{sim}/gtr.json", sim=range(len(parameters)))

rule simulate_tree:
  output:
    "data/simulate-{sim}/tree.new"
  run:
    write_simulated_tree(output[0], params_from_wc(wildcards.sim))

rule simulate_full_gtr_alignment:
  input:
    rules.simulate_tree.output[0]
  output:
    "data/simulate-{sim}/gtr_full.fasta"
  run:
    write_gtr_simulation(input[0], output[0], params_from_wc(wildcards.sim))

rule filter_out_interior_gtr_nodes:
  input:
    rules.simulate_full_gtr_alignment.output[0]
  output:
    "data/simulate-{sim}/gtr.fasta"
  run:
    filter_simulated_alignment(input[0], output[0])

rule simulate_full_mg94_alignment:
  input:
    rules.simulate_tree.output[0]
  output:
    "data/simulate-{sim}/mg94_full.fasta"
  run:
    write_mg94_simulation(input[0], output[0], params_from_wc(wildcards.sim))

rule filter_out_interior_mg94_nodes:
  input:
    rules.simulate_full_mg94_alignment.output[0]
  output:
    "data/simulate-{sim}/mg94.fasta"
  run:
    filter_simulated_alignment(input[0], output[0])

rule fit_gtr:
  input:
    alignment=rules.simulate_full_gtr_alignment.output[0],
    tree=rules.simulate_tree.output[0]
  output:
    "data/simulate-{sim}/gtr.json"
  run:
    write_fit_gtr(input.alignment, input.tree, output[0])

