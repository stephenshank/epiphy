import json

from epiphy import *


parameters = load_parameters()


def params_from_wc(wc):
  return parameters[int(wc)]


rule gtr_fits:
  input:
    expand("data/simulate-{sim}/gtr_to_gtr.json", sim=range(len(parameters)))
  output:
    "data/gtr_simulation_fits.csv"
  run:
    harvest_results(input, output[0])

rule mg94_fits:
  input:
    expand("data/simulate-{sim}/mg94_to_mg94.json", sim=range(len(parameters)))
  output:
    "data/mg94_simulation_fits.csv"
  run:
    harvest_results(input, output[0])

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

rule fit_gtr_to_gtr:
  input:
    alignment=rules.filter_out_interior_gtr_nodes.output[0],
    tree=rules.simulate_tree.output[0]
  output:
    "data/simulate-{sim}/gtr_to_gtr.json"
  run:
    write_fit_gtr(input.alignment, input.tree, output[0])

rule fit_gtr_to_mg94:
  input:
    alignment=rules.filter_out_interior_mg94_nodes.output[0],
    tree=rules.simulate_tree.output[0]
  output:
    "data/simulate-{sim}/gtr_to_mg94.json"
  run:
    write_fit_gtr(input.alignment, input.tree, output[0])

rule fit_mg94_to_mg94:
  input:
    alignment=rules.filter_out_interior_mg94_nodes.output[0],
    tree=rules.simulate_tree.output[0],
    gtr_fit=rules.fit_gtr_to_mg94.output[0]
  output:
    "data/simulate-{sim}/mg94_to_mg94.json"
  run:
    write_fit_mg94(input.alignment, input.tree, input.gtr_fit, output[0])

rule simulate_full_epifel_alignment:
  input:
    rules.simulate_tree.output[0]
  output:
    "data/simulate-{sim}/epifel_full.fasta"
  run:
    write_epifel_simulation(input[0], output[0], params_from_wc(wildcards.sim))

rule filter_out_interior_epifel_nodes:
  input:
    rules.simulate_full_epifel_alignment.output[0]
  output:
    "data/simulate-{sim}/epifel.fasta"
  run:
    filter_simulated_alignment(input[0], output[0])

rule simulated_epifel_fna:
  input:
    tree=rules.simulate_tree.output[0],
    alignment=rules.filter_out_interior_epifel_nodes.output[0]
  output:
    "data/simulate-{sim}/epifel.fna"
  shell:
    "cat {input.alignment} {input.tree} > {output}"

rule empirical_gtr_fit:
  input:
    alignment="data/empirical/{empirical}.fasta",
    tree="data/empirical/{empirical}.new"
  output:
    "data/empirical/{empirical}-gtr.json"
  run:
    write_fit_gtr(input.alignment, input.tree, output[0])

rule empirical_mg94_fit:
  input:
    alignment="data/empirical/{empirical}.fasta",
    tree="data/empirical/{empirical}.new",
    gtr="data/empirical/{empirical}-gtr.json"
  output:
    "data/empirical/{empirical}-{codon1}-{codon2}-mg94.json"
  run:
    write_fit_mg94_pair(
      input.alignment, input.tree, input_gtr_json_path,
      wildcards.codon1, wildcards.codon2, output[0]
    )

rule all:
  input:
    rules.gtr_fits.output[0],
    rules.mg94_fits.output[0]
