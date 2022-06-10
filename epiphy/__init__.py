import json
import itertools as it
import csv

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import PhyloTree
from ete3 import Tree


def autotype(parameters):
    for key in parameters.keys():
        value = parameters[key]
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        parameters[key] = value
    return parameters


def load_parameters():
    with open('epiphy/parameters.csv') as f:
        all_parameters = [
            autotype(parameters)
            for parameters in csv.DictReader(f)
        ]
    return all_parameters


nucleotides = np.array(['A', 'C', 'G', 'T'], dtype='<U1')
nuc2ind = { 'A': 0, 'C': 1, 'G': 2, 'T': 3 }
codons = [''.join(c) for c in it.product(nucleotides, repeat=3)]
stop_codons = ['TAA', 'TAG', 'TGA']
sense_codons = [c for c in codons if not c in stop_codons]
cod2ind = {codon: i for i, codon in enumerate(sense_codons)}


def simulate_tree(parameters, ladderization=1.5):
    number_of_sequences = parameters['number_of_sequences']
    mean_bl = parameters['mean_bl']
    stddev_bl = parameters['stddev_bl']
    seed = parameters['seed']
    np.random.seed(seed)
    tree = Tree(name='root')
    def sample_branch_length():
        return np.max(
            stddev_bl*np.random.randn() + mean_bl,
            0
        )
    first_child = tree.add_child(dist=sample_branch_length())
    second_child = tree.add_child(dist=sample_branch_length())
    children_list = [first_child, second_child]
    children_fitnesses = [.5, .5]
    for i in range(0, number_of_sequences - 2):
        chosen_index = np.random.choice(
            len(children_list), 1, p=children_fitnesses
        )[0]
        chosen_fitness = children_fitnesses[chosen_index]
        chosen_child = children_list[chosen_index]
        del children_fitnesses[chosen_index]
        del children_list[chosen_index]
        first_child = chosen_child.add_child(
            dist=sample_branch_length()
        )
        second_child = chosen_child.add_child(
            dist=sample_branch_length()
        )
        children_list.append(first_child)
        children_list.append(second_child)
        children_fitnesses += 2*[ladderization*chosen_fitness]
        normalization = np.sum(children_fitnesses)
        children_fitnesses = [
            child_fitness/normalization
            for child_fitness in children_fitnesses
        ]

    current_leaf_name = 1
    current_branch_name = 1
    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.name = 'Leaf-%d' % current_leaf_name
            current_leaf_name += 1
        else:
            if node.is_root():
                node.name = 'Node-0'
            else:
                node.name = 'Node-%d' % current_branch_name
            current_branch_name += 1
    return tree


def write_simulated_tree(tree_path, parameters):
    tree = simulate_tree(parameters)
    tree.write(format=1, outfile=tree_path)


def generic_parser(record):
    return [nuc2ind[char] for char in str(record.seq)]


def codon_parser(record):
    error_message = 'Alignment length not a multiple of 3'
    assert len(record.seq) % 3 == 0, error_message
    for i in range(0, len(record.seq), 3):
        return cod2ind[str(record.seq[i:i+3])]


def ef_key(nucleotide, i):
    if type(nucleotide) == int:
        nucleotide = nucleotides[nucleotide]
    key = 'pi%s%d' % (nucleotide, i+1)
    return key


def calculate_pi_stop(parameters):
    pi_stop = 0
    for codon in stop_codons:
        summand = 1
        for i, nucleotide in enumerate(codon):
            key = ef_key(nucleotide, i)
            summand *= parameters[key]
        pi_stop += summand
    return pi_stop


def f3x4(parameters):
    pi_stop = calculate_pi_stop(parameters)
    ef = []
    for codon in sense_codons:
        term = 1
        for i, nucleotide in enumerate(codon):
            key = ef_key(nucleotide, i) 
            term *= parameters[key]
        ef.append(term/(1-pi_stop))
    error_message = 'F3x4 estimator does not sum to 1'
    ef_np = np.array(ef, dtype=float)
    assert np.abs(1-np.sum(ef_np)) < 1e-8, error_message
    return ef_np


def position_specific_empirical_nucleotide_frequencies(alignment):
    F = np.zeros((4, 3))
    total = alignment.shape[0]*alignment.shape[1] // 3
    for nucleotide in range(4):
        for position in range(3):
            count = (alignment[:, position::3] == nucleotide).sum()
            F[nucleotide, position] = count / total
    return F


def mg94_matrix(parameters):
    Q = np.zeros((61, 61), dtype=float)
    for I, codon_I in enumerate(sense_codons):
        for J, codon_J in enumerate(sense_codons):
            if I == J:
                continue
            change1 = codon_I[0] != codon_J[0]
            change2 = codon_I[1] != codon_J[1]
            change3 = codon_I[2] != codon_J[2]
            total_changes = np.sum([change1, change2, change3])
            if total_changes > 1:
                continue
            if change1:
                pos = 0
            elif change2:
                pos = 1
            else:
                pos = 2
            nuc_i = codon_I[pos]
            nuc_j = codon_J[pos]
            if nuc_i < nuc_j:
                theta_key = '%s%s' % (nuc_i, nuc_j)
            else:
                theta_key = '%s%s' % (nuc_j, nuc_i)
            theta = parameters[theta_key]
            pi_key = 'pi%s%d' % (nuc_j, pos+1)
            pi = parameters[pi_key]
            aminoacid_I = str(Seq(codon_I).translate())
            aminoacid_J = str(Seq(codon_J).translate())
            is_nonsynonymous = aminoacid_I != aminoacid_J
            omega = parameters['omega'] if is_nonsynonymous else 1
            Q[I,J] = theta*omega*pi
        Q[I, I] = -np.sum(Q[I, :])
    return Q


def simulate_mg94_root(parameters):
    pi = f3x4(parameters)
    root = np.random.choice(61, parameters['number_of_sites'], p=pi)
    return root


def simulate_mg94(parameters, tree, seed=1):
    np.random.seed(seed)
    seq_dict = {}
    records = []
    root = simulate_mg94_root(parameters)
    Q = mg94_matrix(parameters)
    for node in tree.traverse('preorder'):
        if node.is_root():
            seq_dict[node.name] = root
        else:
            P = expm(node.dist*Q)
            new_sequence = np.empty(parameters['number_of_sites'], dtype=int)
            parent_sequence = seq_dict[node.up.name]
            for I in range(61):
                indices = parent_sequence == I
                new_sequence[indices] = np.random.choice(
                    61, size=np.sum(indices), p=P[I,:]
                )
            seq_dict[node.name] = new_sequence
            sequence = ''.join([sense_codons[I] for I in new_sequence])
            record = SeqRecord(Seq(sequence), id=node.name, description='')
            records.append(record)
    return records


def write_mg94_simulation(input_tree_path, alignment_path, parameters):
    tree = Tree(input_tree_path, format=1)
    records = simulate_mg94(parameters, tree)
    SeqIO.write(records, alignment_path, 'fasta')


def read_alignment(alignment_path, is_codon=True):
    records = SeqIO.parse(alignment_path, 'fasta')
    parser = codon_parser if is_codon else generic_parser
    sequence_data = []
    headers = []
    for record in records:
        sequence_data.append(parser(record))
        headers.append(record.id)
    alignment = np.array(sequence_data, dtype=int)
    return alignment, headers


def read_alignment_and_tree(alignment_path, tree_path, is_codon=True):
    alignment, headers = read_alignment(alignment_path, is_codon)
    with open(tree_path) as tree_file:
        tree_string = tree_file.read()
    tree = Tree(tree_string, format=1)
    tree_hash = set([node.name for node in tree.traverse('postorder')])
    seq_dict = {}
    for i, header in enumerate(headers):
        error_message = 'Header in alignment not found in tree... aborting!'
        assert header in tree_hash, error_message
        seq_dict[header] = alignment[i, :]
    return alignment, seq_dict, tree


def filter_simulated_alignment(input_alignment_path, output_alignment_path):
    records = SeqIO.parse(input_alignment_path, 'fasta')
    filtered_records = [
        record for record in records if 'Leaf' in record.id
    ]
    SeqIO.write(filtered_records, output_alignment_path, 'fasta')


def gtr_pi(parameters, pos):
    return [
        parameters[ef_key(nucleotide, pos)]
        for nucleotide in nucleotides
    ]


def gtr_matrix(parameters, pos):
    Q_gtr = np.array([
        [0,                 parameters['AC'],   parameters['AG'],   parameters['AT'] ],
        [parameters['AC'],  0,                  parameters['CG'],   parameters['CT'] ],
        [parameters['AG'],  parameters['CG'],   0,                  parameters['GT'] ],
        [parameters['AT'],  parameters['CT'],   parameters['GT'],   0                ]
    ])
    pi = gtr_pi(parameters, pos)
    Q = np.dot(Q_gtr, np.diag(pi))
    for i in range(4):
        Q[i, i] = -np.sum(Q[i, :])
    return Q


def simulate_gtr_root(parameters):
    pi0 = gtr_pi(parameters, 0)
    pi1 = gtr_pi(parameters, 1)
    pi2 = gtr_pi(parameters, 2)
    root = np.zeros(parameters['number_of_sites'], dtype=int)
    number_of_triplets = parameters['number_of_sites'] // 3
    root[::3] = np.random.choice(4, number_of_triplets, p=pi0)
    root[1::3] = np.random.choice(4, number_of_triplets, p=pi1)
    root[2::3] = np.random.choice(4, number_of_triplets, p=pi2)
    return root


def simulate_gtr(parameters, tree, seed=1):
    np.random.seed(seed)
    seq_dict = {}
    records = []
    root = simulate_gtr_root(parameters)
    Q1 = gtr_matrix(parameters, 0)
    Q2 = gtr_matrix(parameters, 1)
    Q3 = gtr_matrix(parameters, 2)
    Qs = [Q1, Q2, Q3]
    number_of_sites = parameters['number_of_sites']
    for node in tree.traverse('preorder'):
        if node.is_root():
            seq_dict[node.name] = root
        else:
            new_sequence = np.empty(number_of_sites, dtype=int)
            parent_sequence = seq_dict[node.up.name]
            for p in range(3):
                P = expm(node.dist*Qs[p])
                for i in range(4):
                    all_nucleotide_matches = parent_sequence == i
                    new_nucleotides = np.random.choice(
                        4, size=np.sum(all_nucleotide_matches[p::3]), p=P[i, :]
                    )
                    matching_nuc_position = np.arange(0, number_of_sites) %3 == p
                    indices = matching_nuc_position & all_nucleotide_matches
                    new_sequence[indices] = new_nucleotides
            seq_dict[node.name] = new_sequence
            sequence = ''.join([nucleotides[i] for i in new_sequence])
            record = SeqRecord(Seq(sequence), id=node.name, description='')
            records.append(record)
    return records


def write_gtr_simulation(input_tree_path, output_alignment_path, parameters):
    tree = Tree(input_tree_path, format=1)
    records = simulate_gtr(parameters, tree)
    SeqIO.write(records, output_alignment_path, 'fasta')


def get_node_index(parameters, node):
    if node.is_root():
        return parameters['node_dict']['root']
    return parameters['node_dict'][node.name]


def gtr_prune(parameters, tree, branch_lengths=None):
    Q1 = gtr_matrix(parameters, 0)
    Q2 = gtr_matrix(parameters, 1)
    Q3 = gtr_matrix(parameters, 2)
    number_of_sites = parameters['number_of_sites']
    number_of_sequences = parameters['number_of_sequences']
    L = np.zeros((4, number_of_sites, 2*number_of_sequences-1))
    all_sequence_indices = np.arange(number_of_sites)
    seq_dict = parameters['seq_dict']
    for node in tree.traverse('postorder'):
        node_index = get_node_index(parameters, node)
        if node.is_leaf():
            L[seq_dict[node.name], all_sequence_indices, node_index] = 1
        else:
            for i, child in enumerate(node.children):
                child_index = get_node_index(parameters, child)
                if not branch_lengths is None:
                    t = branch_lengths[child_index]
                else:
                    t = child.dist
                P1 = expm(t*Q1)
                P2 = expm(t*Q2)
                P3 = expm(t*Q3)
                if i == 0:
                    L[:, ::3, node_index] = np.dot(P1, L[:, ::3, child_index])
                    L[:, 1::3, node_index] = np.dot(P2, L[:, 1::3, child_index])
                    L[:, 2::3, node_index] = np.dot(P3, L[:, 2::3, child_index])
                else:
                    L[:, ::3, node_index] *= np.dot(P1, L[:, ::3, child_index])
                    L[:, 1::3, node_index] *= np.dot(P2, L[:, 1::3, child_index])
                    L[:, 2::3, node_index] *= np.dot(P3, L[:, 2::3, child_index])
        if node.is_root():
            pi0 = gtr_pi(parameters, 0)
            pi1 = gtr_pi(parameters, 1)
            pi2 = gtr_pi(parameters, 2)
            l0 = np.dot(pi0, L[:, ::3, node_index])
            l1 = np.dot(pi1, L[:, 1::3, node_index])
            l2 = np.dot(pi2, L[:, 2::3, node_index])
            return np.sum(np.log(l0)) + np.sum(np.log(l1)) + np.sum(np.log(l2))


def build_node_dict(tree):
    node_dict = {}
    for i, node in enumerate(tree.traverse('postorder')):
        if not node.is_root():
            node_key = node.name
        else:
            node_key = 'root'
        node_dict[node_key] = i
    return node_dict


def build_gtr_likelihood(alignment, seq_dict, tree, verbose=True):
    freq = position_specific_empirical_nucleotide_frequencies(alignment)
    node_dict = build_node_dict(tree)
    def likelihood(x):
        parameters = {
            'number_of_sequences': alignment.shape[0],
            'number_of_sites': alignment.shape[1],
            'seq_dict': seq_dict,
            'node_dict': node_dict
        }
        for nuc in range(4):
            for pos in range(3):
                parameters[ef_key(nuc, pos)] = freq[nuc, pos]
        parameters['AC'] = x[0]
        parameters['AG'] = 1
        parameters['AT'] = x[1]
        parameters['CG'] = x[2]
        parameters['CT'] = x[3]
        parameters['GT'] = x[4]
        branch_lengths = x[5:]
        if verbose:
            for k, v in parameters.items():
                print('%s: %s\n' % (str(k), str(v)))
            print('\n----------\n')
        return gtr_prune(parameters, tree, branch_lengths)
    return likelihood


def get_initial_gtr_guess(seq_dict, tree):
    number_of_sequences = len(seq_dict)
    n_branch_lengths = 2*number_of_sequences - 2
    node_dict = build_node_dict(tree)
    branch_lengths = np.zeros(n_branch_lengths)
    for node in tree.traverse('postorder'):
        if not node.is_root():
            node_index = node_dict[node.name]
            branch_lengths[node_index] = node.dist
    return np.hstack([
        .25*np.ones(5),
        branch_lengths
    ])


def maximize(f, x0, jac=None, method='Nelder-Mead', options={}):
    def g(x):
        return -f(x)
    return minimize(g, x0=x0, jac=jac, method=method, options=options)


def gtr_gradient(alignment, tree):
    def gradient(x):
        pass
    return gradient


def fit_gtr(alignment, tree, initial_guess):
    x = minimize(f, x0=[0, 0], method='Nelder-Mead')
    pass


def write_fit_gtr(input_alignment_path, input_tree_path, output_json_path):
    pass


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    #tree_path = 'data/simulate/tree.new'
    #alignment_path = 'data/simulate/gtr.fasta'
    #alignment, seq_dict, tree = read_alignment_and_tree(alignment_path, tree_path, False)
    #x0 = get_initial_gtr_guess(seq_dict, tree)
    #likelihood = build_gtr_likelihood(alignment, seq_dict, tree)
    #x = maximize(likelihood, x0, options={'maxiter': 1000})
    parameters = load_parameters()
    write_simulated_tree('test.new', parameters[0])
