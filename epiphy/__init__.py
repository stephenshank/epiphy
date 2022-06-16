import json
import itertools as it
import csv
from collections import Counter
import sys

import numpy as np
import scipy.sparse as spsp
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import fmin_l_bfgs_b
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree
from joblib import Parallel, delayed

np.seterr(all='raise')


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
nuc2ind = { 
    'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, 'R': 4, 'Y': 4,
    'S': 4, 'W': 4, 'K': 4, 'M': 4, 'B': 4, 'D': 4, 'H': 4,
    'V': 4, 'N': 4
}
codons = [''.join(c) for c in it.product(nucleotides, repeat=3)]
stop_codons = ['TAA', 'TAG', 'TGA']
sense_codons = [c for c in codons if not c in stop_codons]
dicodons = [d[0] + d[1] for d in it.product(sense_codons, sense_codons)]
di2ind = {dicodon: i for i, dicodon in enumerate(dicodons)}
cod2ind = {codon: i for i, codon in enumerate(sense_codons)}
cod2ind['---'] = 61
cod2aa = {codon: str(Seq(codon).translate()) for codon in sense_codons}
aa_charges = {
    'R': '+', 'H': '+', 'K': '+', 'D': '-', 'E': '-',
    'S': 'n', 'T': 'n', 'N': 'n', 'Q': 'n', 'C': 'n',
    'G': 'n', 'P': 'n', 'A': 'n', 'V': 'n', 'I': 'n',
    'L': 'n', 'M': 'n', 'F': 'n', 'Y': 'n', 'W': 'n'
}


def simulate_tree(parameters, ladderization=1.5):
    number_of_sequences = parameters['number_of_sequences']
    mean_bl = parameters['mean_bl']
    stddev_bl = parameters['stddev_bl']
    seed = parameters['seed']
    np.random.seed(seed)
    tree = Tree(name='root')
    def sample_branch_length():
        return np.max([
            stddev_bl*np.random.randn() + mean_bl,
            0
        ])
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
    return np.array([
        cod2ind[str(record.seq[i:i+3])]
        for i in range(0, len(record.seq), 3)
    ], dtype=int)


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
    codon_ef = []
    error_message = 'Nucleotide frequencies at position %d do not sum to 1'
    pi_sums = np.zeros(3)
    for nucleotide, pos in it.product(nucleotides, (0, 1, 2)):
        key = ef_key(nucleotide, pos) 
        nuc_ef = parameters[key]
        pi_sums[pos] += nuc_ef
    assert np.abs(1-pi_sums[0]) < 1e-8, error_message % 1
    assert np.abs(1-pi_sums[1]) < 1e-8, error_message % 2
    assert np.abs(1-pi_sums[2]) < 1e-8, error_message % 3
    for codon in sense_codons:
        term = 1
        for i, nucleotide in enumerate(codon):
            key = ef_key(nucleotide, i) 
            term *= parameters[key]
        codon_ef.append(term/(1-pi_stop))
    ef_np = np.array(codon_ef, dtype=float)
    error_message = 'F3x4 estimator does not sum to 1'
    assert np.abs(1-np.sum(ef_np)) < 1e-8, error_message
    return ef_np


def psenf_from_codons(alignment):
    n_seq, n_site = alignment.shape
    F = np.zeros((4, 3))
    count = 0
    for i in range(n_seq):
        for j in range(n_site):
            codon = sense_codons[alignment[i, j]]
            count += 1
            for i, nuc in enumerate(codon):
                F[nuc2ind[nuc], i] += 1
    return F / count


def position_specific_empirical_nucleotide_frequencies(alignment, is_codon=False):
    if is_codon:
        return psenf_from_codons(alignment)
    F = np.zeros((4, 3))
    total = alignment.shape[0]*alignment.shape[1] // 3
    for nucleotide in range(4):
        for position in range(3):
            count = (alignment[:, position::3] == nucleotide).sum()
            nongaps = (alignment[:, position::3] != 4).sum()
            F[nucleotide, position] = count / nongaps
    error_message = 'Nucleotide frequencies at position %d do not sum to 1'
    for i in range(3):
        assert np.abs(F[:, i].sum() - 1) < 1e-8, error_message % (i+1)
    return F


def freqs_to_pi_dict(freqs):
    pi_dict = {}
    for i, nuc in enumerate(nucleotides):
        for pos in range(3):
           pi_dict[ef_key(nuc, pos)] = freqs[i, pos] 
    return pi_dict


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
    header_hash = set(headers)
    for node in tree.traverse('postorder'):
        error_message = 'Header %s in tree not found in alignment... aborting!'
        if node.is_leaf():
            assert node.name in header_hash, error_message % node.name

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
            encoded_sequence = seq_dict[node.name]
            ambigs = encoded_sequence == 4
            L[encoded_sequence[~ambigs], all_sequence_indices[~ambigs], node_index] = 1
            L[:, all_sequence_indices[ambigs], node_index] = .25
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


def build_gtr_likelihood(alignment, seq_dict, tree, verbosity=0):
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
        l = gtr_prune(parameters, tree, branch_lengths)
        if verbosity > 0:
            print('Log likelihood:', l)
        if verbosity > 1:
            for k, v in parameters.items():
                print('%s: %s\n' % (str(k), str(v)))
            print('\n----------\n')
        return l
    return likelihood


def gtr_vector2dict(x, node_dict, alignment):
    parameters = {}
    parameters['AC'] = x[0]
    parameters['AG'] = 1
    parameters['AT'] = x[1]
    parameters['CG'] = x[2]
    parameters['CT'] = x[3]
    parameters['GT'] = x[4]
    freq = position_specific_empirical_nucleotide_frequencies(alignment)
    for nuc in range(freq.shape[0]):
        for pos in range(freq.shape[1]):
            parameters[ef_key(nuc, pos)] = freq[nuc, pos]
    for node_name, node_index in node_dict.items():
        parameters[node_name] = x[node_index]
    return parameters


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


def numeric_gradient(f, n, h):
    def gradient(x):
        result = np.zeros(n)
        for i in range(n):
            x_copy = np.copy(x)
            x_copy[i] += h
            fph = f(x_copy)
            x_copy[i] -= 2*h
            fmh = f(x_copy)
            result[i] = (fph-fmh)/(2*h)
        return result
    return gradient


def maximize(f, x0, jac=None, method='Nelder-Mead', options={}):
    def g(x):
        return -f(x)
    return minimize(g, x0=x0, jac=jac, method=method, options=options)


def minusf(f):
    def mf(x):
        return -f(x)
    return mf


def coordinate_objective(f, x0, i):
    x = np.copy(x0)
    def objective(xi):
        x[i] = xi
        return -f(x)
    return objective


def gtr_coordinate_descent(alignment, tree, seq_dict, niter=100, verbosity=0):
    # coordinate descent
    likelihood = build_gtr_likelihood(alignment, seq_dict, tree, verbosity)
    x = get_initial_gtr_guess(seq_dict, tree)
    history = []
    for iter in range(niter):
        for i in range(len(x)):
            f_i = coordinate_objective(likelihood, x, i)
            result = minimize_scalar(f_i, bounds=(0, 5), method='bounded')
            xi = result.x
            x[i] = xi
        current_value = -result.fun
        print('Iteration %d, likelihood=%.10f' % (iter+1, current_value))
        history.append(current_value)
    return x, history


def gtr_lbfgs(alignment, tree, seq_dict, niter=100, verbosity=0):
    def callback(x):
        print('Iteration...')
    likelihood = build_gtr_likelihood(alignment, seq_dict, tree, verbosity)
    x0 = get_initial_gtr_guess(seq_dict, tree)
    bounds = len(x0) * [(0, np.inf)]
    result = fmin_l_bfgs_b(minusf(likelihood), x0, approx_grad=True, bounds=bounds, callback=callback)
    return result[0]


def write_fit_gtr(input_alignment_path, input_tree_path, output_json_path, method='lbfgs'):
    alignment, seq_dict, tree = read_alignment_and_tree(
        input_alignment_path, input_tree_path, False
    )
    if method=='lbfgs':
        history = None
        x = gtr_lbfgs(alignment, tree, seq_dict)
    else:
        x, history = gtr_coordinate_descent(alignment, tree, seq_dict, 1, 1)

    node_dict = build_node_dict(tree)
    result = gtr_vector2dict(x, node_dict, alignment)
    if not history is None:
        result['history'] = history
    with open(output_json_path, 'w') as json_file:
        json.dump(result, json_file, indent=2)


def harvest_results(input_jsons, output_csv):
    csv_file = open(output_csv, 'w')
    for i, json_path in enumerate(input_jsons):
        with open(json_path) as json_file:
            datum = json.load(json_file)
        if i == 0:
            fieldnames = {
                key
                for key, value in datum.items()
                if type(value) != list and type(value) != dict
            }
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
        row = {
            key: value
            for key, value in datum.items()
            if key in fieldnames
        }
        writer.writerow(row)
    csv_file.close()


def mg94_prune(parameters, tree):
    Q = mg94_matrix(parameters)
    number_of_sites = parameters['number_of_sites']
    number_of_sequences = parameters['number_of_sequences']
    L = np.zeros((61, number_of_sites, 2*number_of_sequences-1))
    all_sequence_indices = np.arange(number_of_sites)
    seq_dict = parameters['seq_dict']
    for node in tree.traverse('postorder'):
        node_index = get_node_index(parameters, node)
        if node.is_leaf():
            L[seq_dict[node.name], all_sequence_indices, node_index] = 1
        else:
            for i, child in enumerate(node.children):
                child_index = get_node_index(parameters, child)
                t = child.dist
                P = expm(t*Q)
                if i == 0:
                    L[:, :, node_index] = np.dot(P, L[:, :, child_index])
                else:
                    L[:, :, node_index] *= np.dot(P, L[:, :, child_index])
            if np.linalg.norm(L[:, :, node_index], 'fro') == 0:
                pass
        if node.is_root():
            pi = f3x4(parameters)
            l0 = np.dot(pi, L[:, :, node_index])
            return np.sum(np.log(l0))


def build_mg94_likelihood(alignment, seq_dict, tree, verbosity=0, gtr_result=None, pair=False):
    if gtr_result is None:
        freq = position_specific_empirical_nucleotide_frequencies(alignment, True)
    else:
        synchronize_tree_and_results(tree, gtr_result)
    node_dict = build_node_dict(tree)
    def likelihood(x):
        parameters = {
            'number_of_sequences': alignment.shape[0],
            'number_of_sites': alignment.shape[1] if not pair else 2,
            'seq_dict': seq_dict,
            'node_dict': node_dict
        }
        for nuc in range(4):
            for pos in range(3):
                key = ef_key(nuc, pos)
                if gtr_result is None:
                    parameters[key] = freq[nuc, pos]
                else:
                    parameters[key] = gtr_result[key]

        parameters['AC'] = x[0]
        parameters['AG'] = 1
        parameters['AT'] = x[1]
        parameters['CG'] = x[2]
        parameters['CT'] = x[3]
        parameters['GT'] = x[4]
        parameters['omega'] = x[5]
        l = mg94_prune(parameters, tree)
        if verbosity > 0:
            print('Log likelihood:', l)
        if verbosity > 1:
            for k, v in parameters.items():
                print('%s: %s\n' % (str(k), str(v)))
            print('\n----------\n')
        return l
    return likelihood


def mg94_vector2dict(x, node_dict):
    parameters = {}
    parameters['AC'] = x[0]
    parameters['AG'] = 1
    parameters['AT'] = x[1]
    parameters['CG'] = x[2]
    parameters['CT'] = x[3]
    parameters['GT'] = x[4]
    parameters['omega'] = x[5]
    parameters['S'] = x[5]
    for node_name, node_index in node_dict.items():
        parameters[node_name] = x[node_index]
    return parameters


def get_initial_mg94_guess(seq_dict, tree, gtr_result):
    number_of_sequences = len(seq_dict)
    n_branch_lengths = 2*number_of_sequences - 2
    node_dict = build_node_dict(tree)
    branch_lengths = np.zeros(n_branch_lengths)
    for node in tree.traverse('postorder'):
        if not node.is_root():
            node_index = node_dict[node.name]
            branch_lengths[node_index] = gtr_result[node.name]
    return np.hstack([
        .25*np.ones(5),
        np.ones(2),
        branch_lengths
    ])


def mg94_lbfgs(alignment, tree, seq_dict, gtr_result, niter=100, verbosity=0, pair=False):
    def callback(x):
        print('Iteration...')
    likelihood = build_mg94_likelihood(alignment, seq_dict, tree, verbosity, gtr_result, pair=pair)
    x0 = get_initial_mg94_guess(seq_dict, tree, gtr_result)
    bounds = len(x0) * [(0, np.inf)]
    result = fmin_l_bfgs_b(minusf(likelihood), x0, approx_grad=True, bounds=bounds, callback=callback)
    return result[0]


def synchronize_tree_and_results(tree, results):
    for node in tree.traverse('postorder'):
        node_name = node.name if not node.is_root() else 'root'
        node.dist = results[node_name]


def write_fit_mg94(input_alignment_path, input_tree_path, input_gtr_json_path,
        output_json_path, method='lbfgs'):
    alignment, seq_dict, tree = read_alignment_and_tree(
        input_alignment_path, input_tree_path
    )
    with open(input_gtr_json_path) as json_file:
        gtr_result = json.load(json_file)
    synchronize_tree_and_results(tree, gtr_result)
    x = mg94_lbfgs(alignment, tree, seq_dict, gtr_result)
    node_dict = build_node_dict(tree)
    result = mg94_vector2dict(x, node_dict)
    with open(output_json_path, 'w') as json_file:
        json.dump(result, json_file, indent=2)


def independent_dicodon_matrix(parameters):
    I = spsp.eye(61).tocsr()
    C = mg94_matrix(parameters)
    D = spsp.kron(C, I) + spsp.kron(I, C)
    return D


def get_row_and_column_indices(D):
    row_indices = D.indices
    column_indices = np.zeros(len(D.indices), dtype=int)
    for i in range(len(D.indptr)):
        start_index = D.indptr[i]
        if i != len(D.indptr) - 1:
            stop_index = D.indptr[i+1]
            column_indices[start_index: stop_index] = i
        else:
            column_indices[start_index:] = i
    return row_indices, column_indices


def markovize(sparse_matrix):
    n = sparse_matrix.shape[0]
    ones = np.ones(n)
    row_sums = sparse_matrix.dot(ones)
    sparse_correction = spsp.spdiags(row_sums, [0], n, n).tocsc()
    markovized_matrix = sparse_matrix - sparse_correction
    error_message = 'Matrix is not Markovian'
    absolute_error = np.linalg.norm(markovized_matrix.dot(ones))
    relative_error = absolute_error / np.linalg.norm(ones)
    assert relative_error < 1e-12, error_message


def epistatic_matrix(parameters, favored_state_key='no_positives'):
    D = independent_dicodon_matrix(parameters)
    row_indices, column_indices = get_row_and_column_indices(D)
    codon_1_from = row_indices // 61
    codon_2_from = row_indices % 61
    codon_1_to = column_indices // 61
    codon_2_to = column_indices % 61
    charge_index_array = np.array([
        aa_charges[cod2aa[codon]]
        for codon in sense_codons
    ], dtype='<U1')
    codon_1_from_state = charge_index_array[codon_1_from]
    codon_2_from_state = charge_index_array[codon_2_from]
    codon_1_to_state = charge_index_array[codon_1_to]
    codon_2_to_state = charge_index_array[codon_2_to]
    all_favored_states = {
        'no_positives': [
            ('-', '-'), ('n', '-'), ('-', 'n'), ('n', 'n')
        ]
    }
    favored_states = all_favored_states[favored_state_key]

    from_favored = np.array(len(row_indices)*[False])
    for state in favored_states:
        from_1 = codon_1_from_state == state[0]
        from_2 = codon_2_from_state == state[1]
        from_favored = from_favored | (from_1 & from_2)
    from_unfavored = ~from_favored

    to_favored = np.array(len(row_indices)*[False])
    for state in favored_states:
        to_1 = codon_1_to_state == state[0]
        to_2 = codon_2_to_state == state[1]
        to_favored = to_favored | (to_1 & to_2)
    to_unfavored = ~to_favored
    
    from_unfavored_to_favored = from_unfavored & to_favored
    from_favored_to_unfavored = from_favored & to_unfavored

    D.data[from_unfavored_to_favored] *= parameters['epsilon']
    D.data[from_favored_to_unfavored] /= parameters['epsilon']

    return D


def krylov_subpace_exponential(A, b, iter=50):
    dim = A.shape[0]
    V = np.zeros((dim, iter+1))
    beta = np.linalg.norm(b)
    V[:, 0] = b / beta
    H = np.zeros((iter+1, iter))

    for j in range(iter):
        w = A.dot(V[:, j])
        for i in range(1, j):
            H[i, j] = np.dot(w, V[:, i])
            w = w - H[i, j] * V[:, i]
        H[j+1, j] = np.linalg.norm(w)
        if np.abs(H[j+1, j]) < 1e-12:
            print('Krylov iterations terminating early...')
            e = np.zeros(j)
            e[0] = 1
            x = beta*np.dot(V[:, :j], np.dot(expm(H[:j, :j]), e) )
            return x
        V[:, j+1] = w / H[j+1, j]
        e = np.zeros(j+1)
        e[0] = 1
    x = beta*np.dot(V[:, :iter], np.dot(expm(H[:iter, :iter]), e) )
    return x


def simulate_epifel_root(parameters):
    pi = f3x4(parameters)
    root = np.random.choice(61, 2, p=pi)
    dicodon = sense_codons[root[0]] + sense_codons[root[1]]
    dicodon_index = di2ind[dicodon]
    return dicodon_index


def normalize(p):
    return p / p.sum()


def simulate_epifel(parameters, tree, seed=1):
    np.random.seed(seed)
    seq_dict = {}
    records = []
    root = simulate_epifel_root(parameters)
    Q = epistatic_matrix(parameters).transpose().tocsc()
    for node in tree.traverse('preorder'):
        if node.is_root():
            seq_dict[node.name] = root
        else:
            parent_sequence = seq_dict[node.up.name]
            if node.dist < 1e-12:
                new_sequence = parent_sequence
            else:
                e = np.zeros(len(dicodons))
                e[parent_sequence] = 1
                p = krylov_subpace_exponential(node.dist*Q, e)
                new_sequence = np.random.choice(len(dicodons), 1, p=normalize(p))[0]
            seq_dict[node.name] = new_sequence
            sequence = dicodons[new_sequence]
            record = SeqRecord(Seq(sequence), id=node.name, description='')
            records.append(record)
    return records


def write_epifel_simulation(input_tree_path, alignment_path, parameters):
    tree = Tree(input_tree_path, format=1)
    records = simulate_epifel(parameters, tree)
    SeqIO.write(records, alignment_path, 'fasta')


def epifel_prune(parameters, tree):
    Q = epistatic_matrix(parameters)
    number_of_sequences = len(tree.get_leaves())
    L = np.zeros((61**2, 2*number_of_sequences-1))
    seq_dict = parameters['seq_dict']
    for node in tree.traverse('postorder'):
        node_index = get_node_index(parameters, node)
        if node.is_leaf():
            L[seq_dict[node.name], node_index] = 1
        else:
            for i, child in enumerate(node.children):
                child_index = get_node_index(parameters, child)
                t = node.dist
                if i == 0:
                    if node.dist < 1e-10:
                        L[:, node_index] = L[:, child_index]
                    else:
                        L[:, node_index] = krylov_subpace_exponential(t*Q, L[:, child_index])
                else:
                    if node.dist < 1e-10:
                        L[:, node_index] *= L[:, child_index]
                    else:
                        L[:, node_index] *= krylov_subpace_exponential(t*Q, L[:, child_index])
        if node.is_root():
            pi = f3x4(parameters)
            l0 = np.dot(np.kron(pi, pi), L[:, node_index])
            return np.sum(np.log(l0))


def build_epifel_likelihood(seq_dict, tree, gtr_result, verbosity=0):
    node_dict = build_node_dict(tree)
    def likelihood(x):
        parameters = gtr_result | {
            'seq_dict': seq_dict,
            'node_dict': node_dict
        }
        parameters['omega'] = x[0]
        parameters['epsilon'] = x[1]
        l = epifel_prune(parameters, tree)
        if verbosity > 0:
            print('Log likelihood:', l)
        if verbosity > 1:
            for k, v in parameters.items():
                print('%s: %s\n' % (str(k), str(v)))
            print('\n----------\n')
        print(x, l)
        return l
    return likelihood


def epifel_lbfgs(tree, seq_dict, gtr_result, niter=100, verbosity=0):
    def callback(x):
        print('Iteration...')
    likelihood = build_epifel_likelihood(seq_dict, tree, gtr_result, verbosity)
    x0 = np.ones(2)
    print('likelihood:', likelihood(x0))
    bounds = len(x0) * [(0, np.inf)]
    result = fmin_l_bfgs_b(minusf(likelihood), x0, approx_grad=True, bounds=bounds, callback=callback)
    return result[0]


def trim_alignment_and_seq_dict(alignment, seq_dict, codon1, codon2):
    alignment = alignment[:, [codon1, codon2]]
    for key, sequence in seq_dict.items():
        sequence = sequence[[codon1, codon2]]
        seq_dict[key] = sequence


def write_fit_epifel(input_alignment_path, input_tree_path, input_gtr_json_path,
        codon1, codon2, output_json_path, method='lbfgs'):
    alignment, seq_dict, tree = read_alignment_and_tree(
        input_alignment_path, input_tree_path
    )
    for value in seq_dict.values():
        value = 61*value[0] + value[1]
    with open(input_gtr_json_path) as json_file:
        gtr_result = json.load(json_file)
    #synchronize_tree_and_results(tree, gtr_result)
    x = epifel_lbfgs(tree, seq_dict, gtr_result)
    node_dict = build_node_dict(tree)
    with open(output_json_path, 'w') as json_file:
        json.dump(list(result), json_file, indent=2)


def epifel_grid(input_alignment_path, input_tree_path, input_gtr_json_path,
        codon1, codon2, output_json_path, method='lbfgs'):
    alignment, seq_dict, tree = read_alignment_and_tree(
        input_alignment_path, input_tree_path
    )
    for value in seq_dict.values():
        value = 61*value[0] + value[1]
    with open(input_gtr_json_path) as json_file:
        gtr_result = json.load(json_file)
    #synchronize_tree_and_results(tree, gtr_result)
    likelihood = build_epifel_likelihood(seq_dict, tree, gtr_result)
    omegas = np.linspace(.1, 3, 10)
    epsilons = np.linspace(.1, 3, 15)
    xv, yv = np.meshgrid(omegas, epsilons)
    gridpoints = list(zip(xv.flatten(), yv.flatten()))
    likelihoods = Parallel(n_jobs=10)(delayed(likelihood)(gp) for gp in gridpoints)
    result = {
        'gridpoints': gridpoints,
        'likelihoods': likelihoods
    }
    with open(output_json_path, 'w') as json_file:
        json.dump(result, json_file, indent=2)


def write_fit_mg94_pair(input_alignment_path, input_tree_path, input_gtr_json_path,
        codon1, codon2, output_json_path, method='lbfgs'):
    alignment, seq_dict, tree = read_alignment_and_tree(
        input_alignment_path, input_tree_path
    )
    codon1 = int(codon1)
    codon2 = int(codon1)
    with open(input_gtr_json_path) as json_file:
        gtr_result = json.load(json_file)
    synchronize_tree_and_results(tree, gtr_result)
    trim_alignment_and_seq_dict(alignment, seq_dict, codon1, codon2)
    x = mg94_lbfgs(alignment, tree, seq_dict, gtr_result, pair=True)
    node_dict = build_node_dict(tree)
    result = mg94_vector2dict(x, node_dict)
    with open(output_json_path, 'w') as json_file:
        json.dump(result, json_file, indent=2)


def count_charge_pairs(input_alignment_path, output_json_path, col1, col2):
    col1 = int(col1)
    col2 = int(col2)
    res1 = Counter()
    res2 = Counter()
    respair = Counter()
    res1charge = Counter()
    res2charge = Counter()
    paircharge = Counter()
    for record in SeqIO.parse(input_alignment_path, 'fasta'):
        seq = record.seq.translate()
        r1 = str(seq[col1])
        r2 = str(seq[col2])
        res1[r1] += 1
        res2[r2] += 1
        respair[r1+r2] += 1
        res1charge[aa_charges[r1]] += 1
        res2charge[aa_charges[r2]] += 1
        paircharge[aa_charges[r1] + aa_charges[r2]] += 1
    with open(output_json_path, 'w') as json_file:
        json.dump({
            'res1': res1,
            'res2': res2,
            'respair': respair,
            'res1charge': res1charge,
            'res2charge': res2charge,
            'paircharge': paircharge
        }, json_file, indent=2)



if __name__ == '__main__':
    alignment_path = 'data/simulate-0/epifel.fasta'
    tree_path = 'data/simulate-0/tree.new'
    gtr_path = 'data/simulate-0/parameters.json'
    epifel_grid(alignment_path, tree_path, gtr_path,
        0, 1, 'result.json', method='lbfgs')
    #count_charge_pairs(alignment_path, 'charge.json', 10, 26)
    #write_fit_mg94_pair(alignment_path, tree_path, gtr_path, 10, 26, 'mg94.json')

    #alignment_path = 'data/simulate-0/gtr.fasta'
    #tree_path = 'data/simulate-0/tree.new'
    #output_json = 'output.json'
    #write_fit_gtr(alignment_path, tree_path, gtr_path)

    #parameters = load_parameters()[0]
    #epistatic_matrix(parameters)
