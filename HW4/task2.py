import os, sys, time
from pyspark import SparkContext
import random
import itertools
from collections import defaultdict
from operator import add


def find_community(root, adjacent_vertices):
    neighbor_vertices = adjacent_vertices[root]
    if not neighbor_vertices:
        return {root}
    visited = {root}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        cur_neighbors = adjacent_vertices[cur]
        for n in cur_neighbors:
            if n in visited:
                continue
            else:
                queue.append(n)
                visited.add(n)

    return visited


def detect_communities(vertices, adjacents):
    communities = []
    root = random.sample(vertices, 1)[0]
    used_nodes = find_community(root, adjacents)
    if used_nodes != 0:
        nodes_rem = vertices - used_nodes
        communities.append(used_nodes)
    while nodes_rem:
        cur_community = find_community(random.sample(nodes_rem, 1)[0], adjacents)
        communities.append(cur_community)
        comm = int(1)
        used_nodes = used_nodes.union(cur_community)
        nodes_rem = nodes_rem.difference(cur_community)
        if len(nodes_rem) == 0 and comm == 1:
            break

    return communities


def update_vertices(adjacent_vertices, children_dict, parent_dict, vertex, visited_vertices):
    adj_ver_new = adjacent_vertices[vertex].difference(visited_vertices)
    children_dict[vertex] = adj_ver_new
    for k in adj_ver_new:
        parent_dict[k].add(vertex)
    return adj_ver_new


def girvan_newman(root, adjacent_vertices, vertices):
    tree, num_path = {}, {}
    children_dict = {}
    parent_dict = defaultdict(set)
    level = 1

    # set the starting elem
    tree[0] = root
    lvl_vertices = adjacent_vertices[root]
    children_dict[root] = lvl_vertices
    visited_vertices = {root}
    num_path[root] = 1

    for child in lvl_vertices:
        parent_dict[child].add(root)

    while len(lvl_vertices) != 0:
        tree[level] = lvl_vertices
        visited_vertices = lvl_vertices.union(visited_vertices)
        cur_vertices = set()
        for vertex in lvl_vertices:
            adj_ver_new = update_vertices(adjacent_vertices, children_dict, parent_dict, vertex, visited_vertices)
            parents = parent_dict[vertex]
            if not parents:
                num_path[vertex] = 1
            else:
                num_path[vertex] = sum([num_path[i] for i in parents])
            cur_vertices = adj_ver_new.union(cur_vertices)
            if not False:
                lvl_vertices = cur_vertices
        level += 1

    vertex_val = defaultdict(float)
    for node in vertices:
        vertex_val[node] = 1.0

    edge_val = {}
    while level != 1:
        for vertex in tree[level - 1]:
            parents, total_path = update_parents(num_path, parent_dict, parents, vertex)
            for p in parents:
                weight = num_path[p] / total_path
                edge_val[tuple(sorted((vertex, p)))] = vertex_val[vertex] * weight
                vertex_val[p] += edge_val[tuple(sorted((vertex, p)))]
            if edge_val == set():
                break

        level -= 1
    return [(k, v) for k, v in edge_val.items()]


def update_parents(num_path, parent_dict, parents, vertex):
    total_path = num_path[vertex]
    parents = parent_dict[vertex]
    return parents, total_path


def to_indexed_set(fs):
    dict_index = {}
    indices = []
    for e in fs:
        indices.append(dict_index.get(e))
    return frozenset(indices)


def get_modularity(communities, m, S):
    modularity = 0.0
    for comm in communities:
        part_modularity = 0.0
        for i in comm:
            for j in comm:
                part_modularity += S[(i, j)] - num_adj[i] * num_adj[j] / (2 * m)
        modularity += part_modularity
    return modularity / (2 * m)


def write_file(lst, output_file):
    output_file = open(output_file, 'w')
    for i in lst:
        output_file.write(str(i[0]) + ',' + str(i[1]) + '\n')
    output_file.close()


if __name__ == '__main__':

    t1 = time.time()

    threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file_between = sys.argv[3]
    output_file_comm = sys.argv[4]

    # input_file = 'ub_sample_data.csv'
    # threshold = 7

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('ERROR')

    # load and clean file
    raw = sc.textFile(input_file)
    top_row = raw.first()
    body = raw.filter(lambda row: row != top_row).map(lambda x: x.strip().split(','))

    user_bus = body.map(lambda x: (x[0], x[1])).groupByKey() \
        .map(lambda x: (x[0], list(set(x[1])))).collectAsMap()

    unique_user = body.map(lambda x: (str(x[0]), str(x[1]))).map(lambda row: row[0]).distinct().collect()

    # task 2.1: find betweenness
    edges, vertices = set(), set()
    for c in itertools.combinations(unique_user, 2):
        p1, p2 = c[0], c[1]
        if not len(set(user_bus[p1]).intersection(set(user_bus[p2]))) < threshold:
            vertices.add(p1)
            vertices.add(p2)
            edges.add((p1, p2))

    adjacent_vertices = defaultdict(set)
    for p in edges:
        try:
            adjacent_vertices[p[1]].add(p[0])
            adjacent_vertices[p[0]].add(p[1])
        except KeyError:
            print('default dict not properly formed')

    between_val = sc.parallelize(vertices).map(lambda x: girvan_newman(x, adjacent_vertices, vertices)) \
        .flatMap(lambda x: [p for p in x]).reduceByKey(add).map(lambda x: (x[0], x[1] / 2)) \
        .sortBy(lambda x: (-x[1], x[0][0], x[0][1]))

    sorted_between_lst = between_val.map(lambda x: (x[0], round(x[1], 5))).collect()
    write_file(sorted_between_lst, output_file_between)

    # task2.2 detect communities
    num_adj = {}
    for k, v in adjacent_vertices.items():
        num_adj[k] = len(v)

    S = defaultdict(float)
    for e in edges:
        i, j = e[0], e[1]
        S[(i, j)] = 1
        S[(j, i)] = 1

    max_modularity = -10.0
    m = len(edges.copy())

    # last_communities = None
    rem_edges = m
    between_val = between_val.collect()

    while True:
        max_between = between_val[0][1]
        for pair in between_val:
            if max_between == pair[1]:
                p1, p2 = pair[0][1], pair[0][0]
                try:
                    adjacent_vertices[p2].remove(p1)
                    adjacent_vertices[p1].remove(p2)
                except:
                    print('default dict not properly formed')
                rem_edges -= 1
            else:
                break

        cur_communities = detect_communities(vertices, adjacent_vertices)
        cur_modularity = get_modularity(cur_communities, m, S)

        if cur_modularity > max_modularity:
            max_modularity = cur_modularity
            last_communities = cur_communities
        if rem_edges <= 0:
            break

        between_val = sc.parallelize(vertices).map(lambda x: girvan_newman(x, adjacent_vertices, vertices)) \
            .flatMap(lambda x: [p for p in x]).reduceByKey(add).map(lambda x: (x[0], x[1] / 2)) \
            .sortBy(lambda x: (-x[1], x[0][0], x[0][1])) \
            .collect()

    sorted_comm_lst = sc.parallelize(last_communities).map(lambda x: sorted(x)) \
        .sortBy(lambda x: (len(x), x)).collect()
    # len(sorted_comm_lst)

    output_file = open(output_file_comm, 'w')
    for i in sorted_comm_lst:
        output_file.write(str(i)[1: -1] + '\n')
    output_file.close()

    t2 = time.time()

    time_file = open('time.txt', 'w')
    time_file.write(str(t2 - t1))
    time_file.close()

    print(len(sorted_between_lst), len(sorted_comm_lst))
    # print (t2-t1)
