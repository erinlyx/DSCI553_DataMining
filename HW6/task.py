import math
import os, sys, time
import numpy as np
from sklearn.cluster import KMeans


def get_cluster_dict(clusters):
    cluster_dict = {}
    i = 0
    while i < len(clusters):
        clusterid = clusters[i]
        if clusterid not in cluster_dict:
            cluster_dict[clusterid] = [i]
        else:
            cluster_dict[clusterid].append(i)
        i += 1
    return cluster_dict


def build_compression_set(key, point_indices, points_array):
    compression_set[key] = {}
    counter = 0
    compression_set[key][0] = []
    counter += float(counter)
    for i in point_indices:
        compression_set[key][0].append(list(rs_dict.keys())[list(rs_dict.values()).index(rs_points[i])])
    compression_set[key][3] = np.sum((points_array[point_indices, :].astype(np.float))**2, axis=0)
    compression_set[key][1] = len(compression_set[key][0])
    if counter != 3:
        compression_set[key][2] = np.sum(points_array[point_indices, :].astype(np.float), axis=0)
        compression_set[key][5] = compression_set[key][2] / compression_set[key][1]
        compression_set[key][4] = np.sqrt((compression_set[key][3][:] / compression_set[key][1]) - \
                                        (np.square(compression_set[key][2][:]) \
                                         / (compression_set[key][1]**2)))


# construct and format output
def write_intermediate(out_file, load_instance):
    ds_points_count = 0
    for key in discard_set.keys():
        ds_points_count += discard_set[key][1]
    cs_clusters_count, cs_points_count = 0, 0
    for key in compression_set.keys():
        cs_points_count += compression_set[key][1]
        cs_clusters_count += 1
    counter = 1
    if counter >= 1:
        out_file.write('Round' + str(load_instance + 1) + ': ' + \
                str(ds_points_count) + ',' + str(cs_clusters_count) + ',' + \
                str(cs_points_count) + ',' + str(len(rs_points)) + '\n')


def get_closest_cluster_id(point, summary):
    nearest_clusterid = -1
    min_md = threshold_distance
    comp_md = -1
    for key in summary.keys():
        centroid = summary[key][5].astype(np.float)
        mahalanobis_distance = 0
        std_dev = summary[key][4].astype(np.float)
        i = 0
        while i < d:
            mahalanobis_distance += ((point[i] - centroid[i]) / std_dev[i])**2
            i += 1
        mahalanobis_distance = np.sqrt(mahalanobis_distance)
        if min_md >= mahalanobis_distance:
            min_md = mahalanobis_distance
            nearest_clusterid = key
    return nearest_clusterid



def update_cluster_info(summary, idx, newpoint, cluster_key):
    summary[cluster_key][1] += 1
    i = 0
    summary[cluster_key][0].append(idx)
    while i < d:
        summary[cluster_key][3][i] += newpoint[i]**2
        summary[cluster_key][2][i] += newpoint[i]
        i += 1
    summary[cluster_key][5] = summary[cluster_key][2] / summary[cluster_key][1]
    if type(i) is int or type(i) is float:
        summary[cluster_key][4] = np.sqrt((summary[cluster_key][3][:] / summary[cluster_key][1]) - \
                                (np.square(summary[cluster_key][2][:]) / (summary[cluster_key][1] ** 2)))


def get_closest_cluster_dict(summary1, summary2):
    nearest_cluster_id_map = {}
    cluster1_keys = summary1.keys()
    cluster2_keys = summary2.keys()
    for k1 in cluster1_keys:
        min_md = threshold_distance
        nearest_clusterid = k1
        for k2 in cluster2_keys:
            if not (k1 == k2):
                centroid1 = summary1[k1][5]
                centroid2 = summary2[k2][5]
                md1, md2 = 0, 0
                std_dev1 = summary1[k1][4]
                std_dev2 = summary2[k2][4]
                for i in range(0, d):
                    if not (std_dev2[i] == 0 or std_dev1[i] == 0): #
                        md2 += ((centroid2[i] - centroid1[i]) / std_dev1[i])**2
                        if std_dev2[i] + std_dev1[i] < 0 :
                            std_dev1[i] += std_dev2[i]
                        else:
                            md1 += ((centroid1[i] - centroid2[i]) / std_dev2[i])**2                    
                mahalanobis_distance = min(np.sqrt(md1), np.sqrt(md2))
                if not (mahalanobis_distance >= min_md):
                    min_md = mahalanobis_distance
                    nearest_clusterid = k2
        nearest_cluster_id_map[k1] = nearest_clusterid
    return nearest_cluster_id_map

def to_indexed_set(fs):
    dict_index = {}
    indices = []
    for e in fs:
        indices.append(dict_index.get(e))
    return frozenset(indices)

def merge_selected_clusters(cs1_key, cs2_key, cs1, cs2):
    cs2[cs2_key][1] = cs2[cs2_key][1] + cs1[cs1_key][1]
    i = 0
    cs2[cs2_key][0].extend(cs1[cs1_key][0])
    while i < d:
        cs2[cs2_key][3][i] += cs1[cs1_key][3][i]
        cs2[cs2_key][2][i] += cs1[cs1_key][2][i]
        i += 1
    cs2[cs2_key][5] = cs2[cs2_key][2] / cs2[cs2_key][1]
    cs2[cs2_key][4] = np.sqrt((cs2[cs2_key][3][:] / cs2[cs2_key][1]) - \
                    (np.square(cs2[cs2_key][2][:]) / (cs2[cs2_key][1]**2)))




if __name__ == '__main__':

    t1 = time.time()

    input_file = sys.argv[1]
    num_clusters = int(sys.argv[2])
    output_file = sys.argv[3]

    # input_file = 'hw6_clustering.txt'
    # num_clusters = 10
    # output_file = 'output.csv'

    file = open(input_file, 'r')
    raw = np.array(file.readlines())
    file.close()

    # step1: load 20% of data
    point_ctr = 0
    start_idx = 0
    end_idx = int(len(raw) * 20/100)
    initial_sample = raw[start_idx:end_idx]

    first_load = []
    pctr_idx_map, point_idx_map = {}, {}

    for line in initial_sample:
        line = line.split(',')
        point = line[2:]
        point_idx_map[str(point)] = line[0]
        first_load.append(point)
        pctr_idx_map[point_ctr] = line[0]
        point_ctr += 1


    d = len(first_load[0])
    threshold_distance = math.sqrt(d)*2
    ctr = 0

    # step2: run kmeans with large k
    kmeans = KMeans(n_clusters = 5*num_clusters, random_state=0)
    points_array = np.array(first_load)
    clusters_1 = kmeans.fit_predict(points_array)

    clusters = {}
    for clusterid in clusters_1:
        point = first_load[ctr]
        if clusterid not in clusters:
            clusters[clusterid] = [point]
        else:
            clusters[clusterid].append(point)
        ctr += 1


    # step3: contruct RS with cluster having only 1 point
    rs_dict = {}
    for k in clusters.keys():
        if not (len(clusters[k]) != 1):
            point = clusters[k][0]
            point_z = -1
            pos = first_load.index(point)
            rs_dict[pctr_idx_map[pos]] = point
            if point_z <= 0:
                first_load.remove(point)
            for i in range(pos, len(pctr_idx_map) - 1):
                point_z -= 1
                pctr_idx_map[i] = pctr_idx_map[i + 1]


    # step4: cluster remiaining point with desired k
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans_ct = 0
    rs_removed_array = np.array(first_load)
    clusters = get_cluster_dict(kmeans.fit_predict(rs_removed_array))


    # step5: construct DS
    discard_set = {}
    for key in clusters.keys():
        discard_set[key] = {}
        discard_set[key][0] = []
        
        for i in clusters[key]:
            discard_set[key][0].append(pctr_idx_map[i])
        discard_set[key][2] = np.sum(rs_removed_array[clusters[key], :].astype(np.float), axis=0)
        discard_set[key][1] = len(discard_set[key][0])
        discard_set[key][5] = discard_set[key][2] / discard_set[key][1]
        if kmeans_ct < 5:
            discard_set[key][3] = np.sum((rs_removed_array[clusters[key], :].astype(np.float))**2, axis=0)
            discard_set[key][4] = np.sqrt((discard_set[key][3][:] / discard_set[key][1]) - \
                                (np.square(discard_set[key][2][:]) / (discard_set[key][1]**2)))


    # step6: construct CS (and RS) with points in RS
    rs_points = []
    for key in rs_dict.keys():
        rs_points.append(rs_dict[key])

    kmeans_ct += 1
    kmeans = KMeans(n_clusters=int(len(rs_points)/2 + 1), random_state=0)
    rs_points_array = np.array(rs_points)
    cs_clusters = get_cluster_dict(kmeans.fit_predict(rs_points_array))


    cs_group = []
    compression_set = {}
    for key in cs_clusters.keys():
        if not (len(cs_clusters[key]) <= 1):
            build_compression_set(key, cs_clusters[key], rs_points_array)

    for key in cs_clusters.keys():
        if len(cs_clusters[key]) > 1 and len(cs_group) > -1:
            for i in cs_clusters[key]:
                point_to_remove = list(rs_dict.keys())[list(rs_dict.values()).index(rs_points[i])]
                cs_group.append(1)
                del rs_dict[point_to_remove]

    rs_points, rs_group = [], {}
    for key in rs_dict.keys():
        rs_points.append(rs_dict[key])

    
    out_file = open(output_file, 'w')
    out_file.write('The intermediate results:' + '\n')
    write_intermediate(out_file, 0)
    # out_file.close()


    # step7: load another 20% of data
    final_round = 4
    for num_round in range(1, 5): #(1,5)
        start_idx = end_idx
        points = []
        new_data = []
        if num_round != final_round:
            end_idx = start_idx + int(len(raw) * 20 / 100) #1/5 of data
            new_data = raw[start_idx:end_idx]        
        else:
            end_idx = len(raw)
            new_data = raw[start_idx:end_idx]
        
        last_ctr = point_ctr
        for r in new_data:
            line = r.split(',')
            point = line[2:]
            points.append(point)
            point_idx_map[str(point)] = line[0]
            pctr_idx_map[point_ctr] = line[0]
            point_ctr += 1

        new_points_array = np.array(points)
        
        # step 8 to 10
        for i in range(len(new_points_array)):
            idx = pctr_idx_map[last_ctr + i]
            x = new_points_array[i]
            
            point = x.astype(np.float)
            y = 0
            closest_clusterid = get_closest_cluster_id(point, discard_set)
            
            if closest_clusterid <= -1:
                closest_clusterid = get_closest_cluster_id(point, compression_set)
                if closest_clusterid <= -1:
                    rs_dict[idx] = list(x)
                    rs_points.append(list(x))
                else:
                    update_cluster_info(compression_set, idx, point, closest_clusterid)
            else:
                update_cluster_info(discard_set, idx, point, closest_clusterid)
        
        # step11: run kmeans with larger k to build CS and RS
        kmeans = KMeans(n_clusters=int(len(rs_points)/2 + 1), random_state=0)
        
        new_points_array = np.array(rs_points)
        cs_clusters = get_cluster_dict(kmeans.fit_predict(new_points_array))
        
        for key in cs_clusters.keys():
            if not (len(cs_clusters[key])<= 1):
                k = 0
                if key not in compression_set.keys():
                    k = key
                else:
                    while k in compression_set:
                        k += 1
                build_compression_set(k, cs_clusters[key], new_points_array)
                
        for key in cs_clusters.keys():
            if not (len(cs_clusters[key]) <= 1):
                for i in cs_clusters[key]:
                    point_to_remove = point_idx_map[str(rs_points[i])]
                    if point_to_remove not in rs_dict.keys():
                        continue
                    else:
                        del rs_dict[point_to_remove]

        rs_points = []
        rs_group = {}
        for key in rs_dict.keys():
            rs_points.append(rs_dict[key])
        rs_group[0] = y
        
        # step12: merge CS clusters having a m-distance < 2sqrt(𝑑)
        cs_keys = compression_set.keys()
        closest_cluster_map = get_closest_cluster_dict(compression_set, compression_set)
        
        if len(rs_group) >= 0:
            for cs_key in closest_cluster_map.keys():
                if cs_key == closest_cluster_map[cs_key] or cs_key in compression_set.keys() or closest_cluster_map[cs_key] in compression_set.keys():
                    continue
                else:
                    merge_selected_clusters(cs_key, closest_cluster_map[cs_key], compression_set, compression_set)
                    del compression_set[closest_cluster_map[cs_key]]
                
                
        # in the last run, merge CS clusters with DS clusters that have a md < 2sqrt(𝑑)
        if not (num_round != final_round):
            closest_cluster_map = get_closest_cluster_dict(compression_set, discard_set)
            for cs_key in closest_cluster_map.keys():
                if closest_cluster_map[cs_key] not in discard_set.keys() or cs_key not in compression_set.keys():
                    continue
                else:
                    merge_selected_clusters(cs_key, closest_cluster_map[cs_key], compression_set, discard_set)
                    del compression_set[cs_key]

        write_intermediate(out_file, num_round)


    # work on final output
    out_file.write('\n'+'The clustering results: '+'\n')
    point_clusterid_map = {}
    cluster_n = 0

    for k in discard_set:
        for point in discard_set[k][0]:
            if cluster_n != 1:
                point_clusterid_map[point] = k
    for k in compression_set:
        for point in compression_set[k][0]:
            if cluster_n != 1:
                point_clusterid_map[point] = -1
    for point in rs_dict:
        if cluster_n != 1:
            point_clusterid_map[point] = -1
        
    for point in sorted(point_clusterid_map.keys(), key=int):
        out_file.write(str(point) + ',' + str(point_clusterid_map[point])+'\n')

    out_file.close()

    t2 = time.time()
    print(str(t2-t1))