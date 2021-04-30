import numpy as np
import pandas as pd


def multi_value_binary_search(chr_gff: np.ndarray,
                              low_value: int,
                              high_value: int,
                              current_index: int = -1):
    """
    二分查找， 给定（low，high）二值， 在列表（starts, ends）中查找。
    例如：
        # 100 200
        # 200 300
        # 400 500
        # 900 1000

        查找，（120，150）是否包含在上述的列表中，返回 > 0的值，
        查找，（700，800）是否包含在上述的列表中，返回 = 0的值，

    :param start_values:
    :param end_values:
    :param low_value:
    :param high_value:
    :return:
    """

    ##################################################################################
    #   场景1： low ------------- high
    #   场景2：                                                  low ---------high
    #   场景3：                          low ------- high
    #                  start ---------------------------------------- end
    #
    ##################################################################################
    result = -1

    start_values = chr_gff[0]
    end_values = chr_gff[1]

    # print("--current_index: ", current_index)
    if current_index == -1:  # 需要重新开始搜索
        # 场景1：
        low = 0
        high = len(start_values)
        middle = 0
        while low < high:
            middle = int((low + high) / 2)
            middle_value = start_values[middle]
            if middle_value < low_value:
                low = middle + 1
            elif middle_value > high_value:
                high = middle - 1
            else:
                result = middle
                return result

        # 场景 2
        low = 0
        high = len(end_values)
        middle = 0
        while low < high:
            middle = int((low + high) / 2)
            middle_value = end_values[middle]
            if middle_value < low_value:
                low = middle + 1
            elif middle_value > high_value:
                high = middle - 1
            else:
                result = middle
                return result

        # 场景 3
        value = (high_value + low_value) / 2
        low = 0
        high = len(end_values) - 1
        middle = 0
        while low <= high:
            middle = int((low + high) / 2)

            print(start_values[middle], end_values[middle], low_value, high_value, low, middle, high)
            print("\t", start_values[low:high])
            print("\t", end_values[low:high])
            if high_value < start_values[middle]:
                high = middle - 1
                print("<-")
                # print("___1___")
            elif low_value > end_values[middle]:
                low = middle + 1
                # print("___2___")
                print("->")
            #else:

            if start_values[low] <= low_value <= end_values[low] or \
                    start_values[low] <= high_value <= end_values[low] or \
                    low_value <= start_values[low] <= high_value or low_value <= end_values[low] <= high_value:
                result = low
                return result

            if start_values[middle] <= low_value <= end_values[middle] or \
                    start_values[middle] <= high_value <= end_values[middle] or \
                    low_value <= start_values[middle] <= high_value or low_value <= end_values[middle] <= high_value:
                result = middle
                return result

            if start_values[high] <= low_value <= end_values[high] or \
                    start_values[high] <= high_value <= end_values[high] or \
                    low_value <= start_values[high] <= high_value or low_value <= end_values[high] <= high_value:
                result = high
                return result

            print("\t", start_values[low], end_values[low], low_value, high_value)
            print("\t", start_values[middle], end_values[middle], low_value, high_value)
            print("\t", start_values[high], end_values[high], low_value, high_value)
            print("___3___")
    else:
        # 场景1：
        low = current_index - 1
        high = len(start_values)
        middle = 0
        while low < high:
            middle = int((low + high) / 2)
            middle_value = start_values[middle]
            if middle_value < low_value:
                low = middle + 1
            elif middle_value > high_value:
                high = middle - 1
            else:
                result = middle
                return result

        # 场景 2
        low = current_index - 1
        high = len(end_values)
        middle = 0
        while low < high:
            middle = int((low + high) / 2)
            middle_value = end_values[middle]
            if middle_value < low_value:
                low = middle + 1
            elif middle_value > high_value:
                high = middle - 1
            else:
                result = middle
                return result

        # 场景 3
        value = (high_value + low_value) / 2
        low = current_index - 1
        high = len(end_values) - 1
        middle = 0
        while low < high:
            middle = int((low + high) / 2)

            # print('-', start_values[middle], end_values[middle], low_value, high_value)

            if high_value < start_values[middle]:
                high = middle - 1
            elif low_value > end_values[middle]:
                low = middle + 1

            if start_values[low] <= low_value <= end_values[low] or start_values[low] <= high_value <= end_values[low]:
                result = low
                return result

            if start_values[middle] <= low_value <= end_values[middle] or start_values[middle] <= high_value <= end_values[middle]:
                result = middle
                return result

            if start_values[high] <= low_value <= end_values[high] or start_values[high] <= high_value <= end_values[high]:
                result = high
                return result

    return result


def multi_value_range_search(chr_gff: np.ndarray,
                             low_value: int,
                             high_value: int,
                             current_index: int = -1):
    """
    二分查找， 给定（low，high）二值， 在列表（starts, ends）中查找。
    例如：
        # 100 200
        # 200 300
        # 400 500
        # 900 1000

        查找，（120，150）是否包含在上述的列表中，返回 > 0的值，
        查找，（700，800）是否包含在上述的列表中，返回 = 0的值，

    :param start_values:
    :param end_values:
    :param low_value:
    :param high_value:
    :return:
    """

    ##################################################################################
    #   场景1： low ------------- high
    #   场景2：                                                  low ---------high
    #   场景3：                          low ------- high
    #                  start ---------------------------------------- end
    #
    ##################################################################################
    result = -1

    start_values = chr_gff[0]
    end_values = chr_gff[1]

    low_starts = start_values - low_value
    low_starts = low_starts + (end_values - start_values)



    # # print("--current_index: ", current_index)
    # if current_index == -1:  # 需要重新开始搜索
    #     # 场景1：
    #     low = 0
    #     high = len(start_values)
    #     middle = 0
    #     while low < high:
    #         middle = int((low + high) / 2)
    #         middle_value = start_values[middle]
    #         if middle_value < low_value:
    #             low = middle + 1
    #         elif middle_value > high_value:
    #             high = middle - 1
    #         else:
    #             result = middle
    #             return result
    #
    #     # 场景 2
    #     low = 0
    #     high = len(end_values)
    #     middle = 0
    #     while low < high:
    #         middle = int((low + high) / 2)
    #         middle_value = end_values[middle]
    #         if middle_value < low_value:
    #             low = middle + 1
    #         elif middle_value > high_value:
    #             high = middle - 1
    #         else:
    #             result = middle
    #             return result
    #
    #     # 场景 3
    #     value = (high_value + low_value) / 2
    #     low = 0
    #     high = len(end_values) - 1
    #     middle = 0
    #     while low <= high:
    #         middle = int((low + high) / 2)
    #
    #         print(start_values[middle], end_values[middle], low_value, high_value, low, middle, high)
    #         print("\t", start_values[low:high])
    #         print("\t", end_values[low:high])
    #         if high_value < start_values[middle]:
    #             high = middle - 1
    #             print("<-")
    #             # print("___1___")
    #         elif low_value > end_values[middle]:
    #             low = middle + 1
    #             # print("___2___")
    #             print("->")
    #         #else:
    #
    #         if start_values[low] <= low_value <= end_values[low] or \
    #                 start_values[low] <= high_value <= end_values[low] or \
    #                 low_value <= start_values[low] <= high_value or low_value <= end_values[low] <= high_value:
    #             result = low
    #             return result
    #
    #         if start_values[middle] <= low_value <= end_values[middle] or \
    #                 start_values[middle] <= high_value <= end_values[middle] or \
    #                 low_value <= start_values[middle] <= high_value or low_value <= end_values[middle] <= high_value:
    #             result = middle
    #             return result
    #
    #         if start_values[high] <= low_value <= end_values[high] or \
    #                 start_values[high] <= high_value <= end_values[high] or \
    #                 low_value <= start_values[high] <= high_value or low_value <= end_values[high] <= high_value:
    #             result = high
    #             return result
    #
    #         print("\t", start_values[low], end_values[low], low_value, high_value)
    #         print("\t", start_values[middle], end_values[middle], low_value, high_value)
    #         print("\t", start_values[high], end_values[high], low_value, high_value)
    #         print("___3___")
    # else:
    #     # 场景1：
    #     low = current_index - 1
    #     high = len(start_values)
    #     middle = 0
    #     while low < high:
    #         middle = int((low + high) / 2)
    #         middle_value = start_values[middle]
    #         if middle_value < low_value:
    #             low = middle + 1
    #         elif middle_value > high_value:
    #             high = middle - 1
    #         else:
    #             result = middle
    #             return result
    #
    #     # 场景 2
    #     low = current_index - 1
    #     high = len(end_values)
    #     middle = 0
    #     while low < high:
    #         middle = int((low + high) / 2)
    #         middle_value = end_values[middle]
    #         if middle_value < low_value:
    #             low = middle + 1
    #         elif middle_value > high_value:
    #             high = middle - 1
    #         else:
    #             result = middle
    #             return result
    #
    #     # 场景 3
    #     value = (high_value + low_value) / 2
    #     low = current_index - 1
    #     high = len(end_values) - 1
    #     middle = 0
    #     while low < high:
    #         middle = int((low + high) / 2)
    #
    #         # print('-', start_values[middle], end_values[middle], low_value, high_value)
    #
    #         if high_value < start_values[middle]:
    #             high = middle - 1
    #         elif low_value > end_values[middle]:
    #             low = middle + 1
    #
    #         if start_values[low] <= low_value <= end_values[low] or start_values[low] <= high_value <= end_values[low]:
    #             result = low
    #             return result
    #
    #         if start_values[middle] <= low_value <= end_values[middle] or start_values[middle] <= high_value <= end_values[middle]:
    #             result = middle
    #             return result
    #
    #         if start_values[high] <= low_value <= end_values[high] or start_values[high] <= high_value <= end_values[high]:
    #             result = high
    #             return result

    return result


def get_refseq_gff(gff_file: str, include_types: list):
    """
    读取Genebank注释文件
    :param gff_file:
    :param include_types:
    :return:
    """

    # include_types = ['direct_repeat',
    #                  'rRNA',
    #                  'region',
    #                  'regulatory_region',
    #                  'CDS',
    #                  'origin_of_replication',
    #                  'recombination_feature',
    #                  'replication_regulatory_region',
    #                  'D_loop', 'primary_transcript',
    #                  'dispersed_repeat',
    #                  'pseudogene',
    #                  'guide_RNA',
    #                  'GC_rich_promoter_region',
    #                  'nucleotide_motif',
    #                  'matrix_attachment_site',
    #                  'snRNA',
    #                  'RNase_P_RNA',
    #                  'repeat_instability_region',
    #                  'response_element',
    #                  'non_allelic_homologous_recombination_region',
    #                  'exon',
    #                  'replication_start_site',
    #                  'insulator',
    #                  'miRNA',
    #                  'microsatellite',
    #                  'vault_RNA',
    #                  'promoter',
    #                  'sequence_feature',
    #                  'gene',
    #                  'mRNA',
    #                  'tRNA',
    #                  'C_gene_segment',
    #                  'conserved_region',
    #                  'minisatellite',
    #                  'biological_region',
    #                  'snoRNA',
    #                  'locus_control_region',
    #                  'CAGE_cluster',
    #                  'mitotic_recombination_region',
    #                  'scRNA',
    #                  'sequence_secondary_structure',
    #                  'epigenetically_modified_region',
    #                  'chromosome_breakpoint',
    #                  'RNase_MRP_RNA',
    #                  'transcript',
    #                  'CAAT_signal',
    #                  'TATA_box',
    #                  'telomerase_RNA',
    #                  'sequence_alteration',
    #                  'tandem_repeat',
    #                  'J_gene_segment',
    #                  'meiotic_recombination_region',
    #                  'transcriptional_cis_regulatory_region',
    #                  'antisense_RNA',
    #                  'sequence_comparison',
    #                  'stem_loop', 'silencer',
    #                  'D_gene_segment',
    #                  'DNAseI_hypersensitive_site',
    #                  'match',
    #                  'V_gene_segment',
    #                  'mobile_genetic_element',
    #                  'lnc_RNA',
    #                  'cDNA_match',
    #                  'imprinting_control_region',
    #                  'protein_binding_site',
    #                  'nucleotide_cleavage_site',
    #                  'Y_RNA',
    #                  'enhancer',
    #                  'repeat_region',
    #                  'enhancer_blocking_element']

    # 染色体
    chr_dict = {"NC_000001": "chr1",
                "NC_000002": "chr2",
                "NC_000003": "chr3",
                "NC_000004": "chr4",
                "NC_000005": "chr5",
                "NC_000006": "chr6",
                "NC_000007": "chr7",
                "NC_000008": "chr8",
                "NC_000009": "chr9",
                "NC_000010": "chr10",
                "NC_000011": "chr11",
                "NC_000012": "chr12",
                "NC_000013": "chr13",
                "NC_000014": "chr14",
                "NC_000015": "chr15",
                "NC_000016": "chr16",
                "NC_000017": "chr17",
                "NC_000018": "chr18",
                "NC_000019": "chr19",
                "NC_000020": "chr20",
                "NC_000021": "chr21",
                "NC_000022": "chr22",
                "NC_000023": "chrX",
                "NC_000024": "chrY"}

    index = 0
    type_set = set()

    chr_gff_dict = {}
    chr_gff_dict_2 = {}

    record = 0
    with open(gff_file, mode='r', encoding='utf-8') as f:
        for line in f:
            # 过滤掉非 'NC_' 开头的数据
            if line.startswith('NC_') is False:
                continue
            tokens = line.split('	')  # \t

            if len(tokens) < 5:
                continue

            anno_type = tokens[2]
            type_set.add(anno_type)
            if anno_type in include_types:
                # 染色体编号
                chr = tokens[0] #.split('.')[0]
                #chr = chr_dict.get(chr, '')

                if len(chr) > 0:
                    # print(chr, tokens[2], tokens[3], tokens[4])
                    record += 1
                    start = int(tokens[3])
                    end = int(tokens[4])

                    if chr in chr_gff_dict.keys():
                        chr_gff_dict[chr]['start'].append(start)
                        chr_gff_dict[chr]['end'].append(end)
                        chr_gff_dict[chr]['type'].append(anno_type)

                        chr_gff_dict_2[chr].append(np.array([start, end, anno_type]))
                    else:
                        start_list = [start]
                        end_list = [end]
                        type_list = [anno_type]
                        chr_gff_dict[chr] = {}
                        chr_gff_dict[chr]['start'] = start_list
                        chr_gff_dict[chr]['end'] = end_list
                        chr_gff_dict[chr]['type'] = type_list

                        chr_gff_dict_2[chr] = []
                        chr_gff_dict_2[chr].append(np.array([start, end, anno_type]))

            index += 1
    chr_gff_dict_3 = {}

    for k, v in chr_gff_dict_2.items():
        value = np.array(v)
        df = pd.DataFrame(value)
        df.columns = ['start', 'end', 'type']
        df['start'] = pd.to_numeric(df['start'])
        df['end'] = pd.to_numeric(df['end'])
        df.sort_values(by=['start','end'], ascending=[True, True], inplace=True)
        df = df.values
        data = np.vstack([np.array(df[:, 0], dtype=np.int), np.array(df[:, 1], dtype=np.int), df[:, 2]])
        chr_gff_dict_3[k] = data

    return chr_gff_dict_3


def get_gff_array(chr_gff_dict: dict, chr: str, start: int, end: int, position: int, pool_size=8):
    anno_type_dict = {}

    results = []
    current_index = -1

    feature = 0

    chr_gff = chr_gff_dict.get(chr, None)
    if chr_gff is None or len(chr_gff) != 3:
        return results

    starts = chr_gff[0]
    ends = chr_gff[1]
    anno_types = chr_gff[2]

    #print("__1.1.1__")
    anno_index = multi_value_binary_search(chr_gff, start, end, current_index=-1)
    #print("__1.1.2__")
    if anno_index > -1:
        real_start = start
        real_end = end

        if starts[anno_index] > start:
            real_start = starts[anno_index]
        if ends[anno_index] < end:
            real_end = ends[anno_index]

        real_start = real_start - start
        real_end = real_end - start

        results.append([anno_types[anno_index], real_start, real_end])

        # 往下检索
        index = anno_index + 1
        while True:
            value = (start + end) / 2
            if index >= len(starts):
                break

            if start <= starts[index] <= end:
                real_start = start
                real_end = end
                if starts[index] > start:
                    real_start = starts[index]
                if ends[index] < end:
                    real_end = ends[index]
                real_start = real_start - start
                real_end = real_end - start
                results.append([anno_types[index], real_start, real_end])
                index += 1
            elif start <= ends[index] <= end:
                real_start = start
                real_end = end
                if starts[index] > start:
                    real_start = starts[index]
                if ends[index] < end:
                    real_end = ends[index]
                real_start = real_start - start
                real_end = real_end - start
                results.append([anno_types[index], real_start, real_end])
                index += 1
            elif starts[index] <= value <= ends[index]:
                real_start = start
                real_end = end
                if starts[index] > start:
                    real_start = starts[index]
                if ends[index] < end:
                    real_end = ends[index]
                real_start = real_start - start
                real_end = real_end - start
                results.append([anno_types[index], real_start, real_end])
                index += 1
            else:
                break

        # 往上检索
        index = anno_index - 1
        while True:
            value = (start + end) / 2
            if index < 0:
                break

            if start <= starts[index] <= end:
                real_start = start
                real_end = end
                if starts[index] > start:
                    real_start = starts[index]
                if ends[index] < end:
                    real_end = ends[index]
                real_start = real_start - start
                real_end = real_end - start
                results.append([anno_types[index], real_start, real_end])
                index -= 1
            elif start <= ends[index] <= end:
                real_start = start
                real_end = end
                if starts[index] > start:
                    real_start = starts[index]
                if ends[index] < end:
                    real_end = ends[index]
                real_start = real_start - start
                real_end = real_end - start
                results.append([anno_types[index], real_start, real_end])
                index -= 1
            elif starts[index] <= value <= ends[index]:
                real_start = start
                real_end = end
                if starts[index] > start:
                    real_start = starts[index]
                if ends[index] < end:
                    real_end = ends[index]
                real_start = real_start - start
                real_end = real_end - start
                results.append([anno_types[index], real_start, real_end])
                index -= 1
            else:
                break

        return results
    else:
        return results


def get_gene_feature_array(chr_gff_dict: dict, chr: str, start: int, end: int):

    chr_gff = chr_gff_dict.get(chr, None)
    if chr_gff is None or len(chr_gff) != 3:
        return None

    starts = chr_gff[0]
    ends = chr_gff[1]
    anno_types = chr_gff[2]

    data = get_gene_features(starts, ends, anno_types, start, end)

    # print("Data: ", data.head(100))

    results = []
    if data is None or len(data) == 0:
        return results

    for index, row in data.iterrows():
        real_start = start
        real_end = end

        if row['start'] > start:
            real_start = row['start']
        if row['end'] < end:
            real_end = row['end']

        real_start = real_start - start
        real_end = real_end - start
        results.append([row['annotation'], real_start, real_end])

    return results


def get_gene_features(starts, ends, annotations, low_value, high_value, binary_search = True):
    """

    :param starts:
    :param ends:
    :param low_value:
    :param high_value:
    :return:
    """
    ##############################################################################
    # 场景1
    #        50+------+ 110
    # 场景2
    #                      120+----------+160
    # 场景3
    #                                       180+---------------+240
    # 场景4
    #        50+--+90
    # 场景5
    #                                                      230+-------------+300
    # 场景6   50+-------------------------------------------------------------+300
    #
    #               +---------------------------------+
    #               100                               200
    ##############################################################################

    # 需要排除掉一部分
    start_index = 0
    end_index = len(starts)

    start_index = 0
    middle_index = 0
    end_index = len(ends)
    in_range = False

    if binary_search is True:
        low = 0
        high = len(starts)
        middle = 0
        while low < high:
            middle = int((low + high) / 2)
            middle_start_value = starts[middle]
            middle_end_value = ends[middle]

            low_start = low_value - middle_start_value
            low_distance = low_start + (high_value - low_value)
            high_distance = low_value - middle_end_value

            # print("Position: ", low, middle, high)
            # print(middle_start_value, middle_end_value, low_value, high_value)
            # print("<-", low_start, low_distance, high_distance)
            # print("Middle: ", middle)

            if (low_start < 0 and low_distance < 0):
                # print("<-")
                high = middle - 1
            elif (low_start > 0 and high_distance > 0):
                # print("->")
                low = middle + 1
            elif (low_start <= 0 and low_distance >= 0) or (low_start > 0 and high_distance <= 0):
                # print("=")
                high = middle - 1
                middle_index = middle
                in_range = True
                break
            else:
                # print("---")
                break

            # print()
        # start_index = low
        #
        # if start_index > 0:
        #     start_index = start_index - 1
        #
        # start_index = max(start_index - 200, 0)
        # # print("===============", start_index)
        #
        # low = 0
        # high = len(starts)
        # middle = 0
        # while low < high:
        #     middle = int((low + high) / 2)
        #     middle_start_value = starts[middle]
        #     middle_end_value = ends[middle]
        #
        #     low_start = low_value - middle_start_value
        #     low_distance = low_start + (high_value - low_value)
        #     high_distance = low_value - middle_end_value
        #
        #     # print("Position: ", low, middle, high)
        #     # print(middle_start_value, middle_end_value, low_value, high_value)
        #     # print("->", low_start, low_distance, high_distance)
        #
        #     # print("Middle: ", middle)
        #     if (low_start < 0 and low_distance < 0):
        #         # print("-")
        #         high = middle - 1
        #     elif (low_start > 0 and high_distance > 0):
        #         # print("--")
        #         low = middle + 1
        #     elif (low_start <= 0 and low_distance >= 0) or (low_start > 0 and high_distance <= 0):
        #         # print("---")
        #         low = middle + 1
        #     else:
        #         # print("----")
        #         break
        #     # print()
        # end_index = high

        #if end_index < len(ends) - 1:
        #end_index = min(end_index + 200, len(ends))
    if in_range is True:
        start_index = max(middle_index - 200, 0)
        end_index = min(middle_index + 200, len(ends))
    elif middle > 0:
        start_index = max(middle - 200, 0)
        end_index = min(middle + 200, len(ends))
    else:
        return None

    # print("start_index: ", start_index)
    # print("end_index: ", end_index)

    low_starts = low_value - starts[start_index:end_index]
    low_distance = low_starts + (high_value - low_value)
    high_distance = low_value - ends[start_index:end_index]

    # print("low start: ", low_starts)
    # print("low distance: ", low_distance)
    # print("high distance: ", high_distance)

    df = pd.DataFrame()
    df['start'] = starts[start_index:end_index]
    df['end'] = ends[start_index:end_index]
    df['annotation'] = annotations[start_index:end_index]
    df['low_start'] = low_starts
    df['low_distance'] = low_distance
    df['high_distance'] = high_distance

    # print(df.head(100))
    # 过滤条件
    return df[((df['low_start'] <= 0) & (df['low_distance'] >= 0)) | ((df['low_start'] > 0) & (df['high_distance'] <= 0))]



if __name__ == '__main__':

    # starts = np.array([0, 100, 210, 300, 400])
    # ends = np.array([0, 200, 250, 350, 450])
    #
    # low_value = 40
    # high_value = 490
    #
    #
    #
    #
    # def get_gene_features(starts, ends, low_value, high_value):
    #
    #     # 场景 1、4、6
    #     low_starts = low_value - starts
    #     low_distance = low_starts + (high_value - low_value)
    #     high_distance = low_value - ends
    #
    #     print("low start: ", low_starts)
    #     print("low distance: ", low_distance)
    #     print("high distance: ", high_distance)
    #
    #     df = pd.DataFrame()
    #     df['start'] = starts
    #     df['end'] = ends
    #     df['low_start'] = low_starts
    #     df['low_distance'] = low_distance
    #     df['high_distance'] = high_distance
    #
    #     # print(df.head(10))
    #
    #     # 场景 1、4、6 过滤条件
    #     return df[((df['low_start'] <= 0) & (df['low_distance'] >= 0)) | ((df['low_start'] > 0) & (df['high_distance'] <= 0))]
    #
    #
    # print(starts)
    # print(ends)
    #
    # # 场景1
    # low_value, high_value = 50, 110
    # result = get_gene_features(starts, ends, low_value, high_value)
    # print(result)
    # print("\n")
    #
    # # 场景2
    # low_value, high_value = 120, 160
    # result = get_gene_features(starts, ends, low_value, high_value)
    # print(result)
    # print("\n")
    #
    # # 场景3 180+---------------+240
    # low_value, high_value = 180, 240
    # result = get_gene_features(starts, ends, low_value, high_value)
    # print(result)
    # print("\n")
    #
    # # 场景4 50+--+90
    # low_value, high_value = 50, 90
    # result = get_gene_features(starts, ends, low_value, high_value)
    # print(result)
    # print("\n")
    #
    # # 场景5 230+-------------+300
    # low_value, high_value = 230, 300
    # result = get_gene_features(starts, ends, low_value, high_value)
    # print(result)
    # print("\n")
    #
    # # 场景6   50+-------------------------------------------------------------+300
    # low_value, high_value = 50, 300
    # result = get_gene_features(starts, ends, low_value, high_value)
    # print(result)
    # print("\n")
    #
    # # 场景6   50+-------------------------------------------------------------+300
    # low_value, high_value = 500, 800
    # result = get_gene_features(starts, ends, low_value, high_value)
    # print(result)
    # print("\n")

    #high_dis



    #print()
    gff_file = 'F:\\Research\\Genomics\\humen51\\GCF_000001405.25_GRCh37.p13_genomic.gff'

    # 注释类型
    include_types = ['enhancer',
                     'promoter',
                     'pseudogene',
                     'insulator',
                     'conserved_region',
                     'protein_binding_site',
                     'DNAseI_hypersensitive_site',
                     'nucleotide_cleavage_site',
                     'silencer',
                     'CDS',
                     'guide_RNA',
                     'rRNA',
                     'regulatory_region',
                     'snRNA',
                     'RNase_P_RNA',
                     'exon',
                     'insulator',
                     'miRNA',
                     'microsatellite',
                     'vault_RNA',
                     'mRNA',
                     'tRNA',
                     'minisatellite',
                     'snoRNA',
                     'locus_control_region',
                     'CAGE_cluster',
                     'RNase_MRP_RNA',
                     'transcript',
                     'TATA_box',
                     'telomerase_RNA',
                     'transcriptional_cis_regulatory_region',
                     'antisense_RNA',
                     'lnc_RNA',
                     'Y_RNA',
                     'imprinting_control_region',
                     'enhancer_blocking_element',
                     'nucleotide_motif',
                     'gene',
                     'primary_transcript'
                     ]

    chr_gff_dict = get_refseq_gff(gff_file, include_types)

    # # print("chr_gff_dict: ", chr_gff_dict['chr1'])
    #
    # # gene	17369	17436
    # # primary_transcript	17369	17436
    # # exon	17369	17436
    # # miRNA	17369	17391
    # # exon	17369	17391
    # # miRNA	17409	17431
    # # exon	17409	17431
    #
    # print("============")
    # chr_gff = chr_gff_dict.get('NC_000001.10')
    # starts = chr_gff[0]
    # ends = chr_gff[1]
    # anno_types = chr_gff[2]
    #
    # for ii in range(100):
    #     print(starts[ii], ends[ii], anno_types[ii])
    #
    #
    #
    chr, start, end, position = 'NC_000001.10', 1, 1000, 1
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)
    print(result)
    print('========================')

    chr, start, end, position = 'NC_000001.10', 17000, 17300, 2
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)
    print(17000, 17300, result)

    print('========================')

    chr, start, end, position = 'NC_000001.10', 17300, 17400, 3
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)
    print(17300, 17400, result)

    chr, start, end, position = 'NC_000001', 17400, 17410, 4
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)
    print(result)

    chr, start, end, position = 'chr1', 17430, 17450, 5
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)
    print(result)

    chr, start, end, position = 'chr1', 17450, 17460, 6
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)
    print(result)





    # NC_000001.10 42980487 42983487 42981987 KLHDC3_1   gene	42981987	42989032
    chr, start, end, position = 'NC_000001.10', 42980487, 42983487, 7
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)
    print(42980487, 42983487, result)

    import time
    start_t = time.clock()
    # --- NC_000001.10 243326134 243327134 243326634 CEP170_4
    #                  243287730 243418708
    chr, start, end, position = 'NC_000001.10', 243326134, 243327134, 7
    result = get_gene_feature_array(chr_gff_dict, chr, start, end)

    end_t = time.clock()
    print('Running time: %s Seconds' % (end_t - start_t))

    print(243326134, 243327134, result)



