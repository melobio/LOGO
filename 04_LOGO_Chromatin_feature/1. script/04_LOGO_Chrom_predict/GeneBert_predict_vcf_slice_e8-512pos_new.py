import joblib
import argparse
import json
import math
import os
import random
import sys
import numpy as np
import pyfasta
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Lambda, Dense
from multiprocessing import Pool

sys.path.append("../../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number
from bgi.bert4keras.backend import K

if tf.__version__.startswith('1.'):  # tensorflow 1
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
else:  # tensorflow 2
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_dictionary(config_path, encoding="utf-8"):
    '''
    Load dict
    :param config_path:
    :param encoding:
    :return:
    '''
    with open(config_path, mode="r", encoding=encoding) as file:
        str = file.read()
        config = json.loads(str)
        return config


def load_tfrecord(record_names, sequence_length=100, num_classes=919, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                  batch_size=32):
    """给原方法补上parse_function
    """

    def parse_function(serialized):
        features = {
            'x': tf.io.FixedLenFeature([sequence_length], tf.int64),
            # 'segmentId': tf.io.FixedLenFeature([sequence_length], tf.int64),
            'y': tf.io.FixedLenFeature([num_classes], tf.int64),
        }
        features = tf.io.parse_single_example(serialized, features)
        masked_sequence = features['x']
        segment_id = K.zeros_like(masked_sequence, dtype='int64')
        sequence = features['y']
        x = {
            'Input-Token': masked_sequence,
            'Input-Segment': segment_id,
        }
        y = {
            'CLS-Activation': sequence
        }

        return x, y

    if not isinstance(record_names, list):
        record_names = [record_names]

    dataset = tf.data.TFRecordDataset(record_names)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    return dataset


# def load_npz_record(x_data, y_data):
#     """给原方法补上parse_function
#     """
#     def parse_function(x_data, y_data):
#         input_token = x_data
#         input_segment = K.zeros_like(input_token, dtype='int64')
#         x = {
#             'Input-Token': input_token,
#             'Input-Segment': input_segment,
#         }
#         y = {
#             'CLS-Activation': y_data
#         }

#         return x, y

#     dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
#     dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
#     return dataset

def load_npz_record(record_names,
                    batch_size=32,
                    ngram=5,
                    only_one_slice=True,
                    slice_index=None,
                    shuffle=False,
                    seq_len=200,
                    num_classes=919,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    ):
    """
    给原方法补上parse_function
    """

    def parse_function(x, y):
        masked_sequence = x
        segment_id = K.zeros_like(masked_sequence, dtype='int64')
        sequence = y
        y = K.cast(sequence, K.floatx())
        x = {
            'Input-Token': masked_sequence,
            'Input-Segment': segment_id,
        }
        y = {
            'CLS-Activation': y
        }

        # print("x: ", masked_sequence)
        # print("y: ", y)
        return x, y

    if not isinstance(record_names, list):
        record_names = [record_names]

    # 数据生成器
    def data_generator():
        x_data_all = []
        y_data_all = []
        for file_name in record_names:
            if str(file_name).endswith('.npz') is False:
                continue
            loaded = np.load(file_name)
            x_data = loaded['x']
            y_data = loaded['y']

            for ii in range(ngram):
                if slice_index is not None and ii != slice_index:
                    continue

                if only_one_slice is True:
                    kk = ii
                    slice_indexes = []
                    max_slice_seq_len = x_data.shape[1] // ngram * ngram
                    for gg in range(kk, max_slice_seq_len, ngram):
                        slice_indexes.append(gg)
                    x_data_slice = x_data[:, slice_indexes]
                    x_data_all.append(x_data_slice)
                    y_data_all.append(y_data)
                else:
                    x_data_all.append(x_data)
                    y_data_all.append(y_data)

        x_data_all = np.concatenate(x_data_all)
        y_data_all = np.concatenate(y_data_all)

        total_size = len(x_data_all) // batch_size * batch_size
        indexes = np.arange(total_size)
        if shuffle == True:
            np.random.shuffle(indexes)

        for ii in range(total_size):
            # 重新 Shuffle 一次
            if ii == total_size - 1:
                if shuffle == True:
                    np.random.shuffle(indexes)

            index = indexes[ii]
            x = x_data_all[index]
            y = y_data_all[index]
            yield x, y

    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.float32, tf.int16),
                                             output_shapes=(
                                                 tf.TensorShape([seq_len]),
                                                 tf.TensorShape([num_classes])
                                             ))
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    return dataset


def pred_result_auc_BiPath(y_pred, y_label):
    print("y_pred: ", y_pred.shape)
    print("y_label: ", y_label.shape)

    y_pred_val = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
    min_len = min(len(y_label), len(y_pred_val))

    aucs = np.zeros(y_pred_val.shape[1], dtype=np.float)
    for ii in range(y_pred_val.shape[1]):
        try:
            auc = roc_auc_score(y_label[:, ii][:min_len], y_pred_val[:, ii][:min_len])
            aucs[ii] = auc
        except ValueError:
            a = 1

    print('Median AUCs')
    print('- Transcription factors: %.3f' % np.median(aucs[124:124 + 690]))
    print('- DNase I-hypersensitive sites: %.3f' % np.median(aucs[:124]))
    print('- Histone marks: %.3f' % np.median(aucs[124 + 690:124 + 690 + 104]))


# 并行计算 ROC_AUC
def calc_roc_auc_score(index, y_label_slice, y_pred_slice):
    auc = 0
    try:
        auc = roc_auc_score(y_label_slice, y_pred_slice)
    except ValueError:
        auc = 0
    return index, auc


def pred_result_auc_BiPath_list(y_preds, y_label, pool_size=8):
    all_y_pred = None

    print("y_pred: ", len(y_preds))
    print("y_label: ", y_label.shape)
    for jj in range(len(y_preds)):
        y_pred = y_preds[jj]
        if all_y_pred is None:
            all_y_pred = y_pred
        else:
            all_y_pred += y_pred

    y_pred = all_y_pred / len(y_preds)
    y_pred_val = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
    y_pred_val = y_pred_val[:, 0:num_classes]

    min_len = min(len(y_label), len(y_pred_val))
    aucs = np.zeros(y_pred_val.shape[1], dtype=np.float)

    pool = Pool(processes=pool_size)
    results = []
    for index in range(y_pred_val.shape[1]):
        result = pool.apply_async(calc_roc_auc_score,
                                  args=(index,
                                        y_label[:, index][:min_len],
                                        y_pred_val[:, index][:min_len]
                                        ))
        results.append(result)
    pool.close()
    pool.join()

    #  2002 feature TF/DHS/HM index
    TF_index = [125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
                167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
                188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
                209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
                230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
                251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
                272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
                293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
                314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334,
                335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
                356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
                377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
                398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
                419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
                440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460,
                461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481,
                482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502,
                503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523,
                524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544,
                545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565,
                566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586,
                587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607,
                608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628,
                629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
                650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670,
                671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691,
                692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712,
                713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733,
                734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754,
                755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775,
                776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796,
                797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814]
    DHS_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                 124, 827, 828, 829, 830, 831, 859, 860, 861, 862, 863, 882, 883, 884, 885, 886, 909, 910, 911, 912,
                 913, 934, 935, 936, 937, 938, 959, 960, 961, 962, 963, 1044, 1045, 1046, 1047, 1048, 1097, 1098, 1099,
                 1100, 1101, 1108, 1109, 1110, 1111, 1112, 1149, 1150, 1151, 1152, 1153, 1159, 1160, 1161, 1162, 1163,
                 1180, 1181, 1182, 1183, 1184, 1191, 1192, 1193, 1194, 1195, 1201, 1202, 1203, 1204, 1205, 1277, 1278,
                 1279, 1280, 1281, 1307, 1308, 1309, 1310, 1311, 1318, 1319, 1320, 1321, 1322, 1345, 1346, 1347, 1348,
                 1349, 1356, 1357, 1358, 1359, 1360, 1367, 1368, 1369, 1370, 1371, 1383, 1384, 1385, 1386, 1387, 1512,
                 1513, 1514, 1515, 1516, 1523, 1524, 1525, 1526, 1527, 1533, 1534, 1535, 1536, 1537, 1543, 1544, 1545,
                 1546, 1547, 1554, 1555, 1556, 1557, 1558, 1565, 1566, 1567, 1568, 1569, 1576, 1577, 1578, 1579, 1580,
                 1594, 1595, 1596, 1597, 1598, 1605, 1606, 1607, 1608, 1609, 1616, 1617, 1618, 1619, 1620, 1627, 1628,
                 1629, 1630, 1631, 1638, 1639, 1640, 1641, 1642, 1649, 1650, 1651, 1652, 1653, 1660, 1661, 1662, 1663,
                 1664, 1683, 1684, 1685, 1686, 1687, 1694, 1695, 1696, 1697, 1698, 1711, 1712, 1713, 1714, 1715, 1774,
                 1775, 1776, 1777, 1778, 1810, 1833, 1845, 1857, 1869, 1881, 1893, 1905, 1918, 1931, 1943, 1955, 1967,
                 1980]
    HM_index = [815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 832, 833, 834, 835, 836, 837, 838, 839, 840,
                841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 864, 865, 866,
                867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 887, 888, 889, 890, 891, 892,
                893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 914, 915, 916, 917, 918,
                919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 939, 940, 941, 942, 943, 944,
                945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 964, 965, 966, 967, 968, 969, 970,
                971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991,
                992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
                1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026,
                1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043,
                1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065,
                1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082,
                1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1102, 1103, 1104,
                1105, 1106, 1107, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126,
                1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143,
                1144, 1145, 1146, 1147, 1148, 1154, 1155, 1156, 1157, 1158, 1164, 1165, 1166, 1167, 1168, 1169, 1170,
                1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1185, 1186, 1187, 1188, 1189, 1190, 1196, 1197,
                1198, 1199, 1200, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219,
                1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236,
                1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253,
                1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270,
                1271, 1272, 1273, 1274, 1275, 1276, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292,
                1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1312, 1313, 1314,
                1315, 1316, 1317, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336,
                1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1350, 1351, 1352, 1353, 1354, 1355, 1361, 1362, 1363,
                1364, 1365, 1366, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1388, 1389, 1390,
                1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407,
                1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424,
                1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441,
                1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458,
                1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475,
                1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492,
                1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509,
                1510, 1511, 1517, 1518, 1519, 1520, 1521, 1522, 1528, 1529, 1530, 1531, 1532, 1538, 1539, 1540, 1541,
                1542, 1548, 1549, 1550, 1551, 1552, 1553, 1559, 1560, 1561, 1562, 1563, 1564, 1570, 1571, 1572, 1573,
                1574, 1575, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1599, 1600,
                1601, 1602, 1603, 1604, 1610, 1611, 1612, 1613, 1614, 1615, 1621, 1622, 1623, 1624, 1625, 1626, 1632,
                1633, 1634, 1635, 1636, 1637, 1643, 1644, 1645, 1646, 1647, 1648, 1654, 1655, 1656, 1657, 1658, 1659,
                1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681,
                1682, 1688, 1689, 1690, 1691, 1692, 1693, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708,
                1709, 1710, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730,
                1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747,
                1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764,
                1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786,
                1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803,
                1804, 1805, 1806, 1807, 1808, 1809, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821,
                1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1834, 1835, 1836, 1837, 1838, 1839,
                1840, 1841, 1842, 1843, 1844, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1858,
                1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1870, 1871, 1872, 1873, 1874, 1875, 1876,
                1877, 1878, 1879, 1880, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1894, 1895,
                1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913,
                1914, 1915, 1916, 1917, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1932,
                1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1944, 1945, 1946, 1947, 1948, 1949, 1950,
                1951, 1952, 1953, 1954, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969,
                1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1981, 1982, 1983, 1984, 1985, 1986, 1987,
                1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001]

    # 汇总结果
    for result in results:
        index, auc = result.get()
        aucs[index] = auc
    #     print('Median AUCs')
    #     print('- Transcription factors: %.3f' % np.median(aucs[124:124 + 690]))
    #     print('- DNase I-hypersensitive sites: %.3f' % np.median(aucs[:124]))
    #     print('- Histone marks: %.3f' % np.median(aucs[124 + 690:124 + 690 + 104]))
    #     print('- All auc: %.3f' % np.median(aucs[:]))
    print('Median AUCs')
    print('- Transcription factors: %.4f' % np.median(aucs[TF_index]))
    print('- DNase I-hypersensitive sites: %.4f' % np.median(aucs[DHS_index]))
    print('- Histone marks: %.4f' % np.median(aucs[HM_index]))
    print('- All auc: %.4f' % np.median(aucs[:]))


def encodeSeqs_ngram(seqs, kernel=5, inputsize=1000, word_dict=None):
    # 思路：创建指定行列的数据框，对数据框插入数据，再将该数据框转为numpy.array
    # 缺点，太慢了！有没有更快的方式？比如numpy结构直接指定位置的替换
    # 改进版本：采用numpy直接替换
    # 有互补链
    # 将输入的序列列表，每条1100序列取中间1000bp，编码为01234，按字典编码为200，加上互补链
    # encodeSeqs_ngram(seqs, kernel = 5, inputsize=1000, old_dict = "./data/word_dict_0118.pkl")
    # 增加了新词纳入字典并保存的机制

    index_from = 3  # 字典的index从3开始
    kernel = kernel  # ngram长度
    seqsnp = np.zeros((len(seqs), inputsize))  # 创建

    mydict = {'A': 1, 'G': 2, 'C': 3, 'T': 4,
              'a': 1, 'g': 2, 'c': 3, 't': 4,
              'N': 0, 'H': 0, 'n': 0, '-': 0}

    old_dict = word_dict
    print("old dict len", len(old_dict))

    print("Catching middle 1000bp seq from 1100bp seq")
    n = 0
    for line in seqs:
        try:
            # 这步是为了保证取出1100的中间1000bp
            cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(
                math.floor(len(line) - (len(line) - inputsize) / 2.0))]
            for i, c in enumerate(cline):
                seqsnp[n][i] = mydict[c]
            if n % 100000 == 0 and n > 0:
                print(n)
            n = n + 1
        except Exception as e:
            print(e)
            print("index:", i, "base:", c)
            print("cline seq:", cline)

    print("from AGCT index to Ngram index")
    try:
        seqsnp = np.array(seqsnp, dtype=int)  # 将上面提取出的序列数组化
        gene = []
        nn = 0
        new_word_num = 0
        for tem_seq in seqsnp:
            tem_seq = np.array(tem_seq, dtype=int)
            for jj in range(0, len(tem_seq), kernel):  # 间隔为5
                # 获取5bp的片段
                gene_str = ''
                for kk in range(kernel):
                    # print(str(seqsnp[jj + kk]))
                    gene_str += str(tem_seq[jj + kk])
                # 将片段对应字典得到index
                # 若无该样式，就加入字典中，重新赋予
                if gene_str not in old_dict:
                    old_dict[gene_str] = len(old_dict) + index_from
                    print(gene_str, "not in old_dict,adding to the new dictionary, index is", old_dict[gene_str])
                    new_word_num = new_word_num + 1
                    # print(new_word_num)
                    # print(old_dict[gene_str])
                tem_value = old_dict[str(gene_str)]  # 输入value找到key
                # print(tem_value)
                gene.append(tem_value)
            if nn % 100000 == 0 and nn > 0:
                print(nn)
            nn = nn + 1
        # 将list重构为n*(1000/5)的array
        ngram_seqsnp = np.reshape(np.array(gene), (seqsnp.shape[0], int(seqsnp.shape[1] / kernel)))

        # get the complementary sequences
        dataflip = ngram_seqsnp[:, ::-1]  # 补了互补链，变成（2n，200）
        ngram_seqsnp = np.concatenate([ngram_seqsnp, dataflip], axis=0)  # 补了互补链，变成（2n,200）

        print("new word number : ", str(new_word_num), "Now dict len is : ", len(old_dict))
        if new_word_num > 0:
            # print("saving newdict to:", "./data/word_dict_0118"+str(new_word_num)+".pkl")
            # save_obj(old_dict,"./data/word_dict_0118"+str(new_word_num)+".pkl")
            print("new word number : ", str(new_word_num))
            print("saving newdict to:", newdict_path)
            save_obj(old_dict, newdict_path)
            del new_word_num

    except Exception as e:
        print(e)
    return ngram_seqsnp


def fetchSeqs(chr, pos, ref, alt, shift=0, inputsize=1000):
    """Fetches sequences from the genome.
    从基因组提取序列
    """
    windowsize = inputsize + 100  # 为了检索容纳插入缺失标记
    mutpos = int(windowsize / 2 - 1 - shift)  # mutpos等于249~1849
    # return string: ref sequence, string: alt sequence, Bool: whether ref allele matches with reference genome
    seq = genome.sequence({'chr': chr, 'start': pos + shift -
                                                int(windowsize / 2 - 1), 'stop': pos + shift + int(windowsize / 2)})
    # seq1 = seq.copy()
    # seq2 = seq.copy()
    return seq[:mutpos] + ref + seq[(mutpos + len(ref)):], seq[:mutpos] + alt + seq[(mutpos + len(ref)):], seq[mutpos:(
                mutpos + len(ref))].upper() == ref.upper()  # 原版


def fetchSeqs_fromdeepC(chr, pos_start, pos_end, inputsize=1000):
    """Fetches sequences from the genome.
    从基因组提取序列
    """
    # windowsize = inputsize+100
    # mutpos = int(windowsize / 2 - 1 - shift)    #mutpos等于249~1849
    # return string: ref sequence, string: alt sequence, Bool: whether ref allele matches with reference genome

    whole_seq = genome.sequence({'chr': chr, 'start': pos_start, 'stop': pos_end})
    print(len(whole_seq))
    num = (pos_end - pos_start) / inputsize
    print("num", num)
    for i in num:
        tem_seq = whole_seq[inputsize * num: inputsize * (num + 1)]

    # seq1 = seq.copy()
    # seq2 = seq.copy()
    return seq[:mutpos] + ref + seq[(mutpos + len(ref)):], seq[:mutpos] + alt + seq[(mutpos + len(ref)):], seq[mutpos:(
                mutpos + len(ref))].upper() == ref.upper()  # 原版


def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.
    将AGCT序列转为onehot，并补充互补链，用于后续ngram化

    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output

    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize

    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
              'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
              'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
              'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
              'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
              'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(
            math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    dataflip = seqsnp[:, ::-1, ::-1]  # 补了互补链，变成（2n，4，2000）
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)  # 补了互补链，变成（2n，4，2000）
    return seqsnp


def onehot_to_ngram(data=None,
                    n_gram=3,
                    step=1,
                    num_word_dict=None,
                    actg_value=np.array([1, 2, 3, 4]),
                    n_gram_value=None):
    """
    将encode后的onthot进行ngram化
    data : (n, 4, seqsize), example:(8000,4,2000)
    n_gram_value : base on ngram, example: ngram=3, n_gram_value=[100,10,1]
    return ngram array
    """

    index = 0
    x_data = []

    for ii in range(data.shape[0]):
        actg = np.matmul(actg_value, data[ii, :, :])
        gene = []
        for kk in range(0, len(actg), step):
            actg_temp_value = 0
            if kk + n_gram <= len(actg):
                actg_temp_value = np.dot(actg[kk:kk + n_gram], n_gram_value)
                actg_temp_value = int(actg_temp_value)
            else:
                for gg in range(kk, len(actg)):
                    actg_temp_value += actg[gg] * (10 ** (n_gram - gg % n_gram - 1))
                actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))
            gene.append(num_word_dict.get(actg_temp_value, 0))

        x_data.append(np.array(gene))
        # y_test.append(labels[ii])

        if index % 10000 == 0 and index > 0:
            print("Index : {}, Gene len : {}".format(index, len(gene)))
        index += 1
        _ngram_array = np.array(x_data)
    return _ngram_array


def npz2record(x_data_,
               batch_size=32,
               ngram=5,
               only_one_slice=True,
               slice_index=None,
               shuffle=False,
               seq_len=200,
               num_classes=919,
               num_parallel_calls=tf.data.experimental.AUTOTUNE,
               ):
    """
    将输入的ngram数组序列化，用于后续tf模型预测
    来自：load_npz_record
    给原方法补上parse_function
    """

    def parse_function(x, y):
        masked_sequence = x
        segment_id = K.zeros_like(masked_sequence, dtype='int64')
        sequence = y
        y = K.cast(sequence, K.floatx())
        x = {
            'Input-Token': masked_sequence,
            'Input-Segment': segment_id,
        }
        y = {
            'CLS-Activation': y
        }

        # print("x: ", masked_sequence)
        # print("y: ", y)
        return x, y

    # 数据生成器
    def data_generator():
        x_data_all = []
        y_data_all = []

        x_data = x_data_  # 一定要保证是(n, 2000)，ngram=3，stride=1之后也是2000
        print(x_data.shape)
        y_data = np.zeros((x_data.shape[0], num_classes), np.bool_).astype(np.int)  # 伪造数据
        # print(y_data.shape)

        for ii in range(ngram):
            if slice_index is not None and ii != slice_index:
                continue

            if only_one_slice is True:
                kk = ii
                slice_indexes = []
                max_slice_seq_len = x_data.shape[1] // ngram * ngram
                for gg in range(kk, max_slice_seq_len, ngram):
                    slice_indexes.append(gg)
                x_data_slice = x_data[:, slice_indexes]
                x_data_all.append(x_data_slice)
                y_data_all.append(y_data)
            else:
                x_data_all.append(x_data)
                y_data_all.append(y_data)

        x_data_all = np.concatenate(x_data_all)
        y_data_all = np.concatenate(y_data_all)

        # 这一步会漏掉一些预测值
        # total_size = len(x_data_all) // batch_size * batch_size
        total_size = len(x_data_all)
        indexes = np.arange(total_size)
        if shuffle == True:
            np.random.shuffle(indexes)

        for ii in range(total_size):
            # 重新 Shuffle 一次
            if ii == total_size - 1:
                if shuffle == True:
                    np.random.shuffle(indexes)

            index = indexes[ii]
            x = x_data_all[index]
            y = y_data_all[index]
            yield x, y

    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.float32, tf.int16),
                                             output_shapes=(
                                                 tf.TensorShape([seq_len]),
                                                 tf.TensorShape([num_classes])
                                             ))
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    return dataset


def predict_avg(ngram_input=None,
                TEM_BATCH_SIZE=32,
                ngram=5,
                seq_len=2000,
                num_classes=2002):
    """
    ngram次预测结果取平均会更加准确
    ngram_input : ngram numpy array
    TEM_BATCH_SIZE : predict batch size, default=1
    return : y_pred, numpy array
    """
    y_preds = []
    for ii in range(ngram):
        dataset = npz2record(ngram_input, batch_size=TEM_BATCH_SIZE, ngram=ngram, only_one_slice=True,
                             slice_index=ii, shuffle=False, seq_len=seq_len, num_classes=num_classes)
        dataset = dataset.batch(TEM_BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        y_pred = albert.predict(dataset,
                                steps=math.ceil(ngram_input.shape[0] / (TEM_BATCH_SIZE)),
                                verbose=1)
        print()
        print("Predict epoch:{}, y_pred_shape : {}".format(ii, y_pred.shape))
        y_preds.append(y_pred)

    all_y_pred = None
    for jj in range(len(y_preds)):
        y_pred = y_preds[jj]
        if all_y_pred is None:
            all_y_pred = y_pred
        else:
            all_y_pred += y_pred

    y_pred = all_y_pred / len(y_preds)
    y_pred_val = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
    y_pred_val = y_pred_val[:, 0:num_classes]

    return y_pred_val


def preccess_data(seqslist,
                  inputsize,
                  ngram,
                  step,
                  word_dict,
                  actg_value,
                  n_gram_value
                  ):
    tem_encoded = encodeSeqs(seqslist, inputsize=inputsize).astype(np.float32)  # shape是(20, 4, 2000),因为包含了正和反两条链
    print('seqslist length : {}, \ntem_encoded shape : {}'.format(len(seqslist), tem_encoded.shape))
    tem_ngram_input = onehot_to_ngram(data=tem_encoded, n_gram=ngram, step=step,
                                      num_word_dict=word_dict,
                                      actg_value=actg_value,
                                      n_gram_value=n_gram_value)
    tem_ngram_input_lenth = tem_ngram_input.shape[0]
    print(tem_ngram_input_lenth)
    if tem_ngram_input_lenth / 2 == len(seqslist):
        pos_ngram_input = tem_ngram_input[:int(tem_ngram_input_lenth / 2), :]  # pos
        neg_ngram_input = tem_ngram_input[int(tem_ngram_input_lenth / 2):, :]  # pneg
        output = (pos_ngram_input, neg_ngram_input)
    else:
        output = tem_ngram_input

    return output


# =================================


if __name__ == '__main__':

    _argparser = argparse.ArgumentParser(
        description='A simple example of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # _argparser.add_argument(
    # '--save', type=str, required=True, metavar='PATH',
    # help='A path where the best model should be saved / restored from')
    _argparser.add_argument(
        '--inputfile', type=str, metavar='PATH', default=None,
        help='Path to vcf file')
    _argparser.add_argument(
        '--backgroundfile', type=str, metavar='PATH', default=None,
        help='Path to background file')
    _argparser.add_argument(
        '--maxshift', type=int, default=0, metavar='INTEGER',
        help='The number of shift seq')
    _argparser.add_argument(
        '--reffasta', type=str, metavar='PATH', default='/data/male.hg19.fasta',
        help='Path to a file of reference')
    _argparser.add_argument(
        '--weight-path', type=str, metavar='PATH', default=None,
        help='Path to a pretain weight')
    _argparser.add_argument(
        '--epochs', type=int, default=150, metavar='INTEGER',
        help='The number of epochs to train')
    _argparser.add_argument(
        '--lr', type=float, default=2e-4, metavar='FLOAT',
        help='Learning rate')
    _argparser.add_argument(
        '--batch-size', type=int, default=32, metavar='INTEGER',
        help='Training batch size')
    _argparser.add_argument(
        '--seq-len', type=int, default=256, metavar='INTEGER',
        help='Max sequence length')
    _argparser.add_argument(
        '--we-size', type=int, default=128, metavar='INTEGER',
        help='Word embedding size')
    _argparser.add_argument(
        '--model', type=str, default='universal', metavar='NAME',
        choices=['universal', 'vanilla'],
        help='The type of the model to train: "vanilla" or "universal"')
    _argparser.add_argument(
        '--CUDA-VISIBLE-DEVICES', type=int, default=0, metavar='INTEGER',
        help='CUDA_VISIBLE_DEVICES')
    _argparser.add_argument(
        '--num-classes', type=int, default=10, metavar='INTEGER',
        help='Number of total classes')
    _argparser.add_argument(
        '--vocab-size', type=int, default=20000, metavar='INTEGER',
        help='Number of vocab')
    # _argparser.add_argument(
    # '--slice', type=list, default=[6],
    # help='Slice')
    _argparser.add_argument(
        '--slice-size', type=int, default=10000, metavar='INTEGER',
        help='Slice size')
    _argparser.add_argument(
        '--ngram', type=int, default=6, metavar='INTEGER',
        help='length of char ngram')
    _argparser.add_argument(
        '--stride', type=int, default=2, metavar='INTEGER',
        help='stride size')
    _argparser.add_argument(
        '--has-segment', action='store_true',
        help='Include segment ID')
    # _argparser.add_argument(
    # '--word-prediction', action='store_true',
    # help='Word prediction')
    # _argparser.add_argument(
    # '--class-prediction', action='store_true',
    # help='class prediction')
    _argparser.add_argument(
        '--num-heads', type=int, default=4, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--model-dim', type=int, default=128, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--transformer-depth', type=int, default=2, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--num-gpu', type=int, default=1, metavar='INTEGER',
        help='Number of GPUs')
    _argparser.add_argument(
        '--task', type=str, default='train', metavar='NAME',
        choices=['train', 'valid', 'test'],
        help='The type of the task')
    _argparser.add_argument(
        '--verbose', type=int, default=2, metavar='INTEGER',
        help='Verbose')
    _argparser.add_argument(
        '--steps-per-epoch', type=int, default=10000, metavar='INTEGER',
        help='steps per epoch')
    _argparser.add_argument(
        '--shuffle-size', type=int, default=1000, metavar='INTEGER',
        help='Buffer shuffle size')
    _argparser.add_argument(
        '--num-parallel-calls', type=int, default=16, metavar='INTEGER',
        help='Num parallel calls')
    _argparser.add_argument(
        '--prefetch-buffer-size', type=int, default=4, metavar='INTEGER',
        help='Prefetch buffer size')
    _argparser.add_argument(
        '--pool-size', type=int, default=16, metavar='INTEGER',
        help='Pool size of multi-thread')

    _args = _argparser.parse_args()

    # save_path = _args.save

    # model_name = _args.model_name
    # config_save_path = os.path.join(save_path, "{}_config.json".format(model_name))

    batch_size = _args.batch_size
    epochs = _args.epochs
    num_gpu = _args.num_gpu

    max_seq_len = _args.seq_len
    initial_epoch = 0

    ngram = _args.ngram
    stride = _args.stride

    word_from_index = 3
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)

    num_classes = _args.num_classes
    only_one_slice = True
    # vocab_size = len(word_dict) + word_from_index
    vocab_size = len(word_dict) + word_from_index + 10

    slice_size = _args.slice_size
    pool_size = _args.pool_size

    max_depth = _args.transformer_depth
    model_dim = _args.model_dim
    embedding_size = _args.we_size
    num_heads = _args.num_heads

    shuffle_size = _args.shuffle_size
    num_parallel_calls = _args.num_parallel_calls
    prefetch_buffer_size = _args.prefetch_buffer_size

    word_seq_len = max_seq_len // ngram * int(ngram / ngram)
    print("max_seq_len: ", max_seq_len, " word_seq_len: ", word_seq_len)

    # Distributed Training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    with strategy.scope():
        # 模型配置
        config = {
            "attention_probs_dropout_prob": 0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0,
            "embedding_size": embedding_size,
            "hidden_size": model_dim,
            "initializer_range": 0.02,
            "intermediate_size": model_dim * 4,
            "max_position_embeddings": 512,       # 必须修改为512
            "num_attention_heads": num_heads,
            "num_hidden_layers": max_depth,
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 0,
            "vocab_size": vocab_size,
            "custom_masked_sequence": False,
        }
        bert = build_transformer_model(
            configs=config,
            # checkpoint_path=checkpoint_path,
            model='albert',
            return_keras_model=False,
        )

        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        output = Dense(
            name='CLS-Activation',
            units=num_classes,
            activation='sigmoid',
            kernel_initializer=bert.initializer
        )(output)

        albert = tf.keras.models.Model(bert.model.input, output)
        albert.summary()
        albert.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy()], metrics=['accuracy'])

    with strategy.scope():
        pretrain_weight_path = _args.weight_path
        if pretrain_weight_path is not None and len(pretrain_weight_path) > 0:
            albert.load_weights(pretrain_weight_path, by_name=True)
            print("Load weights: ", pretrain_weight_path)

    steps_per_epoch = _args.steps_per_epoch

    lr_scheduler = LRSchedulerPerStep(model_dim,
                                      warmup=2500,
                                      initial_epoch=initial_epoch,
                                      steps_per_epoch=steps_per_epoch)

    # loss_name = "val_loss"
    # LOG_FILE_PATH = LoG_PATH + '_checkpoint-{}.hdf5'.format(epoch)
    # filepath = os.path.join(save_path, model_name + "_weights_{epoch:02d}-{accuracy:.4f}-{val_accuracy:.4f}.hdf5")

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    print("GLOBAL_BATCH_SIZE: ", GLOBAL_BATCH_SIZE)
    print("shuffle_size: ", shuffle_size)

    actg_value = np.array([1, 2, 3, 4])
    print("actg_value", actg_value)
    n_gram_value = np.ones(ngram)
    for ii in range(ngram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (ngram - ii - 1)))
    print("n_gram_value: ", n_gram_value)

    # mc = ModelCheckpoint(filepath, monitor=loss_name, save_best_only=False, verbose=1)

    # save_path= '/data/BGI-Gene_new/data'
    # model_name = 'genebert_3_gram_2_layer_8_heads_256_dim_test'
    # config_save_path = os.path.join(save_path, "{}_config.json".format(model_name))
    # batch_size = 32
    # epochs =150
    # num_gpu = 1
    # max_seq_len = 2000
    # initial_epoch = 0

    # ngram = 3
    # stride = 1
    # actg_value = np.array([1, 2, 3, 4])
    # print("actg_value", actg_value)
    # n_gram_value = np.ones(ngram)
    # for ii in range(ngram):
    # n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (ngram - ii - 1)))
    # print("n_gram_value: ", n_gram_value)

    # word_from_index = 3
    # word_dict = get_word_dict_for_n_gram_number(ngram=ngram)
    # num_word_dict = word_dict
    # num_classes = 2002
    # only_one_slice = True
    # vocab_size = len(word_dict) + word_from_index + 10
    # slice = [6]
    # max_depth = 2
    # model_dim = 256
    # embedding_size = 128
    # num_heads = 8
    # class_prediction = True
    # word_prediction = False
    # shuffle_size = 4000
    # num_parallel_calls = 16
    # prefetch_buffer_size = 4
    # steps_per_epoch = 10000
    # word_seq_len =  max_seq_len // ngram * int(ngram/ngram)
    # print("max_seq_len: ", max_seq_len, " word_seq_len: ", word_seq_len)
    # task = 'valid'
    # pool_size = 16
    # # exp mat
    # weight_path = '/data/BGI-Gene_new/data_exp_mat/genebert_3_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_67-0.9601-0.9674.hdf5'
    # print(weight_path)
    # GLOBAL_BATCH_SIZE = batch_size * num_gpu
    # print("GLOBAL_BATCH_SIZE: ", GLOBAL_BATCH_SIZE)
    # print("shuffle_size: ", shuffle_size)

    # =================================

    genome = pyfasta.Fasta(_args.reffasta)
    CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
            'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
            'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

    # 数据染色质名称标准化
    inputfile = _args.inputfile
    # inputfile = "402_var_from_Fine-mapping_refalt.vcf"
    vcf = pd.read_csv(inputfile, sep='\t', header=None, comment='#')  # 10个
    # standardize 把vcf文件中，染色体名字为数字的，改成chr+数字的格式
    vcf.iloc[:, 0] = 'chr' + vcf.iloc[:, 0].map(str).str.replace('chr', '')
    vcf = vcf[vcf.iloc[:, 0].isin(CHRS)]  # 判断输入的VCF文件是否都符合条件
    vcf.columns = ['chr', 'pos', 'name', 'ref', 'alt'] + list(vcf.columns[5:])  # 修改了对vcf的修改，支持info的输出
    vcf.pos = vcf.pos.astype(int)
    # vcf = vcf[:10]
    print('VCF file shape is : \n', vcf.shape)
    # vcf

    # =================================
    maxshift = _args.maxshift
    inputsize = _args.seq_len
    # maxshift = 0
    # inputsize = 2000
    for shift in [0, ] + list(range(-200, -maxshift - 1, -200)) + list(range(200, maxshift + 1, 200)):
        refseqs = []
        altseqs = []
        ref_matched_bools = []
        print("shift is ", shift)
        print("__Fetching Seqs...__")
        for i in range(vcf.shape[0]):
            refseq, altseq, ref_matched_bool = fetchSeqs(
                vcf.iloc[i, 0], vcf.iloc[i, 1], vcf.iloc[i, 3], vcf.iloc[i, 4], shift=shift, inputsize=inputsize)
            refseqs.append(refseq)  # 10条vcf信息对应的refseq，长度为1100
            altseqs.append(altseq)  # 10条vcf信息对应的altseq，长度为1100
            ref_matched_bools.append(ref_matched_bool)  # ref等位基因是否与参考基因组匹配上
        print("catch refseq length:", len(refseq))
        print("catch altseq length:", len(altseq))

        if shift == 0:
            # only need to be checked once
            print("Number of variants with reference allele matched with reference genome:")
            print(np.sum(ref_matched_bools))
            print("Number of input variants:")
            print(len(ref_matched_bools))
            print("Computing 1st file..")

        try:
            # print('__Processing REF Seqs and ALT Seqs__')
            # ref_encoded = encodeSeqs(refseqs, inputsize=inputsize).astype(np.float32)    #shape是(20, 4, 2000),因为包含了正和反两条链
            # print('ref_encoded shape', ref_encoded.shape)
            # # 输入序列，得到转换后可被tf-deepsea接受的格式，2100bp变成中间的2000bp变成片段变成词index
            # ref_ngram_input = onehot_to_ngram(data=ref_encoded, n_gram = ngram, step = stride,
            # num_word_dict = word_dict,
            # actg_value = actg_value,
            # n_gram_value = n_gram_value)
            # alt_encoded = encodeSeqs(altseqs, inputsize=inputsize).astype(np.float32)    #shape是(20, 4, 2000),因为包含了正和反两条链
            # print('alt_encoded shape', alt_encoded.shape)
            # alt_ngram_input = onehot_to_ngram(data=alt_encoded, n_gram = ngram, step = stride,
            # num_word_dict = word_dict,
            # actg_value = actg_value,
            # n_gram_value = n_gram_value)

            # print('__Predicting__')
            # y_pred_ref_ori = predict_avg(ngram_input=ref_ngram_input,
            # TEM_BATCH_SIZE=GLOBAL_BATCH_SIZE,
            # ngram=ngram,
            # seq_len=word_seq_len,
            # num_classes=num_classes)
            # y_pred_ref = np.where(y_pred_ref_ori>0.0000001, y_pred_ref_ori, 0.0000001)
            # print("y_pred_ref shape",y_pred_ref.shape)
            # y_pred_alt_ori = predict_avg(ngram_input=alt_ngram_input,
            # TEM_BATCH_SIZE=GLOBAL_BATCH_SIZE,
            # ngram=ngram,
            # seq_len=word_seq_len,
            # num_classes=num_classes)
            # y_pred_alt = np.where(y_pred_alt_ori>0.0000001, y_pred_alt_ori, 0.0000001)
            # print("y_pred_alt shape",y_pred_alt.shape)

            print('__Processing REF Seqs and ALT Seqs__')
            # 使用多进程并行处理
            pool = Pool(processes=pool_size)
            data_all_list = []

            for tem_seqs in [refseqs, altseqs]:

                num_row = len(tem_seqs)
                results = []
                for ii in range(math.ceil(num_row / slice_size)):
                    # print('{}:{}'.format(slice_size*ii, slice_size*(ii+1)))
                    slice_seqslist = tem_seqs[slice_size * ii: slice_size * (ii + 1)]  # 溢出也没关系
                    # print(len(slice_seqslist))

                    result = pool.apply_async(preccess_data,
                                              args=(slice_seqslist,
                                                    inputsize,
                                                    ngram,
                                                    stride,
                                                    word_dict,
                                                    actg_value,
                                                    n_gram_value
                                                    ))
                    results.append(result)

                pos_data_all = []
                neg_data_all = []

                # 汇总结果
                for result in results:
                    pos_data, neg_data = result.get()
                    if len(pos_data) > 0 and len(neg_data) > 0 and len(pos_data) == len(neg_data):
                        pos_data_all.extend(pos_data)
                        neg_data_all.extend(neg_data)

                pos_data_all = np.array(pos_data_all)
                neg_data_all = np.array(neg_data_all)

                data_all = np.vstack([pos_data_all, neg_data_all])
                print("data_all: ", data_all.shape)
                data_all_list.append(data_all)

            pool.close()
            pool.join()

            print('__Predicting__')
            y_pred_ref_ori = predict_avg(ngram_input=data_all_list[0],
                                         TEM_BATCH_SIZE=int(batch_size),
                                         ngram=ngram,
                                         seq_len=word_seq_len,
                                         num_classes=num_classes)
            y_pred_ref = np.where(y_pred_ref_ori > 0.0000001, y_pred_ref_ori, 0.0000001)
            print("y_pred_ref shape", y_pred_ref.shape)
            y_pred_alt_ori = predict_avg(ngram_input=data_all_list[1],
                                         TEM_BATCH_SIZE=int(batch_size),
                                         ngram=ngram,
                                         seq_len=word_seq_len,
                                         num_classes=num_classes)
            y_pred_alt = np.where(y_pred_alt_ori > 0.0000001, y_pred_alt_ori, 0.0000001)
            print("y_pred_alt shape", y_pred_alt.shape)

        except Exception as e:
            print(e)

        # 将ref与alt相减，data中分别包含了logfolddiff数组与diff数组
        data = np.hstack([
            np.log2(y_pred_alt / (1 - y_pred_alt + 1e-12)) - np.log2(y_pred_ref / (1 - y_pred_ref + 1e-12)),
            y_pred_alt - y_pred_ref])
        data = data[:int((data.shape[0] / 2)), :] / 2.0 + data[int((data.shape[0] / 2)):, :] / 2.0  # 各自取平均
        print("logfoldchange and diff array shape :", data.shape)

        # print("Writing to csv")
        # header = np.loadtxt('/alldata/LChuang_data/myP/DeepSEA/DeepSEA-v0.94/resources/predictor.names',dtype=np.str)
        header = list(range(num_classes))
        wfile1 = "{}_{}bs_{}gram_{}feature.out.ref.csv".format(inputfile, batch_size, ngram, int(data.shape[1] / 2))
        wfile2 = "{}_{}bs_{}gram_{}feature.out.alt.csv".format(inputfile, batch_size, ngram, int(data.shape[1] / 2))
        wfile3 = "{}_{}bs_{}gram_{}feature.out.logfoldchange.csv".format(inputfile, batch_size, ngram,
                                                                         int(data.shape[1] / 2))
        wfile4 = "{}_{}bs_{}gram_{}feature.out.diff.csv".format(inputfile, batch_size, ngram, int(data.shape[1] / 2))
        wfile6 = "{}_{}bs_{}gram_{}feature.out.evalue.csv".format(inputfile, batch_size, ngram, int(data.shape[1] / 2))
        wfile7 = wfile6.replace('evalue.csv', 'evalue_gmean.csv')
        wfile8 = wfile6.replace('evalue.csv', 'funsig.csv')
        vcf_row = vcf.shape[0]

        # write reference allele prediction, alternative allele prediction, relative difference and absolution difference files
        y_pred_ref = y_pred_ref[:int((y_pred_ref.shape[0] / 2)), :] / 2.0 + y_pred_ref[int((y_pred_ref.shape[0] / 2)):,
                                                                            :] / 2.0  # 正链结果/2  + 负链结果/2
        y_pred_alt = y_pred_alt[:int((y_pred_alt.shape[0] / 2)), :] / 2.0 + y_pred_alt[int((y_pred_alt.shape[0] / 2)):,
                                                                            :] / 2.0
        temp = pd.DataFrame(y_pred_ref)
        temp.columns = header
        if vcf_row == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile1, float_format='%.8f', header=True, index=False)
            print("Saving ", wfile1)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")

        temp = pd.DataFrame(y_pred_alt)
        temp.columns = header
        if vcf_row == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile2, float_format='%.8f', header=True, index=False)
            print("Saving ", wfile2)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")

            # 相对差异和绝对差异data的前num_classes列和后num_classes列
        # logfoldchange
        temp = pd.DataFrame(data[:, :num_classes])
        temp.columns = header
        if vcf_row == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile3, float_format='%.8f', header=True, index=False)
            print("Saving ", wfile3)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")

        # diff
        temp = pd.DataFrame(data[:, num_classes:])
        temp.columns = header
        if vcf_row == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile4, float_format='%.8f', header=True, index=False)
            print("Saving ", wfile4)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")


        # #compute E-values for chromatin effects（版本1）
        # backgroundfile = _args.backgroundfile
        # ecdfs=joblib.load(backgroundfile)
        # datae=np.ones((data.shape[0],num_classes))
        # for i in range(num_classes):
        # datae[:,i]=1-ecdfs[i](np.abs(data[:,i+num_classes]*data[:,i]))
        # #将0值替换
        # datae[datae==0]=1e-6

        # compute E-values for chromatin effects（版本2）
        print("compute E-values for chromatin effects V2")
        datae = np.ones((data.shape[0], num_classes))
        # json_pkl_path = '/alldata/Nzhang_data/project/T2D/2.background/1.2002mark_5gram/'
        json_pkl_path = _args.backgroundfile
        tem_file = os.listdir(json_pkl_path)
        pkl_filelist = []
        for item in tem_file:
            if item.endswith('.json'):
                json_file_name = os.path.join(json_pkl_path, item)
                print(json_file_name)
                pkl_dict = json.load(open(json_file_name))  # load dict
            elif item.endswith('.pkl'):
                pkl_filelist.append(item)
        print("pkl_filelist length is :", len(pkl_filelist))
        # 对每一列计算evalue
        for i in range(num_classes):
            tem_background_pkl = pkl_dict[str(i)]  # get pkl file
            tem_background_pkl = os.path.join(json_pkl_path, tem_background_pkl)
            ecdfs = joblib.load(tem_background_pkl)
            datae[:, i] = 1 - ecdfs(np.abs(data[:, i + num_classes] * data[:, i]))
            if i % 100 == 0:
                print("Pkl finnished :", i)
                # print("Finished:", tem_background_pkl)
        # 将0值替换
        datae[datae == 0] = 1e-6
        print("Finished all, writing E-value output file...")

        # write E-values for chromatin effects
        temp_evalue = pd.DataFrame(datae[:, :num_classes])
        temp_evalue.columns = header
        if vcf_row == temp_evalue.shape[0]:
            temp = pd.concat([vcf, temp_evalue], axis=1)
            temp.to_csv(wfile6, float_format='%.8f', header=True, index=False)
            print("Saving ", wfile6)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")

        del temp

        # write gmean of E-values for chromatin effects
        print("Writing gmean of E-value output file...")
        import scipy
        from scipy import stats

        gmean_row_value = scipy.stats.gmean(temp_evalue, axis=1)
        gmean_df = pd.DataFrame(list(gmean_row_value))
        gmean_df.columns = ['gmean']
        print(gmean_df.shape)

        new_df = pd.concat([vcf, gmean_df], axis=1)
        new_df.to_csv(wfile7, sep=',', index=None)
        print(new_df.shape)
        print("Saving ", wfile7)

        # write compute E-values for Functional Significance scores
        #print("Writing gmean of Functional-Significance output file...")
        #datadeepsea = np.exp(np.mean(np.log(datae), axis=1))
        #datadeepsea_df = pd.DataFrame(list(datadeepsea))
        #datadeepsea_df.columns = ['Functional significance score']
        #print(datadeepsea_df.shape)

        #new_df = pd.concat([vcf, datadeepsea_df], axis=1)
        #new_df.to_csv(wfile8, sep=',', index=None)
        #print(new_df.shape)
        #print("Saving ", wfile8)

        # 立即跳出程序
        os._exit(0)



"""
# 
"""

'''
# 并行版本(2002)
cd /data/BGI-Gene_new/examples/vcf-predict
CUDA_VISIBLE_DEVICES=2,3 python GeneBert_predict_vcf_slice.py \
--inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
--reffasta /data/male.hg19.fasta \
--maxshift 0 \
--weight-path /data/BGI-Gene_new/data_exp_mat/genebert_3_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_67-0.9601-0.9674.hdf5 \
--seq-len 2000 \
--model-dim 256 \
--transformer-depth 2 \
--num-heads 8 \
--batch-size 256 \
--num-classes 2002 \
--shuffle-size 4000 \
--pool-size 64 \
--slice-size 10000 \
--ngram 3 \
--stride 1
# 并行版本(3357)
cd /data/BGI-Gene_new/examples/vcf-predict
CUDA_VISIBLE_DEVICES=4,5,6,7 python GeneBert_predict_vcf_slice.py \
--inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
--reffasta /data/male.hg19.fasta \
--maxshift 0 \
--weight-path /data/BGI-Gene_new/data_wanrenexp_3357_selene/genebert_3_gram_2_layer_8_heads_256_dim_wanrenexp_[baseonEpoch10]_weights_106-0.9746-0.9768.hdf5 \
--seq-len 2000 \
--model-dim 256 \
--transformer-depth 2 \
--num-heads 8 \
--batch-size 256 \
--num-classes 3357 \
--shuffle-size 4000 \
--pool-size 64 \
--slice-size 10000 \
--ngram 3 \
--stride 1

'''

'''
# 非并行版本

# cd /data/BGI-Gene_new/examples/vcf-predict
# CUDA_VISIBLE_DEVICES=0 python GeneBert_predict_vcf.py \
# --inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
# --reffasta /data/male.hg19.fasta \
# --maxshift 0 \
# --weight-path /data/BGI-Gene_new/data_exp_mat/genebert_3_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_109-0.9613-0.9678.hdf5 \
# --seq-len 2000 \
# --model-dim 256 \
# --transformer-depth 2 \
# --num-heads 8 \
# --batch-size 256 \
# --num-classes 2002 \
# --shuffle-size 4000 \
# --ngram 3 \
# --stride 1

# wanrenexp3357 5gram ori
# cd /data/BGI-Gene_new/examples/vcf-predict
# CUDA_VISIBLE_DEVICES=1 python GeneBert_predict_vcf.py \
# --inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
# --reffasta /data/male.hg19.fasta \
# --maxshift 0 \
# --weight-path /data/BGI-Gene_new/data_wanrenexp_3357_selene/genebert_3_gram_2_layer_8_heads_256_dim_wanrenexp_[baseonEpoch10]_weights_108-0.9746-0.9764.hdf5 \
# --seq-len 2000 \
# --model-dim 256 \
# --transformer-depth 2 \
# --num-heads 8 \
# --batch-size 256 \
# --num-classes 3357 \
# --shuffle-size 4000 \
# --ngram 3 \
# --stride 1

# --weight-path /data/BGI-Gene_new/data_wanrenexp_3357_selene/genebert_3_gram_2_layer_8_heads_256_dim_wanrenexp_[baseonEpoch10]_weights_108-0.9746-0.9764.hdf5
# --weight-path /data/BGI-Gene_new/data_exp_mat/genebert_3_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_109-0.9613-0.9678.hdf5
'''
