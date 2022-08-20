import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from torch import nn
import tensorflow as tf
from copy import deepcopy
import pickle

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.Environments.utils.action_utils import replace_with_synonym_greedy, get_similarity, possible_actions

# importing the text model attacked and used for embedding
from src.TextModels.Bert import BertTextModel
from src.TextModels.WordLSTM import WordLSTM
from src.TextModels.XLNet import XLNetTextModel


# importing the textfooler word_importance
from src.Attacks.textfooler_attack import calc_word_importance

# fix randomness
from src.Attacks.utils.optim_utils import seed_everything


class GenFooler(nn.Module):
    def __init__(self):
        super(GenFooler, self).__init__()
        self.fc1 = nn.Linear(768, 400, bias=True)
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(400, 400, bias=True)
        self.fc2.bias.data.fill_(0)
        self.fc21 = nn.Linear(400, 400, bias=True)
        self.fc21.bias.data.fill_(0)
        self.fc22 = nn.Linear(400, 400, bias=True)
        self.fc22.bias.data.fill_(0)
        self.fc23 = nn.Linear(400, 400, bias=True)
        self.fc23.bias.data.fill_(0)
        self.fc24 = nn.Linear(400, 400, bias=True)
        self.fc24.bias.data.fill_(0)
        self.fc3 = nn.Linear(400, 150, bias=True)
        self.fc3.bias.data.fill_(0)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(-1, 768)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc21(x))
        x = self.act(self.fc22(x))
        x = self.act(self.fc23(x))
        x = self.act(self.fc24(x))
        x = self.fc3(x)
        return x


def prepare_data(path, df_train, df_test, text_model, use_pre_calculated=False, imp_type='tf', sess=None):
    if use_pre_calculated:
        with open(path + f'_genfooler_{model_type}/data_{imp_type}.pkl', 'rb') as f:
            return pickle.load(f)

    # embed all the texts
    full_train_emb = embed_all_texts(text_model, df_train.content.values.tolist())
    test_emb = embed_all_texts(text_model, df_test.content.values.tolist())
    print("Embedding calculated for all texts")

    # generate masks indicating how many words each text has
    full_train_mask = mask_all_texts(df_train.content.values.tolist())
    test_mask = mask_all_texts(df_test.content.values.tolist())
    print("Masks calculated for all texts")

    # generate textfooler importance labels
    full_train_imp = importance_all_texts(text_model, df_train.content.values.tolist(), imp_type, sess)
    test_imp = importance_all_texts(text_model, df_test.content.values.tolist(), imp_type, sess)
    print("Importance calculated for all texts")

    with open(path + f'_genfooler_{model_type}/data_{imp_type}.pkl', 'wb') as f:
        pickle.dump((full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp), f)

    return full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp


def embed_all_texts(text_model: BertTextModel, texts: list):
    # maybe change to batch proccessing in the future
    emb_list = []
    for text in texts:
        emb_list.append(text_model.embed(text))
    return torch.cat(emb_list)


def mask_all_texts(texts: list):
    mask_list = []
    for text in texts:
        mask_list.append(zero_pad([1] * len(text.split())).reshape(1, -1))
    return torch.tensor(np.concatenate(mask_list))


def importance_all_texts(text_model: BertTextModel, texts: list, imp_type: str, sess: tf.Session):
    imp_list = []
    for text in texts:
        imp_list.append(zero_pad(calc_word_importance(text, text_model, imp_type, sess)).reshape(1, -1))
    return torch.tensor(np.concatenate(imp_list))


def pred_importance_all_texts(model, dataloader):
    imp_pred_list = []
    model.eval()
    for batch in dataloader:
        inp = batch[0].cuda()
        with torch.no_grad():
            imp_pred_list.append(model(inp).cpu().numpy())

    return np.concatenate(imp_pred_list)


def zero_pad(val_list, fixed_size=150):
    return np.pad(np.array(val_list), (0, fixed_size-len(val_list)))


def train_epoch(model, train_dl, val_dl, criterion, optim, device='cuda'):
    train_epoch_loss = 0
    for batch in train_dl:
        model.zero_grad()
        inp, label, mask = batch[0], batch[1], batch[2]
        inp = inp.to(device)
        label = label.float().to(device)
        mask = mask.to(device)

        pred = model(inp)
        mse = criterion(pred, label)
        loss = ((mse * mask).sum(axis=1) / mask.sum(axis=1)).mean()
        loss.backward()
        optim.step()
        train_epoch_loss += loss.detach().cpu()

    val_epoch_loss = None
    if val_dl is not None:
        val_epoch_loss = 0
        for batch in val_dl:
            inp, label, mask = batch[0], batch[1], batch[2]
            inp = inp.to(device)
            label = label.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                pred = model(inp)
                mse = criterion(pred, label)
                loss = ((mse * mask).sum(axis=1) / mask.sum(axis=1)).mean()

            val_epoch_loss += loss.detach().cpu()
    return train_epoch_loss, val_epoch_loss


def train(train_emb, train_mask, train_imp, val_emb=None, val_mask=None, val_imp=None, n_epochs=10000, device='cuda',
          bs=16, lr=1e-6, n_early_stop_rounds=10):
    # build model
    model = GenFooler()
    model.to(device)

    # build dataset & dataloader
    train_dataset = TensorDataset(train_emb, train_imp, train_mask)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs)
    val_dataloader = None
    if val_emb is not None and val_imp is not None and val_mask is not None:
        val_dataset = TensorDataset(val_emb, val_imp, val_mask)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=bs)

    # define loss & optimiser
    criterion = nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # start training
    best_epoch = None
    best_loss = 1000000
    early_stop_counter = 0
    for epoch in range(n_epochs):
        print("--------------------------------------------")
        train_loss, val_loss = train_epoch(model, train_dataloader, val_dataloader, criterion, optim, device)
        print(f"Epoch {epoch} - train loss: {train_loss}, val_loss: {val_loss}")

        # early stopping
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= n_early_stop_rounds:
            return model, best_epoch, best_loss

    return model, best_epoch, best_loss


def attack_sent(sent: str, text_model: BertTextModel, max_turns: int, sess: tf.Session, word_importance: np.array):
    word_rank = list(reversed(np.argsort(word_importance)))
    orig_pred = np.argmax(text_model.predict_proba(sent)[0])
    legal_actions = possible_actions(sent)
    word_rank = [w for w in word_rank if w in legal_actions]
    cur_sent = deepcopy(sent)
    for word_index in word_rank[:max_turns]:
        cur_sent = replace_with_synonym_greedy(cur_sent, word_index, text_model, sess)
        if text_model.predict(cur_sent)[0] != orig_pred:
            print((np.array(cur_sent.split()) != np.array(sent.split())).sum())
            return cur_sent, get_similarity([sent, cur_sent], sess)[0]

    return cur_sent, 0


if __name__ == '__main__':
    # constants
    path = 'LUNATC/data/aclImdb/imdb'
    N_SPLITS = 5
    MAX_TURNS = 300000
    SEED = 42
    model_type = 'bert'
    train_size = 750  # this lets us load the pre-computed vectors and just sub-sample them for efficiency
    attack_type = 'pwws'  # whether we are generalising textfooler or PWWS
    pre_calc_data = False  # whether to use pre-calculated data (embedding, masks and word - importance)

    train_inds = [0, 1, 3, 10, 11, 16, 20, 34, 57, 59, 71, 73, 76, 88, 90, 108, 117, 119, 121, 125, 131, 132, 134, 137, 141, 145, 147, 154, 159, 164, 168, 174, 181, 183, 186, 192, 194, 203, 206, 210, 212, 214, 215, 227, 228, 229, 235, 236, 242, 243, 244, 259, 269, 275, 279, 284, 292, 293, 300, 306, 314, 320, 322, 331, 340, 346, 350, 351, 354, 357, 361, 368, 373, 375, 377, 385, 388, 389, 394, 404, 405, 408, 411, 418, 423, 429, 432, 433, 445, 451, 452, 453, 486, 492, 495, 499, 505, 506, 508, 511, 512, 513, 528, 529, 531, 532, 539, 543, 548, 552, 554, 555, 559, 560, 564, 567, 579, 585, 592, 595, 601, 613, 616, 623, 624, 629, 630, 633, 637, 638, 644, 645, 646, 647, 658, 664, 666, 672, 675, 680, 687, 688, 709, 710, 716, 718, 719, 721, 726, 727, 729, 730, 734, 750, 752, 754, 755, 756, 769, 777, 787, 791, 798, 803, 806, 811, 819, 822, 831, 835, 836, 843, 845, 855, 858, 859, 863, 866, 872, 873, 878, 892, 898, 900, 904, 906, 911, 917, 932, 938, 943, 953, 956, 962, 966, 968, 973, 981, 988, 991, 993, 994, 995, 998, 1013, 1016, 1020, 1021, 1023, 1027, 1028, 1030, 1033, 1034, 1041, 1043, 1044, 1046, 1052, 1064, 1066, 1067, 1071, 1074, 1078, 1082, 1089, 1097, 1100, 1104, 1107, 1108, 1110, 1113, 1114, 1123, 1125, 1126, 1127, 1135, 1137, 1144, 1155, 1156, 1160, 1165, 1166, 1168, 1170, 1171, 1172, 1178, 1180, 1181, 1184, 1192, 1198, 1201, 1202, 1204, 1205, 1206, 1211, 1215, 1216, 1217, 1218, 1220, 1231, 1238, 1245, 1249, 1260, 1262, 1266, 1268, 1273, 1275, 1276, 1280, 1284, 1285, 1295, 1304, 1306, 1308, 1309, 1314, 1315, 1317, 1321, 1322, 1326, 1333, 1341, 1344, 1349, 1350, 1356, 1357, 1364, 1366, 1367, 1368, 1370, 1373, 1383, 1385, 1391, 1402, 1403, 1418, 1419, 1421, 1431, 1436, 1438, 1439, 1448, 1450, 1456, 1459, 1478, 1480, 1485, 1486, 1494, 1498, 1499, 1501, 1503, 1512, 1526, 1530, 1533, 1535, 1539, 1542, 1548, 1550, 1552, 1554, 1557, 1558, 1565, 1581, 1582, 1584, 1593, 1596, 1599, 1605, 1607, 1609, 1612, 1627, 1629, 1634, 1635, 1638, 1641, 1652, 1655, 1661, 1675, 1680, 1681, 1682, 1687, 1689, 1693, 1695, 1697, 1703, 1707, 1709, 1715, 1718, 1719, 1724, 1727, 1728, 1737, 1738, 1741, 1746, 1751, 1754, 1757, 1760, 1766, 1768, 1770, 1778, 1784, 1788, 1795, 1796, 1800, 1804, 1807, 1811, 1816, 1818, 1820, 1824, 1829, 1833, 1834, 1836, 1840, 1841, 1867, 1874, 1877, 1882, 1888, 1891, 1894, 1895, 1903, 1909, 1911, 1916, 1924, 1926, 1928, 1929, 1931, 1933, 1934, 1940, 1944, 1952, 1956, 1958, 1960, 1962, 1965, 1969, 1975, 1979, 1980, 1984, 1988, 1989, 1992, 1994, 1996, 1998, 2004, 2007, 2015, 2016, 2017, 2018, 2026, 2027, 2029, 2034, 2046, 2048, 2050, 2051, 2056, 2058, 2069, 2070, 2073, 2079, 2082, 2083, 2089, 2094, 2095, 2097, 2100, 2102, 2106, 2114, 2115, 2116, 2119, 2124, 2127, 2128, 2133, 2135, 2142, 2143, 2166, 2170, 2173, 2174, 2179, 2180, 2181, 2183, 2187, 2188, 2193, 2199, 2205, 2206, 2210, 2214, 2217, 2218, 2225, 2227, 2230, 2231, 2232, 2235, 2236, 2237, 2240, 2246, 2247, 2252, 2253, 2254, 2262, 2264, 2268, 2279, 2281, 2282, 2288, 2291, 2293, 2296, 2298, 2299, 2300, 2308, 2309, 2318, 2321, 2324, 2333, 2334, 2337, 2347, 2353, 2354, 2361, 2362, 2363, 2368, 2369, 2370, 2371, 2374, 2386, 2392, 2399, 2408, 2412, 2413, 2414, 2415, 2416, 2420, 2422, 2429, 2430, 2440, 2445, 2449, 2454, 2456, 2461, 2467, 2472, 2474, 2478, 2483, 2485, 2491, 2492, 2494, 2498, 2503, 2506, 2510, 2511, 2514, 2521, 2523, 2539, 2543, 2546, 2548, 2549, 2550, 2567, 2570, 2571, 2572, 2573, 2577, 2579, 2582, 2583, 2584, 2585, 2593, 2595, 2597, 2598, 2609, 2613, 2616, 2618, 2619, 2625, 2628, 2629, 2631, 2635, 2637, 2641, 2644, 2646, 2648, 2650, 2651, 2656, 2658, 2660, 2675, 2680, 2682, 2684, 2688, 2689, 2692, 2694, 2698, 2703, 2706, 2712, 2714, 2718, 2719, 2727, 2731, 2732, 2735, 2736, 2740, 2750, 2753, 2755, 2762, 2774, 2789, 2791, 2803, 2804, 2808, 2811, 2813, 2814, 2827, 2828, 2834, 2835, 2836, 2838, 2839, 2841, 2848, 2856, 2858, 2860, 2867, 2872, 2873, 2879, 2881, 2892, 2898, 2904, 2908, 2910, 2912, 2917, 2918, 2932, 2934, 2938, 2939, 2941, 2943, 2946, 2950, 2953, 2955, 2959, 2961, 2963, 2975, 2983, 2984, 2998, 3006, 3010, 3012, 3014, 3032, 3035, 3042, 3050, 3053, 3056, 3058, 3061, 3062, 3079, 3083, 3091, 3105, 3107, 3109, 3113, 3114, 3120, 3126, 3129, 3134, 3135, 3148, 3152, 3156, 3158, 3165, 3173, 3183, 3185, 3188, 3200, 3201, 3216, 3219, 3224, 3227, 3230, 3235]
    test_inds = [8, 13, 17, 35, 83, 103, 111, 129, 182, 184, 189, 221, 232, 237, 255, 264, 297, 328, 336, 338, 345, 356, 363, 430, 435, 436, 472, 473, 474, 477, 488, 489, 493, 521, 527, 537, 549, 589, 608, 610, 634, 642, 656, 669, 723, 740, 743, 761, 766, 768, 786, 794, 805, 812, 813, 832, 844, 861, 887, 889, 923, 961, 963, 982, 986, 992, 1024, 1050, 1068, 1077, 1086, 1090, 1129, 1145, 1158, 1167, 1175, 1226, 1235, 1242, 1261, 1269, 1311, 1318, 1342, 1351, 1363, 1369, 1392, 1430, 1443, 1447, 1454, 1466, 1468, 1473, 1488, 1508, 1509, 1515, 1520, 1604, 1622, 1626, 1630, 1632, 1647, 1663, 1672, 1684, 1698, 1706, 1721, 1731, 1739, 1747, 1769, 1794, 1823, 1825, 1872, 1884, 1900, 1901, 1902, 1917, 1938, 1941, 1948, 1950, 1955, 1967, 1971, 1976, 2005, 2009, 2013, 2038, 2041, 2042, 2045, 2088, 2093, 2098, 2099, 2101, 2108, 2122, 2134, 2145, 2190, 2197, 2198, 2222, 2226, 2238, 2259, 2272, 2283, 2284, 2303, 2310, 2316, 2326, 2346, 2372, 2373, 2375, 2378, 2404, 2410, 2431, 2460, 2469, 2542, 2551, 2558, 2561, 2564, 2568, 2592, 2604, 2611, 2621, 2623, 2626, 2636, 2645, 2665, 2670, 2700, 2713, 2724, 2733, 2752, 2756, 2773, 2819, 2842, 2843, 2852, 2854, 2890, 2891, 2896, 2901, 2906, 2911, 2920, 2921, 2925, 2977, 2989, 2994, 3027, 3031, 3052, 3069, 3086, 3117, 3136, 3151, 3159, 3174, 3180, 3213, 3220, 3225, 3228, 3237, 3243, 3249, 3252, 3260, 3263, 3270, 3283, 3285, 3286, 3288, 3291, 3301, 3302, 3308, 3309, 3311, 3312, 3316, 3323, 3324, 3326, 3329, 3330, 3333, 3335, 3336, 3341, 3343, 3344, 3345, 3351, 3352, 3353, 3355, 3356, 3358, 3359, 3368, 3371, 3373, 3374, 3377, 3378, 3388, 3392, 3394, 3397, 3402, 3403, 3404, 3405, 3406, 3408, 3413, 3421, 3422, 3427, 3429, 3436, 3438, 3440, 3441, 3448, 3451, 3453, 3458, 3462, 3470, 3478, 3485, 3486, 3491, 3492, 3505, 3507, 3511, 3517, 3518, 3522, 3523, 3525, 3527, 3528, 3533, 3536, 3537, 3543, 3544, 3551, 3555, 3560, 3564, 3567, 3572, 3573, 3574, 3584, 3588, 3589, 3590, 3592, 3595, 3596, 3597, 3598, 3601, 3603, 3616, 3617, 3622, 3629, 3636, 3641, 3648, 3649, 3651, 3661, 3666, 3667, 3670, 3672, 3676, 3677, 3679, 3680, 3683, 3686, 3689, 3690, 3691, 3694, 3696, 3702, 3705, 3706, 3708, 3710, 3711, 3712, 3713, 3714, 3715, 3717, 3720, 3724, 3725, 3727, 3733, 3740]

    if attack_type == 'pwws':
        tf_sess = tf.Session()
        tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    else:
        tf_sess = None
    os.makedirs(path + f'_genfooler_{model_type}', exist_ok=True)

    # read data
    df = pd.read_csv(path + f'_sample_{model_type}.csv')
    df.drop_duplicates('content', inplace=True)

    text_model = None
    if model_type == 'bert':
        text_model = BertTextModel(trained_model=path + '_bert.pth')
    elif model_type == 'lstm':
        text_model = WordLSTM(trained_model=path + '_word_lstm.pth')
    elif model_type == "xlnet":
        text_model = XLNetTextModel(trained_model=path + '_xlnet.pth')
    else:
        print('non-existent model type selected, please select one of ["lstm", "bert"]')
        exit()

    df_train = df.iloc[train_inds]
    df_test = df.iloc[test_inds]
    print('model and data loaded')

    full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp = \
        prepare_data(path, df_train, df_test, text_model, use_pre_calculated=pre_calc_data, imp_type=attack_type,
                     sess=tf_sess)

    full_train_emb, full_train_mask, full_train_imp = \
        full_train_emb[:train_size], full_train_mask[:train_size], full_train_imp[:train_size]
    print("data pre - processed")

    test_dataset = TensorDataset(test_emb, test_imp, test_mask)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    # fix seed
    seed_everything(SEED)
    print("seeds fixed, running CV for choosing the number of epochs")

    # train in cross-validation to choose the right number of epochs
    kfold = KFold(shuffle=True, n_splits=N_SPLITS)
    stop_epoch = []
    for train_ind, val_ind in kfold.split(full_train_emb):
        print('****************** New Fold ***************')
        train_emb, train_imp, train_mask = full_train_emb[train_ind], full_train_imp[train_ind], full_train_mask[
            train_ind]
        val_emb, val_imp, val_mask = full_train_emb[val_ind], full_train_imp[val_ind], full_train_mask[val_ind]
        _, best_epoch, _ = train(train_emb, train_mask, train_imp, val_emb, val_mask, val_imp)
        stop_epoch.append(best_epoch)
    print(f"CV finished, number of epochs chosen: {int(sum(stop_epoch)/len(stop_epoch))}")

    # train the final model on all the training data for the right number of epochs
    model, _, _ = train(full_train_emb, full_train_mask, full_train_imp, n_epochs=int(sum(stop_epoch)/len(stop_epoch)),
                        n_early_stop_rounds=100000000000)
    print("final model finished training!")

    # predict the word importance on the test texts
    test_imp_pred = pred_importance_all_texts(model, test_dataloader)
    print("Word importance predicted, performing the attack")

    # attack using the predicted importance
    tf_sess = tf.Session()
    tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    sim_scores = []
    best_sents = []
    for i, text in enumerate(df_test.content.values.tolist()):
        pred_imp = test_imp_pred[i][:len(text.split())]
        best_sent, sim_score = attack_sent(text, text_model, MAX_TURNS, tf_sess, pred_imp)
        print(f"------{i}--- score: {sim_score}-----")
        best_sents.append(best_sent)
        sim_scores.append(sim_score)

    # save results
    print("Attack finished! saving results...")
    df_test['max_score'] = sim_scores
    df_test['best_sent'] = best_sents
    df_test.to_csv(path + f'_genfooler_{model_type}/attack_{attack_type}{train_size}_{SEED}.csv', index=False)
    torch.save(model.state_dict(), path + f'_genfooler_{model_type}/model_{attack_type}{train_size}_{SEED}.pth')


