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
from src.TextModels.E2EBert import E2EBertTextModel

# importing the textfooler word_importance
from src.Agents.textfooler_attack import calc_word_importance

# fix randomness
from src.Agents.utils.optim_utils import seed_everything


class GenFooler(nn.Module):
    def __init__(self):
        super(GenFooler, self).__init__()
        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 400)

        self.fc21 = nn.Linear(400, 400)
        self.fc22 = nn.Linear(400, 400)
        self.fc23 = nn.Linear(400, 400)
        self.fc24 = nn.Linear(400, 400)

        self.fc3 = nn.Linear(400, 150)

    def forward(self, x):
        x = x.view(-1, 768)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc21(x))
        x = torch.tanh(self.fc22(x))
        x = torch.tanh(self.fc23(x))
        x = torch.tanh(self.fc24(x))
        x = self.fc3(x)
        return x


def prepare_data(path, df_train, df_test, text_model, use_pre_calculated=False):
    if use_pre_calculated:
        with open(path + '_genfooler/data.pkl', 'rb') as f:
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
    full_train_imp = importance_all_texts(text_model, df_train.content.values.tolist())
    test_imp = importance_all_texts(text_model, df_test.content.values.tolist())
    print("Importance calculated for all texts")

    with open(path + '_genfooler/data.pkl', 'wb') as f:
        pickle.dump((full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp), f)

    return full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp


def embed_all_texts(text_model: E2EBertTextModel, texts: list):
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


def importance_all_texts(text_model:E2EBertTextModel, texts: list):
    imp_list = []
    for text in texts:
        imp_list.append(zero_pad(calc_word_importance(text, text_model)).reshape(1, -1))
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
        label = label.to(device)
        mask = mask.to(device)

        pred = model(inp)
        mse = criterion(pred, label)
        loss = ((mse * mask).sum(axis=1) / mask.sum()).mean()
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
                loss = ((mse * mask).sum(axis=1) / mask.sum()).mean()

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


def attack_sent(sent: str, text_model: E2EBertTextModel, max_turns: int, sess: tf.Session, word_importance: np.array):
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
    path = '../../data/aclImdb/imdb'
    N_SPLITS = 5
    os.makedirs(path + '_genfooler', exist_ok=True)

    # read data
    df = pd.read_csv(path + '_sample.csv')
    df.drop_duplicates('content', inplace=True)
    train_inds = [0, 1, 3, 10, 11, 16, 20, 34, 57, 59, 71, 73, 76, 88, 90, 108, 117, 119, 121, 125, 131, 132, 134, 140,
                  144, 146, 153, 158, 163, 167, 173, 182, 185, 191, 193, 202, 205, 209, 211, 213, 214, 226, 227, 228,
                  234, 235, 241, 242, 243, 254, 258, 268, 274, 278, 283, 291, 292, 299, 313, 319, 321, 330, 339, 345,
                  349, 350, 353, 356, 360, 367, 372, 376, 384, 387, 388, 393, 403, 404, 407, 410, 417, 422, 431, 432,
                  444, 450, 451, 452, 485, 491, 494, 498, 504, 505, 507, 510, 511, 512, 527, 528, 530, 531, 538, 542,
                  547, 551, 553, 554, 558, 559, 563, 578, 584, 591, 594, 612, 615, 618, 622, 623, 628, 629, 632, 636,
                  637, 643, 644, 646, 657, 663, 665, 671, 674, 679, 686, 687, 708, 709, 715, 717, 718, 720, 725, 726,
                  728, 729, 733, 739, 749, 751, 753, 754, 755, 768, 776, 786, 790, 797, 802, 805, 810, 818, 821, 830,
                  834, 835, 842, 844, 854, 857, 858, 862, 865, 871, 872, 877, 891, 897, 899, 903, 905, 910, 916, 931,
                  937, 942, 952, 955, 960, 961, 965, 967, 972, 980, 987, 988, 991, 993, 994, 995, 998, 1013, 1016, 1020,
                  1021, 1023, 1027, 1028, 1030, 1033, 1034, 1041, 1043, 1044, 1046, 1050, 1052, 1064, 1066, 1067, 1068,
                  1071, 1074, 1078, 1082, 1089, 1097, 1100, 1104, 1107, 1108, 1110, 1113, 1123, 1125, 1126, 1127, 1135,
                  1137, 1144, 1155, 1156, 1165, 1166, 1167, 1168, 1170, 1171, 1172, 1178, 1180, 1181, 1192, 1198, 1201,
                  1202, 1204, 1205, 1206, 1211, 1215, 1216, 1217, 1218, 1220, 1231, 1238, 1245, 1249, 1260, 1262, 1266,
                  1268, 1269, 1273, 1275, 1276, 1280, 1284, 1285, 1295, 1304, 1306, 1308, 1309, 1314, 1315, 1317, 1321,
                  1322, 1326, 1333, 1341, 1349, 1350, 1356, 1357, 1363, 1364, 1366, 1367, 1368, 1370, 1373, 1383, 1385,
                  1391, 1402, 1403, 1418, 1419, 1421, 1431, 1436, 1438, 1439, 1448, 1450, 1456, 1459, 1478, 1480, 1485,
                  1486, 1488, 1494, 1498, 1499, 1501, 1503, 1512, 1526, 1530, 1533, 1539, 1542, 1548, 1550, 1552, 1554,
                  1557, 1558, 1565, 1581, 1582, 1584, 1593, 1596, 1599, 1605, 1607, 1609, 1612, 1627, 1629, 1634, 1635,
                  1638, 1641, 1652, 1655, 1661, 1675, 1680, 1681, 1682, 1687, 1689, 1693, 1695, 1697, 1698, 1703, 1707,
                  1709, 1715, 1718, 1719, 1724, 1727, 1728, 1736, 1737, 1740, 1745, 1750, 1753, 1756, 1759, 1765, 1767,
                  1769, 1777, 1783, 1787, 1794, 1795, 1799, 1803, 1806, 1810, 1815, 1817, 1819, 1822, 1823, 1828, 1832,
                  1833, 1835, 1839, 1840, 1866, 1873, 1876, 1881, 1883, 1887, 1890, 1893, 1894, 1902, 1908, 1910, 1915,
                  1923, 1925, 1927, 1928, 1930, 1932, 1933, 1939, 1943, 1951, 1955, 1957, 1959, 1961, 1964, 1968, 1974,
                  1978, 1979, 1983, 1987, 1988, 1991, 1993, 1995, 1997, 2003, 2004, 2006, 2014, 2015, 2016, 2017, 2025,
                  2026, 2028, 2033, 2045, 2047, 2049, 2050, 2055, 2057, 2068, 2069, 2072, 2078, 2081, 2082, 2088, 2093,
                  2094, 2096, 2098, 2099, 2101, 2105, 2113, 2114, 2115, 2118, 2123, 2126, 2127, 2132, 2134, 2141, 2142,
                  2144, 2165, 2169, 2172, 2173, 2178, 2180, 2182, 2186, 2192, 2198, 2205, 2209, 2213, 2216, 2217, 2224,
                  2225, 2227, 2230, 2231, 2232, 2235, 2236, 2237, 2240, 2246, 2247, 2252, 2253, 2254, 2262, 2264, 2268,
                  2279, 2281, 2282, 2288, 2291, 2293, 2296, 2298, 2299, 2300, 2308, 2309, 2318, 2321, 2324, 2333, 2334,
                  2337, 2347, 2353, 2354, 2361, 2362, 2363, 2368, 2369, 2370, 2371, 2374, 2386, 2392, 2399, 2404, 2408,
                  2412, 2413, 2414, 2415, 2416, 2420, 2422, 2429, 2430, 2440, 2445, 2449, 2453, 2455, 2460, 2466, 2468,
                  2471, 2473, 2477, 2482, 2484, 2490, 2491, 2493, 2497, 2502, 2505, 2509, 2510, 2513, 2520, 2522, 2538,
                  2542, 2545, 2547, 2548, 2549, 2560, 2566, 2569, 2570, 2571, 2572, 2576, 2578, 2581, 2582, 2583, 2584,
                  2592, 2594, 2595, 2596, 2607, 2611, 2614, 2616, 2617, 2623, 2624, 2626, 2627, 2629, 2633, 2635, 2639,
                  2642, 2644, 2646, 2648, 2654, 2656, 2658, 2673, 2678, 2682, 2686, 2687, 2690, 2692, 2696, 2701, 2704,
                  2710, 2712, 2716, 2717, 2725, 2729, 2730, 2733, 2734, 2738, 2748, 2750, 2751, 2753, 2760, 2772, 2787,
                  2789, 2801, 2802, 2809, 2811, 2812, 2825, 2826, 2832, 2833, 2834, 2836, 2837, 2839, 2846, 2854, 2856,
                  2858, 2865, 2870, 2871, 2877, 2879, 2890, 2896, 2902, 2906, 2908, 2910, 2915, 2916, 2930, 2932, 2936,
                  2937, 2939, 2941, 2944, 2948, 2951, 2953, 2959, 2961, 2973, 2981, 2982, 2992, 2996, 3004, 3010, 3012,
                  3030, 3033, 3040, 3048, 3051, 3054, 3056, 3059, 3060, 3077, 3081, 3089, 3103, 3105, 3107, 3111, 3112,
                  3118, 3124, 3127, 3132, 3133, 3146, 3150, 3154, 3156, 3163, 3171, 3178, 3183, 3186, 3198, 3199, 3214,
                  3217, 3222]
    test_inds = [3225, 3228, 3233, 3235, 3241, 3247, 3250, 3258, 3261, 3283, 3284, 3286, 3289, 3299, 3300, 3306, 3307,
                 3310, 3314, 3321, 3322, 3327, 3328, 3331, 3333, 3334, 3339, 3341, 3342, 3343, 3351, 3353, 3354, 3356,
                 3357, 3366, 3369, 3371, 3372, 3375, 3390, 3395, 3400, 3401, 3402, 3404, 3411, 3419, 3420, 3425, 3434,
                 3438, 3439, 3446, 3449, 3451, 3456, 3468, 3476, 3489, 3503, 3505, 3509, 3516, 3520, 3523, 3531, 3534,
                 3535, 3541, 3542, 3553, 3562, 3570, 3571, 3572, 3582, 3586, 3588, 3590, 3593, 3594, 3595, 3596, 3599,
                 3601, 3614, 3615, 3620, 3634, 3647, 3649, 3668, 3670, 3674, 3677, 3678, 3681, 3684, 3687, 3688, 3689,
                 3694, 3703, 3704, 3706, 3708, 3709, 3710, 3711, 3712, 3715, 3718, 3723, 3731, 3738]

    text_model = E2EBertTextModel(trained_model=path + 'e2e_bert.pth')
    df_train = df.iloc[train_inds]
    df_test = df.iloc[test_inds]
    print('model and data loaded')

    full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp = \
        prepare_data(path, df_train, df_test, text_model, use_pre_calculated=True)
    print("data pre - processed")

    test_dataset = TensorDataset(test_emb, test_imp, test_mask)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    # fix seed
    seed_everything(42)
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
        best_sent, sim_score = attack_sent(text, text_model, 30, tf_sess, pred_imp)
        print(f"------{i}--- score: {sim_score}-----")
        best_sents.append(best_sent)
        sim_scores.append(sim_score)

    # save results
    print("Attack finished! saving results...")
    df_test['max_score'] = sim_scores
    df_test['best_sent'] = best_sents
    df_test.to_csv(path + '_genfooler/attack.csv', index=False)
    torch.save(model.state_dict(), path + '_genfooler/model.pth')


