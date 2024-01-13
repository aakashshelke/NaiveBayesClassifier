import numpy as np
import pandas as pd
import sys

def probability(x, mean, var):
    expo = np.exp(-((x - mean) ** 2 / (2 * var)))
    return (1 / (np.sqrt(2 * np.pi * var))) * expo

def naive_bayes(att_1, att_2, target):
    data = pd.DataFrame({
        'att_1': att_1,
        'att_2': att_2,
        'target': target
    })

    data_class_A = data[data['target'] == 'A']
    data_class_B = data[data['target'] == 'B']

    prob_class_A = len(data_class_A) / len(data)
    prob_class_B = len(data_class_B) / len(data)

    mean_class_A_1, var_class_A_1 = data_class_A['att_1'].mean(), data_class_A['att_1'].var(ddof=1)
    mean_class_B_1, var_class_B_1 = data_class_B['att_1'].mean(), data_class_B['att_1'].var(ddof=1)

    mean_class_A_2, var_class_A_2 = data_class_A['att_2'].mean(), data_class_A['att_2'].var(ddof=1)
    mean_class_B_2, var_class_B_2 = data_class_B['att_2'].mean(), data_class_B['att_2'].var(ddof=1)

    n = 0
    for i, row in data.iterrows():
        gauss_class_A_1 = probability(row['att_1'], mean_class_A_1, var_class_A_1)
        gauss_class_B_1 = probability(row['att_1'], mean_class_B_1, var_class_B_1)
        gauss_class_A_2 = probability(row['att_2'], mean_class_A_2, var_class_A_2)
        gauss_class_B_2 = probability(row['att_2'], mean_class_B_2, var_class_B_2)

        p_class_A = prob_class_A * gauss_class_A_1 * gauss_class_A_2
        p_class_B = prob_class_B * gauss_class_B_1 * gauss_class_B_2

        if p_class_A > p_class_B:
            pred_class = "A"
        elif p_class_B > p_class_A:
            pred_class = "B"
        else:
            pass

        # if row['target'] != pred_class:
        #     n += 1

    print(mean_class_A_1, var_class_A_1, mean_class_A_2, var_class_A_2, prob_class_A, sep=",")
    print(mean_class_B_1, var_class_B_1, mean_class_B_2, var_class_B_2, prob_class_B, sep=",")
    print(n)


if __name__ == "__main__":
    expected_args = ["--data"]
    arg_len = len(sys.argv)
    info = []

    for i in range(len(expected_args)):
        for j in range(1, len(sys.argv)):
            if expected_args[i] == sys.argv[j] and sys.argv[j + 1]:
                info.append(sys.argv[j + 1])

    data = pd.read_csv(info[0], header=None)
    arr = np.array(data)
    target = arr[:, 0]
    att_1, att_2 = arr[:, 1], arr[:, 2]
    naive_bayes(att_1, att_2, target)
