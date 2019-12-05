from challenge_functions.src.score_submission import score_subm as ss
from challenge_functions.src.verify_submission import verify_subm as vs
from challenge_functions.src.baseline_algorithm import rec_popular as rp
from challenge_functions.src.new_algorithm import rec_popular as na
import pandas as pd
import numpy as np


subm_csv = 'C:/Users/Konrad/PycharmProjects/podzial/dane/submission_popular.csv'
gt_csv = 'C:/Users/Konrad/PycharmProjects/podzial/dane/test_whole.csv'
test_csv = 'C:/Users/Konrad/PycharmProjects/podzial/dane/test.csv'
data_path = '../dane/'

def newalgorithm():
    na.main()


def baseline():
    rp.main()


def verify():
    vs.main()


def score():
    ss.main()


def remove_duplicated_sessions(df):
    session_list = []
    final_df = pd.DataFrame()
    d = len(df)
    j = 0
    while j < d:
        if df['session_id'].values[j] not in session_list:
            session_list.append(df['session_id'].values[j])
            final_df = final_df.append(df.iloc[[j]])
            while j+1 < d and df['step'].values[j+1] != 1:
                j = j+1
                final_df = final_df.append(df.iloc[[j]])
        else:
            while j+1 < d and df['step'].values[j+1] != 1:
                j = j+1
        #print(str(j)+",")
        j = j+1
    return final_df.reset_index()


def remove_duplicated_sessions2(df):
    session_list = []
    idx_step_1 = np.array(df[df.step == 1].index)
    idx_last_step = [x - 1 for x in idx_step_1[1:]]
    idx_last_step.append(len(df)-1)
    d = len(idx_step_1)
    j = 0
    while j < d:
        if df['session_id'].values[idx_step_1[j]] not in session_list:
            session_list.append(df['session_id'].values[idx_step_1[j]])
            j = j + 1
        else:
            df.drop(df.index[idx_step_1[j]:idx_last_step[j]+1], inplace=True)
            df.reset_index(drop=True, inplace=True)
            idx_step_1 = np.array(df[df.step == 1].index)
            idx_last_step = [x - 1 for x in idx_step_1[1:]]
            idx_last_step.append(df.index[-1])
            d = len(idx_step_1)
    return df


def remove_duplicated_sessions3(df):
    session_list = []
    indexes_to_remove = []
    idx_step_1 = np.array(df[df.step == 1].index)
    idx_last_step = [x - 1 for x in idx_step_1[1:]]
    idx_last_step.append(len(df)-1)
    d = len(idx_step_1)
    print("step1 len: "+str(len(idx_step_1)))
    j = 0
    while j < d:
        if df['session_id'].values[idx_step_1[j]] not in session_list:
            session_list.append(df['session_id'].values[idx_step_1[j]])
        else:
            indexes_to_remove.extend(df.index[idx_step_1[j]:idx_last_step[j]+1])
        j += 1
    print("to remove len: "+str(len(indexes_to_remove)))
    df = df[~df.index.isin(indexes_to_remove)]
    return df.reset_index(drop=True)


def split_data():
    df = pd.read_csv("C:/Users/Konrad/Downloads/trivagoRecSysChallengeData2019_v2/train.csv")
    print(len(df))
    #df = remove_duplicated_sessions3(df)
    print(len(df))

    unique_sessions = df['session_id'].unique()
    train_set_last_session = unique_sessions[int(len(unique_sessions)*0.8)]
    train_set_len = int(df[df.session_id==train_set_last_session][-1:].index[0])

    if df['step'].values[train_set_len] == 1 and df['step'].values[train_set_len+1] == 2:
        train_set_len = train_set_len+1
    while df['step'].values[train_set_len] != 1:
        train_set_len = train_set_len+1

    df_train = df[:train_set_len]

    df_whole_test_set = df[train_set_len:]
    df_test = df_whole_test_set.copy()
    df_labels = df_test[['session_id', 'action_type', 'reference']].copy()

    last_clickout = -1
    d = range(len(df_test))

    for j in d:
        if df_test['action_type'].values[j] == 'clickout item':
            last_clickout = j
        if j < len(df_test)-1 and df_test['session_id'].values[j] != df_test['session_id'].values[j + 1] and last_clickout >= 0:
            df_test['reference'].values[last_clickout] = np.nan

    df_labels.to_csv("C:/Users/Konrad/PycharmProjects/podzial/dane/labels_gener.csv")
    df_train.to_csv("C:/Users/Konrad/PycharmProjects/podzial/dane/train.csv")
    df_test.to_csv("C:/Users/Konrad/PycharmProjects/podzial/dane/test.csv")
    df_whole_test_set.to_csv("C:/Users/Konrad/PycharmProjects/podzial/dane/test_whole.csv")


if __name__ == '__main__':
    split_data()
    #newalgorithm()
    #baseline()
    #verify()
    #score()
