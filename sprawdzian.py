from challenge_functions.src.score_submission import score_subm as ss
from challenge_functions.src.verify_submission import verify_subm as vs
from challenge_functions.src.baseline_algorithm import rec_popular as rp
import pandas as pd

subm_csv = '../data/submission_popular.csv'
gt_csv = '../data/groundTruth.csv'
test_csv = '../data/test.csv'
data_path = '../dane/'

def baseline():
    rp.main()

def verify():
    vs.main()


def score():
    ss.main()

if __name__ == '__main__':

    #baseline()
    #verify()
    score()
