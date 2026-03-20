import datetime as dt
import itertools
import os
import pathlib
import tempfile

import pandas as pd
from p_tqdm import p_map

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
TMP_DIR = pathlib.Path(tempfile.gettempdir()) / "abides_version_testing"

os.chdir(ROOT_PATH)
import sys

sys.path.insert(0, str(ROOT_PATH))

import version_testing.runasof as runasof


def get_paths(parameters):
    specific_path = f'{parameters["new"]["config"]}/{parameters["shared"]["end-time"].replace(":", "-")}/{parameters["shared"]["seed"]}'  # can add as many as there are parameters
    specific_path_underscore = f'{parameters["new"]["config"]}_{parameters["shared"]["end-time"].replace(":", "-")}_{parameters["shared"]["seed"]}'  # TODO: maybe something better
    return specific_path, specific_path_underscore


def run_test(test_):
    parameters, old_new_flag = test_
    specific_path, specific_path_underscore = get_paths(parameters)

    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d%H%M%S")

    time = runasof.run_command(
        parameters["command"][old_new_flag],
        commit_sha=parameters[old_new_flag]["sha"],
        specific_path_underscore=specific_path_underscore,
        git_path=str(ROOT_PATH),
        old_new_flag=old_new_flag,
        pass_logdir_sha=(
            "--log_dir",
            lambda x: str(TMP_DIR / f"{old_new_flag}_{stamp}" / x / specific_path),
        ),
    )

    output = {}

    output["sha"] = parameters[old_new_flag]["sha"]
    output["config"] = parameters[old_new_flag]["config"]
    output["end-time"] = parameters["shared"]["end-time"]
    output["seed"] = parameters["shared"]["seed"]
    output["time"] = time
    if parameters["with_log"]:
        path_to_ob = str(
            TMP_DIR
            / f"{old_new_flag}_{stamp}"
            / parameters[old_new_flag]["sha"]
            / specific_path
            / "ORDERBOOK_ABM_FULL.bz2"
        )
    else:
        path_to_ob = "no_log"
    output["path_to_ob"] = path_to_ob
    output["flag"] = old_new_flag

    return output


def compute_ob(path_old, path_new):
    ob_old = pd.read_pickle(path_old)
    ob_new = pd.read_pickle(path_new)
    if ob_old.equals(ob_new):
        return 0
    else:
        return 1


def run_tests(LIST_PARAMETERS, varying_parameters):

    old_new_flags = ["old", "new"]
    tests = list(itertools.product(LIST_PARAMETERS, old_new_flags))

    # test_ = tests[0]
    # run_test(test_)
    outputs = p_map(run_test, tests)

    df = pd.DataFrame(outputs)

    df_old = df[df["flag"] == "old"]
    df_new = df[df["flag"] == "new"]

    print(f"THERE ARE {len(df_new)} TESTS RESULTS.")

    if LIST_PARAMETERS[0]["with_log"]:
        path_olds = list(df_old["path_to_ob"])
        path_news = list(df_new["path_to_ob"])

        # compute_ob(path_olds[0], path_news[0])

        ob_comps = p_map(compute_ob, path_olds, path_news)

        if sum(ob_comps) == 0:
            print("ALL TESTS ARE SUCCESS!")
        else:
            print(f"ALERT: {sum(ob_comps)}TEST FAILURE")
    df_old = df_old[varying_parameters + ["seed", "time"]].set_index(
        varying_parameters + ["seed"]
    )
    df_new = df_new[varying_parameters + ["seed", "time"]].set_index(
        varying_parameters + ["seed"]
    )
    df_diff = df_old - df_new  # /df_old
    df_results = df_diff.groupby(["config", "end-time"])["time"].describe()[
        ["mean", "std"]
    ]

    df_diff_pct = 100 * (df_old - df_new) / df_old
    df_results_pct = df_diff_pct.groupby(["config", "end-time"])["time"].describe()[
        ["mean", "std"]
    ]
    print("*********************************************")
    print("*********************************************")
    print("OLD RUNNING TIME")
    # with pd.option_context('display.float_format', '{:0.2f}'.format):
    print(df_old.groupby(["config", "end-time"])["time"].describe()[["mean", "std"]])
    print("*********************************************")
    print("*********************************************")
    print("NEW RUNNING TIME")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(
            df_new.groupby(["config", "end-time"])["time"].describe()[["mean", "std"]]
        )
    print("*********************************************")
    print("*********************************************")
    print("TIME DIFFERENCE in seconds")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        df_results["mean"] = df_results["mean"].dt.total_seconds()
        df_results["std"] = df_results["std"].dt.total_seconds()
    print(df_results)
    print("*********************************************")
    print("*********************************************")
    print("TIME DIFFERENCE in %")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(df_results_pct)
