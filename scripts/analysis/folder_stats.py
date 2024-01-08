import enum
import os
import folderstats
import tqdm
import multiprocessing
import pandas as pd
from textblob import Word
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

#from utilities.utilities import mergeDictionary

def mergeDictionary():
    pass

CLONED_REPO_DIR = "/mnt/volume1/mlexpmining/cloned_repos"
FOLDER_STATS_OUTPUT_DIR = "/mnt/volume1/mlexpmining/folder_stats"
# folder_stats_result_dir = os.path.join(os.path.abspath("."), "results_test/8-project_stats_analysis")
n_processes = 6

FOLDER_STATS_DIR = "/mnt/volume1/mlexpmining/folder_stats"
RESULT_DIR = "results_test/folder_contents"


# def folder_stats():
#     for i, folder in enumerate(os.scandir(CLONED_REPO_DIR)):
#         foldername = folder.name
#         abs_path = f"{CLONED_REPO_DIR}/{foldername}"
#         print(f"Handling repo {i} : {abs_path}")
#         stats_df = folderstats.folderstats(abs_path, ignore_hidden=True)
#         os.makedirs(FOLDER_STATS_OUTPUT_DIR, exist_ok=True)
#         stats_df.to_csv(
#             f"{FOLDER_STATS_OUTPUT_DIR}/{foldername}.csv", index=False)


def folder_stats(foldername):

    # foldername = folder.name
    stats_folder = f"{FOLDER_STATS_OUTPUT_DIR}/{foldername}.csv"
    if not os.path.exists(stats_folder):
        abs_path = f"{CLONED_REPO_DIR}/{foldername}"
        print(f"Handling repo : {abs_path}")
        try:
            stats_df = folderstats.folderstats(abs_path, ignore_hidden=True)
            stats_df.to_csv(stats_folder, index=False)
        except:
            pass


"""Generates total counts and per project counts for each folder name"""


def folder_name():
    total_count_dict = {}
    proj_count_dict = {}
    folders = os.listdir(FOLDER_STATS_DIR)

    for i, filename in enumerate(folders):
        try:
            df = pd.read_csv(f"{FOLDER_STATS_DIR}/{filename}")
        except:
            continue
        print(f"{i:5d} / {len(folders)} : {filename}")
        df = df[df["folder"] == True]  # Parse only folders
        df.name = df.name.astype(str)
        temp_proj_dict = {}
        for index, row in df.iterrows():
            key = Word(row["name"].lower()).singularize()
            temp_proj_dict[key] = temp_proj_dict.get(key, 0) + 1

        total_count_dict = mergeDictionary(
            temp_proj_dict, total_count_dict, sum_values=True
        )
        # set all values to 1
        temp_proj_dict.update({}.fromkeys(temp_proj_dict, 1))
        proj_count_dict = mergeDictionary(
            temp_proj_dict, proj_count_dict, sum_values=True
        )

    folder_name_df = pd.DataFrame.from_dict(
        mergeDictionary(total_count_dict, proj_count_dict, sum_values=False),
        orient="index",
        columns=["total_count", "proj_count"],
    )
    folder_name_df["folder_name"] = folder_name_df.index
    folder_name_df = folder_name_df[[
        "folder_name", "proj_count", "total_count"]]
    folder_name_df = folder_name_df.sort_values(
        by=["proj_count"], ascending=False)

    folder_name_df.to_csv(
        f"{RESULT_DIR}/1-folders_n_file-extensions/folder_name_count.csv", index=False
    )


"""Generates total counts and per project counts for each file extension"""


def file_type():
    total_count_dict = {}
    proj_count_dict = {}
    folders = os.listdir(FOLDER_STATS_DIR)

    for i, filename in enumerate(folders):
        try:
            df = pd.read_csv(f"{FOLDER_STATS_DIR}/{filename}")
        except:
            continue
        print(f"{i:5d} / {len(folders)} : {filename}")
        df.extension = df.extension.astype(str)
        temp_proj_dict = {}
        for index, row in df.iterrows():
            key = row["extension"].lower()
            temp_proj_dict[key] = temp_proj_dict.get(key, 0) + 1

        total_count_dict = mergeDictionary(
            temp_proj_dict, total_count_dict, sum_values=True
        )
        # set all values to 1
        temp_proj_dict.update({}.fromkeys(temp_proj_dict, 1))
        proj_count_dict = mergeDictionary(
            temp_proj_dict, proj_count_dict, sum_values=True
        )

    file_type_df = pd.DataFrame.from_dict(
        mergeDictionary(total_count_dict, proj_count_dict, sum_values=False),
        orient="index",
        columns=["total_count", "proj_count"],
    )
    file_type_df["file_type"] = file_type_df.index

    file_type_df = file_type_df[["file_type", "proj_count", "total_count"]]
    file_type_df = file_type_df.sort_values(by=["proj_count"], ascending=False)

    file_type_df.to_csv(
        f"{RESULT_DIR}/1-folders_n_file-extensions/file_type_count.csv", index=False
    )


def folder_stats_parallel():
    folders = os.listdir(CLONED_REPO_DIR)
    os.makedirs(FOLDER_STATS_OUTPUT_DIR, exist_ok=True)
    pool = multiprocessing.Pool(processes=n_processes)
    # tqdm.tqdm(pool.imap_unordered(
    #     folder_stats, os.scandir(CLONED_REPO_DIR)), total=40000)
    # tqdm.tqdm(pool.imap_unordered(folder_stats, folders), total=len(folders))
    for _ in tqdm.tqdm(pool.imap_unordered(folder_stats, folders), total=len(folders)):
        pass


def folder_stats_single():
    os.makedirs(FOLDER_STATS_OUTPUT_DIR, exist_ok=True)

    for i, folder in enumerate(os.scandir(CLONED_REPO_DIR)):
        print(i)
        folder_stats(folder)


"""Collect all contents found in each of the 100 most common folders"""


def folder_contents(folder_name):
    temp_folder_df = pd.DataFrame()
    folder_stats_dirs = os.listdir(FOLDER_STATS_DIR)
    for j, filename in enumerate(folder_stats_dirs):
        try:
            df_project_stats = pd.read_csv(
                f"{FOLDER_STATS_DIR}/{filename}")
        except (pd.errors.ParserError, IsADirectoryError):
            continue

        # print(f"{i}/100 folder names | {j}/{len(folder_stats_dirs)} repos")

        df_folders = df_project_stats[df_project_stats["folder"] == True]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_folders.name = df_folders.name.astype(str)
            df_folders.name = df_folders.name.apply(
                lambda x: Word(x.lower()).singularize()
            )
        df_folder = df_folders[df_folders["name"] == folder_name]
        for index, row in df_folder.iterrows():
            folder_id = row["id"]
            df_folder_children = df_project_stats[
                df_project_stats["parent"] == folder_id
            ]
            temp_folder_df = pd.concat(
                [temp_folder_df, df_folder_children], axis=0)
            temp_folder_df.drop_duplicates(
                ["extension", "parent", "depth"], keep="last", inplace=True
            )

    folder_path = os.path.join(
        f"{RESULT_DIR}/2-top_100_folder_contents", f"{folder_name}_content.csv"
    )
    temp_folder_df.to_csv(folder_path, index=False)


def get_folder_contents():
    """Save the folder contents for the first 100 folders in .csv files. (Simple implementation)"""

    folder_name_path = os.path.join(
        f"{RESULT_DIR}/1-folders_n_file-extensions", "folder_name_count.csv"
    )
    top_100_folder_names = pd.read_csv(
        folder_name_path).loc[30:100, "folder_name"].values

    for i, folder_name in enumerate(top_100_folder_names):
        print(f'{i:2d}/{len(top_100_folder_names)}')
        folder_contents(folder_name)


def get_folder_contents_parallel():
    """Save the folder contents for the first 100 folders in .csv files. (Parallelized implementation)"""

    folder_name_path = os.path.join(
        f"{RESULT_DIR}/1-folders_n_file-extensions", "folder_name_count.csv"
    )
    top_100_folder_names = pd.read_csv(
        folder_name_path).loc[30:100, "folder_name"].values
    pool = multiprocessing.Pool(processes=n_processes)
    for _ in tqdm.tqdm(pool.imap_unordered(folder_contents, top_100_folder_names), total=len(top_100_folder_names)):
        pass


"""Plot the frequency of file extensions in folders"""


def plot_folder_content(filter_percentile):
    input_folder = f"{RESULT_DIR}/2-top_100_folder_contents"
    for filename in os.listdir(input_folder):
        pass
        df_folder_contents = pd.read_csv(f"{input_folder}/{filename}")
        print(filename)
        f = filename.split(".")[0]
        ext_counts_df = (
            df_folder_contents.groupby(["extension"])
            .size()
            .sort_values(ascending=False)
            .reset_index(name="count")
        )
        if ext_counts_df.empty:
            continue
        ext_counts_df = ext_counts_df.query(
            f"count > count.quantile({filter_percentile})"
        )
        if ext_counts_df.empty:
            continue
        # print(ext_counts_df)
        # print(ext_counts_df.head())
        # print(ext_counts_df.tail())
        ext_counts_df.plot(
            x="extension",
            y="count",
            kind="bar",
            color="C1",
            title=f"{f}: count of unique file extension in each folder and depth",
        )
        folder_path = os.path.join(RESULT_DIR, f"{folder_name}_content.csv")

        figure_file = os.path.join(
            f"{RESULT_DIR}/3-top_100_folder_content_plots/{filter_percentile}",
            f'{filename.split(".")[0]}.png',
        )

        plt.savefig(
            figure_file,
            bbox_inches="tight",
        )


"""Collect and structure (for further analysis) the depth where file extensions are found"""


def extension_depth():

    folder_name_path = os.path.join(
        f"{RESULT_DIR}/1-folders_n_file-extensions", "file_type_count.csv"
    )
    top_100_file_types = pd.read_csv(
        folder_name_path).loc[:100, "file_type"].values

    print(top_100_file_types)

    depth_dict = {}

    input_folder = f"{RESULT_DIR}/2-top_100_folder_contents"

    for filename in os.listdir(input_folder):
        df_folder_contents = pd.read_csv(f"{input_folder}/{filename}")
        print(filename)
        for index, row in df_folder_contents.iterrows():
            key = row["extension"]
            if (
                key in top_100_file_types
            ):  # consider only extenstion from the top 100 list
                if key in depth_dict:
                    depth_dict[key].append(row["depth"])
                else:
                    depth_dict[key] = [row["depth"]]
    # print(depth_dict)
    depth_df = pd.DataFrame()
    for d in depth_dict:
        # if len(depth_dict[d]) > 100:
        depth_df[d] = pd.Series(depth_dict[d])
    # else:
    #    print(f"Skipping {d}")

    depth_df.to_csv(
        f"{RESULT_DIR}/4-file_ext_depth/extention_depths.csv", index=False)


def extension_depth_result():
    input = f"{RESULT_DIR}/4-file_ext_depth/extention_depths.csv"
    df = pd.read_csv(input)

    df_depth_name = pd.Series(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
        ],
        name="depth",
    ).to_frame()
    df_depth_name.depth = df_depth_name.depth.astype(int)
    # print(df_depth_name)

    filtered_n_grouped_extensions = [
        "bat",
        "bib",
        "bin",
        "c",
        "cc",
        "cfg",
        "conf",
        "config",
        "cpp",
        "csv",
        "dat",
        "data",
        "db",
        "docx",
        "exe",
        "gradle",
        "h",
        "h5",
        "hdf5",
        "hpp",
        "in",
        "ini",
        "ipynb",
        "jar",
        "java",
        "joblib",
        "jpeg",
        "jpg",
        "json",
        "log",
        "m",
        "mat",
        "md",
        "meta",
        "mp3",
        "mp4",
        "names",
        "npy",
        "out",
        "pb",
        "pbtxt",
        "pdf",
        "pickle",
        "pkl",
        "plist",
        "png",
        "pptx",
        "properties",
        "proto",
        "ps1",
        "pt",
        "pth",
        "pxd",
        "py",
        "pyc",
        "pyx",
        "r",
        "rst",
        "sav",
        "sh",
        "sql",
        "svg",
        "tex",
        "toml",
        "ts",
        "tsv",
        "txt",
        "wav",
        "xls",
        "xlsx",
        "xml",
        "yaml",
        "yml",
    ]

    data_group_exts = [
        "txt",
        "png",
        "csv",
        "json",
        "jpg",
        "svg",
        "jpeg",
        "xlsx",
        "npy",
        "sql",
        "tsv",
        "mp4",
        "dat",
        "meta",
        "db",
        "wav",
        "hdf5",
        "ts",
        "sav",
        "data",
        "names",
        "xls",
        "pxd",
        "mp3",
        "pkl",
        "pickle",
        "yaml",
        "yml",
        "xml",
        "pdf",
        "rst",
        "h5",
    ]

    binary_group_exts = [
        "pkl",
        "pickle",
        "jar",
        "exe",
        "bin",
        "out",
        "proto",
    ]

    configuration_group_exts = [
        "yml",
        "cfg",
        "yaml",
        "xml",
        "in",
        "ini",
        "toml",
        "conf",
        "properties",
        "config",
        "gradle",
        "plist",
        "joblib",
        "json",
    ]

    documentation_group_exts = ["md", "pdf", "rst", "tex", "docx", "pptx", "bib"]

    model_group_exts = ["proto", "pt", "pth", "pbt", "pkl", "pickle"]

    log_group_exts = ["log"]

    source_group_exts = [
        "py",
        "ipynb",
        "sh",
        "pyc",
        "bat",
        "h",
        "cpp",
        "c",
        "pb",
        "java",
        "r",
        "pyx",
        "m",
        "mat",
        "cc",
        "ps1",
        "hpp",
    ]

    index = 0
    # for (columnName, columnData) in df.iteritems():

    cat = "data" # 10, 11
    #cat = "binary" # 10, 3
    #cat = "config" # 10, 6
    # cat = "doc" # 10, 2.5
    #cat = "model" # 10, 2.5
    #cat = "log" #10, 0.5
    #cat = "source" #10, 8
    fig_width = 10
    fig_height = 11
    for columnName in data_group_exts:
        print("Column Name : ", columnName)
        # print("Column Contents : ", columnData.values)

        # df[columnName].plot.hist(bins=15, alpha=0.5, y="Depth", title=f"{columnName}: File position depth from project root")

        cname = f".{columnName}"

        try:
            vc = df.groupby([columnName]).size().reset_index(name=cname)
            vc.rename({columnName: "depth"}, axis=1, inplace=True)
            vc.sort_values(ascending=False, by="depth", inplace=True)

            print(vc)
            max = vc[cname].max()
            vc[cname] = vc[cname] / max
            print(vc)

            df_depth_name = pd.merge(df_depth_name, vc, how="outer", on="depth")
            # df_cd.sort_values(ascending=True, inplace=True, by="depth")
            print(df_depth_name)

            index = index + 1
        except:
            print("Error: key not found")

        # if index == 30:
        #    break
        # print(vc)

    x_axis_labels = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]  # labels for x-axis
    boundaries = [0, 0.25, 5, 75, 100]
    # y_axis_labels = [11,22,33,44,55,66,77,88,99,101,111,121] # labels for y-axis
    fig, ax = plt.subplots(figsize=(fig_width,fig_height))  # data 10, 12 - binary 10, 4 -
    s = sns.heatmap(
        df_depth_name.drop("depth", axis=1).transpose(),
        xticklabels=x_axis_labels,
        cmap="Blues",
        # linewidths=0.1,
        vmin=0,
        vmax=1,
    )
    s.set(ylabel="Files extensions", xlabel="Location depths")
    #s.set_title("Depth of extension files from project root directory")
    cbar = s.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["low", "mid", "high"])

    # plt.show()
    output = f"{RESULT_DIR}/5-ext_depth_plots/group_{cat}_depth.png"
    plt.savefig(output, bbox_inches="tight")


if __name__ == "__main__":
    # folder_stats_parallel()
    # folder_name()
    # file_type()
    #get_folder_contents_parallel()
    #plot_folder_content(0.85)
    #plot_folder_content(0.95)
    plot_folder_content(0.98)
    #extension_depth()

    #extension_depth_result()
