import pandas as pd
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

"""Analyze the results from commit_stages"""

RESULTS_DIR="results_test/commit_stages"
ANALYSIS_DIR = "results_test/analysis_out"
stages=['Acquisition', 'Preparation', 'Modeling', 'Training', 'Evaluation','Prediction']
FIGSIZE=(12,6)
WSPACE=0.5
plt.rcParams.update({'font.size': 16})

def change_rate(filepath,interval=(0,-1)):
    """
    Determine the proportion of commits affecting each ml stage
    Args:
        filepath:   .csv file containing the commit_stages info
    Returns:    Dictionaty with the proportion of commits affecting each stage and the time in days between first and last commit
    """

    df=pd.read_csv(filepath)[interval[0]:interval[1]]
    n_commits=len(df)
    if n_commits<(interval[1]-interval[0]):
        n_changes={c:float('nan') for c in stages}
    else:
        n_changes={c:0 for c in stages}
        for stage in stages:
            commits=(df.loc[:,stage])
            relevant_commits=commits[commits>0]
            n_changes[stage]=len(relevant_commits)/n_commits

    # n_changes['n_commits']=n_commits
    # start_date=datetime.strptime((df.loc[:,'time'].iloc[0]), '%Y-%m-%d %H:%M:%S%z')
    # end_date=datetime.strptime((df.loc[:,'time'].iloc[-1]), '%Y-%m-%d %H:%M:%S%z')
    # n_changes['time']=(end_date-start_date).days
    return n_changes

def single_run():
    """Bin the data depending on the number of commits and calculate averge change rate for each group and stage."""

    #Get change rate
    filepaths=os.listdir(RESULTS_DIR)
    results_list=[]
    for i,f in enumerate(filepaths):
        print(f'{i}/{len(filepaths)}')
        filepath=os.path.join(RESULTS_DIR,f)
        new_line=change_rate(filepath)
        results_list.append(new_line)
    result_df=pd.DataFrame(results_list)
    print(result_df)

    #Bin results
    result_df['bins']=pd.cut(result_df['n_commits'], [50,70,100,150,300,1000000])
    #Print mean values
    mean=result_df.groupby(['bins']).mean()
    count=result_df.groupby(['bins']).count()
    print('Proportion of commits affecting each ml stage:')
    print(mean)
    print('Number of repositories in each bin')
    print(count)
    mean.to_csv(f'{ANALYSIS_DIR}/commit_stages_mean.csv')
    count.to_csv(f'{ANALYSIS_DIR}/commit_stages_count.csv')

def analyze_intervals():
    #Get change rate
    filepaths=os.listdir(RESULTS_DIR)
    interval_length=10
    results_list=[]
    for interval_start in np.arange(0,1000,interval_length):
        interval_end=interval_start+interval_length
        interval_results_list=[]
        print(f"Interval from {interval_start}")
        for i,f in enumerate(filepaths):
            if (i%5000)==0:
                print(f'{i}/{len(filepaths)}')
            filepath=os.path.join(RESULTS_DIR,f)
            new_line=change_rate(filepath,(interval_start,interval_end))
            interval_results_list.append(new_line)
        interval_results_df=pd.DataFrame(interval_results_list)
        interval_results_mean=interval_results_df.mean()
        interval_results_mean['interval_start']=interval_start
        interval_results_mean['interval_end']=interval_end
        results_list.append(interval_results_mean)

    results_df=pd.DataFrame(results_list)
    print(results_df)
    results_df.to_csv(f'{ANALYSIS_DIR}/commit_stages_intervals',index=False)

def make_plots():
    """Visualize plots for commit intervals"""

    data=pd.read_csv(f'{ANALYSIS_DIR}/commit_stages_intervals')
    fig=plt.figure(figsize=FIGSIZE)

    colors={
        'Acquisition': '#D5E8D4',
        'Preparation':'#DAE8FC',
        'Modeling':'#F8CECC',
        'Training': '#E1D5E7',
        'Evaluation':'#FFE6CC',
        'Prediction':'#BAC8D3'
    }

    for stage in data.columns[:-2]:
        change_rate = data[stage]
        line, = plt.plot(data['interval_start'], change_rate, color=colors[stage], label=stage, linewidth=3)
        line.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='black')])  # Add border
    plt.ylabel("Share of projects")
    plt.xlabel("Commits")
    plt.legend(loc=(0.75,0.45))
    plt.show()
    fig.savefig('paper/fig/commit_stages.pdf',bbox_inches='tight',pad_inches=0)

if __name__=="__main__":
    # analyze_intervals()
    make_plots()
