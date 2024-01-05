This repo contains the appendix/instruments for our paper, "A Large-Scale Study of ML-Related Python Projects".

The contents of each folder are listed below:

**data**: Contains the data ordered by filtering/ processing steps
- **1-dependents**: Names of GitHub projects that are dependent on SciKit-Learn and TensorFlow libraries.
- **1a-dependents_queried**: List of projects with additional information obtained from the GitHub REST API
- **2-forks_removed**: List of GitHub projects left after removing forks.
- **3a-number_commits_queried**: List of projects with their number of commits.
- **3-number_commits_filtered**: List of projects with commits count >= 50.
- **4-library_calls_filtered**: List of projects with relevant library calls.
- **5-attributes**: List of projects along with the following attributes: number of contributors, branches, pull requests, tags, number of releases, issues, files, and their ML development 
phases.
- **6-ml_stages**: List of projects with information about how many files related to each ML stage are present
- **commit_stages**: Information about which ml stage was changed in which commit. Contains one .csv file for each project.
- **API-dictionary**: API dictionary mapping libraries calls to ML development phases.


**scripts**: Python scripts used in this study.
  - **utilities**: Collection of functions required for the scripts
  - **data_acquisition**: Scripts required to obtain the data from GitHub
  - **filtering**: Scripts for filtering the data obtained from GitHub
  - **data_processing**: Generating further information required for analysis
  - **analysis**: Used to generate the results presented in the paper based on filtering and data_processing



**Installation**
  - Required Python version: 3.8
  - Dependencies can be found in requirements.txt
  - Suggested installation procedure:
    - conda create -n ml-systems-study python=3.8
    - conda activate ml-systems-study
    - pip install -r requirements.txt
    - conda develop "path/to/cloned/repository"

**Reproducing the results from the paper:**
  - Obtaining the data from GitHub: Execute scripts/data_acquisition/01_get_dependants.py
  - Filtering the data: Execute the Python scripts in scripts/filtering in the given order, except for the last one (12_filter_experiments.py)
  - Additional processing required for results to RQ3: Run scripts/data_processing/commit_stages.py
  - Analysis:
    - Table 1:
    - Figure 4: scripts/analysis/make_plots.py
    - RQ2 (fig. 3&4, Table I&II): scripts/analysis/ml_stages.py
    - RQ3 (fig. 5): scripts/analysis/commit_stats
  - Results for RO3: Run scripts/filtering/12_filter_experiments.py
