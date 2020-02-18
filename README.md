# Machine Learning for Morphological Profiling Data



## Installation

### install conda environment

For example, for a linux envirnoment download and install the latest conda platform

    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
    bash Anaconda3-2019.10-Linux-x86_64.sh -b
    rm -rf Anaconda3-2019.10-Linux-x86_64.sh
    # logout and back in to update environment

create an environment and activate it

    conda update -n base -c defaults conda
    conda create --name lab
    conda activate lab

install pre-requisite packages

   conda install setuptools pandas joblib scikit-learn umap-learn
   conda install -c conda-forge umap-learn datashader

Note that `hdbscan` on conda-forge doesn't yet support python 3.8 so I had to build it from source

   mkdir ~/opt/hdbscan
   cd ~/opt/hdbscan
   wget https://github.com/scikit-learn-contrib/hdbscan/archive/master.zip
   unzip master.zip
   rm master.zip
   cd hdbscan-master
   pip install -r requirements.txt
   python setup.py install

download and install MPLearn package

    git clone git@github.com:momeara/MPLearn.git
    cd MPLearn
    python setup.py install
