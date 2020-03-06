![MPLearn Logo](MPLearn_logo.png "")


# Machine Learning for Morphological Profiling Data



## Installation

### install conda environment on window

    install OpenSSL:
         https://slproweb.com/products/Win32OpenSSL.html
	 ![Win64 OpenSSL v1.1.1d Light (EXE)](https://slproweb.com/download/Win64OpenSSL_Light-1_1_1d.exe)

    follow instructions on https://docs.conda.io/en/latest/miniconda.html
    

### install conda environment on linux

For example, for a linux envirnoment download and install the latest conda platform

    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
    bash Anaconda3-2019.10-Linux-x86_64.sh -b
    rm -rf Anaconda3-2019.10-Linux-x86_64.sh
    # logout and back in to update environment

create an environment and activate it

    conda update -n base -c defaults conda
    conda create --name lab
    conda activate lab

download and install MPLearn package

    git clone git@github.com:momeara/MPLearn.git
    cd MPLearn
    make install_prereqs
    make install

