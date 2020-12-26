Installation
==================
Python environment
------------------
Install Miniconda3 from executable or command line (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-in-silent-mode).

On Linux or Mac
~~~~~~~~~~~~~~~~~~
    wget  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

    sha256sum ~/miniconda.sh

    bash ~/miniconda.sh -b -p $HOME/miniconda

    eval "$(~/miniconda/bin/conda shell.bash hook)"

CUDA Toolkit
------------------
If you want to use your GPU capabilities, download and follow the instructions here : https://developer.nvidia.com/cuda-toolkit-archive
We have tested the following configuration : Windows 10, CUDA 10.2. In case you have a different configuration, please change the environment.yml accordingly.

thermalcognition environment
------------------
Import a new conda environment :

    conda env create -f environment.yml

Then the newly created environment must be activated :

    conda activate thermalcognition

FLIR Science File environment
------------------
To open FLIR files, we provide the official SDK in ./FLIR Science File SDK. It can be downloaded from can be downloaded there : https://flir.custhelp.com/app/account/fl_download_software if you create a free FLIR developer account.
Please select one of them and run it. Then, follow the instructions in the default location C:/Program%20Files/FLIR%20Systems/sdks/file/python/doc/index.html :

    python setup.py install

in the C:\Program Files\FLIR Systems\sdks\file\python directory.

thermalcognition package
------------------
The thermalcognition package is installed by running in the root directory :

    pip install .

It can be uninstalled by running :

    pip uninstall thermalcognition

Development mode
------------------
thermalcognition can be installed in development mode using :

    pip install -e .
