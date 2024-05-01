# AutoCI
Official PyTorch implementation for the following manuscripts:

[Automated causal inference in application to randomized controlled clinical trials](https://www.nature.com/articles/s42256-022-00470-y), Nature Machine Intelligence (2022). 

[Prognostic impact and causality of age on oncological outcomes in women with endometrial cancer: a multimethod analysis of the randomised PORTEC-1, -2 and -3 trials](https://www.sciencedirect.com/science/article/pii/S1470204524001426), Lancet Oncology (2024). 


<p align="center">
<img src="visual/fig1.png" width="1600px"/>  
<br>
The overall model illustration of the proposed AutoCI.       
</p>

<a href="https://www.nature.com/articles/s42256-022-00470-y"><img src="https://img.shields.io/badge/Nature-MachineIntelligence-brightgreen" height=22.5></a> \
<a href="https://www.nature.com/articles/s42256-022-00470-y"><img src="https://img.shields.io/badge/Lancet-Oncology-orange" height=22.5></a> \
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  

If you find this repository helpful for your research, we would appreciate your citation of our studies.
```
@article{wu2022automated,
    title={Automated causal inference in application to randomized controlled clinical trials},
    author={Wu, Ji Q and Horeweg, Nanda and de Bruyn, Marco and Nout, Remi A and J{\"u}rgenliemk-Schulz, Ina M and Lutgens, Ludy CHW and Jobsen, Jan J and Van der Steen-Banasik, Elzbieta M and Nijman, Hans W and Smit, Vincent THBM and others},
    journal={Nature Machine Intelligence},
    volume={4},
    number={5},
    pages={436--444},
    year={2022},
    publisher={Nature Publishing Group UK London}
}

@article{WAKKERMAN2024,
    title = {Prognostic impact and causality of age on oncological outcomes in women with endometrial cancer: a multimethod analysis of the randomised PORTEC-1, PORTEC-2, and PORTEC-3 trials},
    author = {Famke C Wakkerman and Jiqing Wu and Hein Putter and Ina M Jürgenliemk-Schulz and Jan J Jobsen and Ludy C H W Lutgens and Marie A D Haverkort and Marianne A {de Jong} and Jan Willem M Mens and Bastiaan G Wortman and Remi A Nout and Alicia Léon-Castillo and Melanie E Powell and Linda R Mileshkin and Dionyssios Katsaros and Joanne Alfieri and Alexandra Leary and Naveena Singh and Stephanie M {de Boer} and Hans W Nijman and Vincent T H B M Smit and Tjalling Bosse and Viktor H Koelzer and Carien L Creutzberg and Nanda Horeweg},
    journal = {The Lancet Oncology},
    year = {2024},
    issn = {1470-2045},
    doi = {https://doi.org/10.1016/S1470-2045(24)00142-6},
    url = {https://www.sciencedirect.com/science/article/pii/S1470204524001426},
}
```


## Installation
This implementation is dependent on heavily refactored [HOUDINI](https://github.com/trishullab/houdini) and libraries from [AICP](https://github.com/juangamella/aicp). While [HOUDINI](https://github.com/trishullab/houdini) is compatible to a large variety of PyTorch versions, the libraries that are refactored from [AICP](https://github.com/juangamella/aicp.) require certain specific packages.

**Prerequisites**    
This implementation has been successfully tested under the following configurations,
while these configurations could be downgraded to a wide range of earlier versions:

- Ubuntu 22.04
- Nvidia driver 525.147
- CUDA 11.7
- Python 3.9
- PyTorch 2.1
- Miniconda 
- Docker 
- Nvidia-Docker2

Note that lower PyTorch version should also work.
After the prerequisites are satisfied, you could either build the docker image
or install conda dependencies.

**Docker Installation**     
Assume Docker and Nvidia-Docker2 toolkit are correctly installed, build the Docker image as follows

```
cd /Path/To/AutoCI/
docker build -t autoci .
```

**Conda Installation**      
Assume a local Conda environment is installed and activated, 
install the dependencies as follows   

```
pip install scipy pandas pycox pylatex pygam pyaml pyreadstat matplotlib scikit-learn termcolor sklearn joblib lifelines  networkx==2.4 sempler==0.1.1
```
You could also try creating a conda environment from [environment.yml](environment.yml).



## Data Preparation
Download the following toy datasets and unzip them resp.:   
- Finite sample setting  
    - [0 confounder](https://zenodo.org/records/10042871/files/fin.zip?download=1)
    - [1 confounder](https://zenodo.org/records/10042871/files/fin1.zip?download=1)
    - [2 confounders](https://zenodo.org/records/10042871/files/fin2.zip?download=1)
- ABCD setting  
    - [0 confounder](https://zenodo.org/records/10042871/files/abcd.zip?download=1)
    - [1 confounder](https://zenodo.org/records/10042871/files/abcd1.zip?download=1) 
    - [2 confounders](https://zenodo.org/records/10042871/files/abcd2.zip?download=1)

## Run the Toy Experiments
In case of using Docker image, you could first launch the Docker image

**Launch the Docker Image**
```
sudo docker run -it \
    -v /Path/to/AutoCI:/root/AutoCI \
    -v /Path/to/Data/:/root/Data \
    --gpus '"device=0"'  autoci
```


**Run the AutoCI experiments**
```
cd /Path/To/AutoCI/

python -m HOUDINI.Run.LGANM --lganm-dir /Path/to/Data \
                            --res-dir /Path/to/Result  
```

**Run the (N)ICP experiments**
```
cd /Path/To/AutoCI/

python -m HOUDINI.Run.baseline_lganm --lganm-dir /Path/to/Data \
                                     --res-dir /Path/to/Result \
                                     --method nicp (or icp)
```
**Run the AICP experiments**    
Please see [AICP](https://github.com/juangamella/aicp) repository for more details.

## Run the PORTEC123 Experiments

**Run the AutoCI analysis**
```
cd /Path/To/AutoCI/
sh run.sh
```

**Run the correlation analysis**

Please check the R script [correlation_analysis.R](correlation_analysis.R) for the implementation detail. 



## Tipps for training models without program synthesis  
If you want to train the neural network model while excluding the program sythesis module, then you could refactor the following codes to obtain simple neural network training codes:     
- [HOUDINI/Interpreter/Interpreter.py](HOUDINI/Interpreter/Interpreter.py): includes the core training functions
- [HOUDINI/Library/NN.py](HOUDINI/Library/NN.py): includes the simple neural network components
- [HOUDINI/Run/LGANM.py](HOUDINI/Run/LGANM.py): includes the critical training parameters and data loader functions


## Acknowledgement
This implementation is built upon [HOUDINI](https://github.com/trishullab/houdini). Besides, we borrow codes from [AICP](https://github.com/juangamella/aicp).
We would like to convey our gratitudes to all the authors working on those projects.
