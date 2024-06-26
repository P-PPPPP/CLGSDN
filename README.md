# CLGSDN

## Introduction
Introduction


<span id='Data preparation'/>

## 1. Data preparation:
The traffic data files (TP_Stroe/raw_files/) are available at [Google Drive](https://drive.google.com/file/d/1rHJYc8cgNFPPvWLRpwynGj2xohqcc2R7/view?usp=sharing) (For Google Drive) or [One Drive](https://1drv.ms/u/c/023ed7fa29970c01/EZIqCXyHPf9Lg3ggSn0TrwYBx4wGLVqJ6zU8dgzh6O_qAg?e=5uNa4K) (For Microsoft One Drive), and should be put into the main/ folder. 
### The folders should be:
```
├─TP_Store  
│    	├─raw_files
│		├─metr_la
│		├─pems_bay
│		├─pems04
│		├─pems08
├─configs
├─model
├─utils
├─engine.py
├─exp.py
├─Readme.pdf
├─Readme.md
├─requirement.txt
```

---------
<span id='Environment'/>
## 2. Create a conda environment for CLGSDN：
### 1.1 Create an environment.
```
# Create an environment.
    conda create -n CLGSDN_envs python=3.11
```
* Notes: You can specify another version of Python，but Python>=3.9.

### 1.2 Activate the environment.
```
# Activate the encironment.
conda activate CLGSDN_envs
```

###1.3 Install required package.
* Pytorch.
Open the Link: https://pytorch.org/get-started/locally/, find your device, and install Pytorch.

* Other packages.
```
# Install packages. 
pip install -r requirement.txt
```
* A special Package.
Pytables cannot be installed via command “pip install”， please use “conda install”.
```
# Install packages.
pip install pytables
```
### 1.4 Test your environment.
```
# A Tesing on environment.
python exp.py
```
The following content indicates that the program ran successfully.
 

## 3. Conduct the Experiment
### 3.1 Specify parameters (model, dataset, etc.).
You can experiment with the following command.
```
#Run the code.
python exp.py --<argument1> <parameter1> --<argument2> <parameter2>…
```
For example：
Run the code.
 python exp.py --model_name dcrnn --dataset metr_la --graphgen_name CLGSDN

This command indicates that <CLGSDN> is used to generate graphs on the <Metr-la> dataset and the <DCRNN> model is used for prediction.

	Select a model.
--model_name <model name>
The options for <model name> are: agcrn, astgcn, tgcn, astgcn, gw, agcrn, dstagnn, lightcts, megacrn and ddgcrn.

	Select a dataset.
Dataset	Data channel	Command
Metr-la	[0]: speed	--dataset metr_la --choise_channels [0]
Pems-bay	[0]: speed	--dataset pems_bay --choise_channels [0]
Pems04	[2]: speed	--dataset pems04 --choise_channels [2]
Pems08	[2]: speed	--dataset pems08 --choise_channels [2]
Taxibj13	[1]: In flow	--dataset taxibj13 --choise_channels [1]
Taxibj13	[-1] or [0,1]:
In and out flow	--dataset taxibj13 --choise_channels [-1]
--dataset taxibj13 --choise_channels [0,1]
Pems04/08	[0]: flow	--dataset pems04 --choise_channels [0]
When the selected dataset is Metr-la or Pems-bay, the parameter <--choise_channels> does not need to be specified. Its default value is [-1], which means all the channels are selected; However, there is only one channel: vehicle speed.

	Running CLGSDN (as a Graph Generator)
	--graphgen_name CLGSDN
	This command indicates that the <model> will use the graph generated by <CLGSDN> and will be optimized simultaneously with CLGDSN.
	--graphgen_name None
This command means that only the <graph provided by the dataset> is used. If the dataset does not provide any graph, it is the <identity matrix>.

4.	Details.
	Final Report
After the program ends, the following result will be output.
 
Idx: The epoch with the minimum loss on validation set. (Results below refers to its testing errors.)

	Logs
More detailed information can be found in the log. For example, the log for Epoch 41 is
 
where: 
<Info Report> is the result on the training/validation/test set.
<Step 1-12> is the MAE result of 12 steps (on test set). For example, the results in the paper DCRNN or Graph Wavenet show the errors at the 3rd step (15 min), the 6th step (30 min), and the 12th step (1 hour).

	Reproduce the results (Baseline)
ATTENTION PLEASE: Our experiment focuses on performance changes (whether to use CLGSDN as the Graph Generator). Thus, All BASELINE SETTINGS ARE THE SAME, AND THEIR PERFORMACE IS NOT ALWAYS THE BEST (Compared to the original paper).
If you want to fully reproduce the results of certain models (of certain published paper) through this program, please ensure the consistency of parameters. For example, if you want to reproduce the results of <DCRNN> on the <Metr-la> dataset, use the following command:
DCRNN.
 python -exp.py --model_name dcrnn --graphgen_name None --dataset metr_la 
--dataset_prob [0.7,0.1,0.2] --epochs 100

The results are:
		MAE		
Source	15 min	30 min	1 hour	Average
Original	2.77	3.15	3.60	\
This Implementation	2.63	3.04	3.62	3.02
With CLGSDN	2.605	2.98	3.54	2.97
Table: MAE Results of DCRNN
However, the parameters required for each baseline are different.

	Reproduce the results (Ours)
Using the command <--graphgen_name CLGSDN>. For example:
DCRNN.
 python -exp.py --model_name dcrnn --graphgen_name CLGSDN --dataset metr_la --dataset_prob [0.7,0.1,0.2] --epochs 100 
	
