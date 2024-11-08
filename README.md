# Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision
Code for the paper "*Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision*" by Mélanie Gaillochet, Christian Desrosiers and Hervé Lombaert, presented at MICCAI-MedAGI (2024)



## Setup

### 1) Environment

- Option 1: Create conda environment
```
$ conda create -n py310 python=3.10 pip
$ conda activate py310
$ conda install pytorch torchvision -c pytorch -c nvidia

$ conda install numpy matplotlib scikit-learn scikit-image h5py
$ pip install nibabel comet_ml flatten-dict pytorch_lightning
$ pip install transformers monai opencv-python
```

- Option 2: Create virtual environment with pip
```
$ virtualenv venv
$ source venv/bin/activate

$ pip install torch numpy torchvision matplotlib scikit-image monai h5py nibabel comet_ml flatten-dict pytorch_lightning pytorch-lightning-spells opencv-python
```


### 2) Installing MedSAM backbone
#### Clone MedSAM repository
```
$ git clone https://github.com/bowang-lab/MedSAM
$ cd MedSAM
$ pip install -e .
```

#### Download pre-trained model
1) Get package to download google drive files: `$ pip install gdown`
2) Create target folder: `$ mkdir <path-to-models-folder>/medsam`
3) Download MedSAM model checkpoint: </br>
    `$ gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O <path-to-models-folder>/medsam/medsam_vit_b.pth`
    <br/>
   <u>Note</u>: For this project, the model checkpoint is `medsam_vit_b.pth`
<br/> <br/>

## Data

### 1) Download raw data
- [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html) ([link](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb/folder/637218e573e9f0047faa00fc) to train & test sets) 
- [CAMUS challenge](https://www.creatis.insa-lyon.fr/Challenge/camus/) ([link](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8) to data)
- [HC Challenge](https://hc18.grand-challenge.org/) ([link](https://zenodo.org/records/1327317) to data)
<br/> 

### 2) Preprocess data 
Examples of the preprocessing applied are shown in the notebook `src/JupyterNotebook/Preprocessing_ACDC.ipynb`

The notebook preprocessed the data and generate the following folders that can be used by our dataloader: `train_2d_images`, `train_2d_masks`, `val_2d_images`, `val_2d_masks`, `test_2d_images`, `test_2d_masks`.
<br/> 

<u>Note</u>: For this project, preprocessed outputs folders are `ACDC/preprocessed_sam_slice_minsize_nobgslice_crop256_256`, `CAMUS_public/preprocessed_sam_slice_nobgslice_crop512_512` and `HC18/preprocessed_sam_slice_foreground_crop640_640`
<br/> <br/>

### 3) Save image and positional embeddings
To use pre-computed image embeddings:
- Create a new folder `image_embeddings/medsam_vit_b-pth` in the preprocessed data folder.
- Create subfolders `train_2d_images`, `val_2d_images` and `test_2d_images` in `medsam_vit_b-pth`.
- Run MedSAM and save each image embedding with the same name as the original files, in `train_2d_images`, `val_2d_images` and `test_2d_images`.

To use pre-computed positional encodings:
- Create folder `image_positional_embeddings/medsam_vit_b-pth` in the preprocessed data folder.
- Create subfolders `train_2d_images`, `val_2d_images` and `test_2d_images` in `medsam_vit_b-pth`.
- Run MedSAM and save each computed positional encoding with the same name as the original files, in `train_2d_images`, `val_2d_images` and `test_2d_images`.
- (For fixed positional encoding) Copy positional encoding of first file in `image_positional_embeddings/medsam_vit_b-pth/fixed`
</br>
</br> 

The data folder (`data_dir`) will have the following structure:

```
<data-dir>
├── <dataset_name> 
│ └── raw
│   └── ...
│ └── <preprocessed-folder-name>  
|   └── train_2d_images 
|       └── <patient_name_slice_number.nii.gz>
|       └── ...
|   └── train_2d_masks
|       └── <patient_name_slice_number.nii.gz>
|       └── ...
|   └── val_2d_images     
|   └── val_2d_masks          
|   └── test_2d_images     
|   └── test_2d_masks               
|   └── image_embeddings
|       └── └── <model-checkpoint>
|           └── train_2d_images
|               └── <patient_name_slice_number.nii.gz>
|               └── ...
|           └── val_2d_images
|           └── test_2d_images
|   └── image_positional_embeddings   
|       └── <model-checkpoint>
|           └── fixed
|               └── <patient_name_slice_number.h5>
```


## Running Code

The code is located in the `src` folder.

- Change path to sam_checkpoint in `Configs/model_config/medsam_config/yaml`.
- If using Comet-ml logger, fill in api-key and workspace name in `Configs/logger_config.yaml`.


### Bash script
Experiments can be run with:
```
$ bash src/Bash_scripts/run_experiments_<dataset-name>.sh <path-to-model-config> <class-id> <path-to-train-config> <gpu-idx>
```

ie: `bash bash src/Bash_scripts/run_experiments.sh model_config/medsam_module_config.yaml 1 train_config/train_config_100_0001.yaml 0`


Options:
- `<dataset-name>`: ACDC, CAMUS or HC
- `<path-to-model-config>`: All located in `Configs/model_config` folder. Path starting from `model_config`.
- `<class-id>`: starting from 1 
- `<gpu-idx>`: depending on server
<br/> <br/>


### Main training function
The bash script runs the `main.py` file that trains the prompt module.
There are several input arguments, but most of them get filled in when using the appropriate bash script.

Input of `main.py`:
```
# These are the paths to the data and output folder
--data_dir          # path to directory where we saved our (preprocessed) data
--output_dir        # path to directory where we want to save our output model

# These are config file names located in src/Config
--data_config     
--model_config  
--module_config    
--train_config
--logger_config 
--loss_config       # can be several file names

# Additional training configs (seed and gpu)
--seed
--num_gpu           # number of GPU devices to use
--gpu_idx           # otherwise, gpu index, if we want to use a specific gpu

# Training hyper-parameters that we should change according to the dataset
# Note: arguments start with 'train__' if modifying train_config, 
# and with 'data__'  if modifying model_config, etc.
# Name is determined by hierarchical keywords to value in config, each separated by '__'
--data__use_precomputed_sam_embeddings                  # whether to use precomputed embeddings in data folder
--train__sam_positional_encoding                        # Whether to use fixed positional encoding from data folder                      
--train__train_indices                                  # indices of labelled training data for the main segmentation task
```

Most hyperparameters are set to the default value used to perform the experiments in our paper.

The main file can be run directly - 
ie: `python src/main.py --data_dir <my-data-dir> --output-dir <my-output-dir> --data_config data_config/data_config_ACDC_256.yaml --data__class_to_segment 1  --seed 0`