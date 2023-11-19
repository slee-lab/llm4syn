# CAVA
Synthesis recipe generation

## (0) Create folders
./data/
./models/

## (1) download dataset
download dataset from the two sources below. Place the .json files in ./data/
https://www.nature.com/articles/s41597-019-0224-1#citeas
https://www.nature.com/articles/s41597-022-01317-2


## (2) Set up personal accounts for Hugging Face and WandB
### (2-1) Hugging face
https://huggingface.co/
Issue API key for 'Writing' (and 'Reading')
Duplicate 'env_config_template.py' and rename it as 'env_config.py'
Paste the API key and your Username in 'env_config.py'. 

### (2-2) WandB
https://wandb.ai/site
MMake accoutn and login

## (3) Setup envs
'''
torch==2.0.0+cu118__
transformers==4.33.2__
other basic libraries like numpy, matplotlib etc. __
'''

## (4) Use train code
### (4-1) set dataset 
data_path: path to the .json data file. 
dataset function: decide the prompt and output text. Check the input parameters for each 
dataset_lhs2rhs: Predict RHS of a chemical equation given LHS
dataset_rhs2lhs: Predict LHS of a chemical equation given RHS
dataset_ope2ceq: predict chemical equation given target compounds and synthesis operaations. ver1
dataset_ceq2ope: predict synthesis operaations given chemical equation. ver1
dataset_ope2ceq_2: predict chemical equation given target compounds and synthesis operaations. ver2
dataset_ceq2ope_2: predict synthesis operaations given chemical equation. ver2
dataset_ope2ceq_3: predict chemical equation given target compounds and synthesis operaations. ver3
dataset_ceq2ope_3: predict synthesis operaations given chemical equation. ver3

## (4-2) Define model to load
hf_model: the open-access model loaded from Hugging Face. We initialize model from this one. 
e.g. "gpt2", "distilgpt2", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-j-6B" 
model_name: where to save the trainedmodel in your Hugging Face account. If load_pretrained==True, we load from our own pre-trained model and update by further training. 


### (4-3) Others
separator: Predicted text format (prompt separator answer)


## Reference
```
@article{kononova2019text,
  title={Text-mined dataset of inorganic materials synthesis recipes},
  author={Kononova, Olga and Huo, Haoyan and He, Tanjin and Rong, Ziqin and Botari, Tiago and Sun, Wenhao and Tshitoyan, Vahe and Ceder, Gerbrand},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={203},
  year={2019},
  publisher={Nature Publishing Group UK London}
}

@article{wang2022dataset,
  title={Dataset of solution-based inorganic materials synthesis procedures extracted from the scientific literature},
  author={Wang, Zheren and Kononova, Olga and Cruse, Kevin and He, Tanjin and Huo, Haoyan and Fei, Yuxing and Zeng, Yan and Sun, Yingzhi and Cai, Zijian and Sun, Wenhao and others},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={231},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```