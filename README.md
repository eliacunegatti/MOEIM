# MOEIM

**Many-Objective Evolutionary Influence Maximization: Balancing Spread, Budget, Fairness ,and Time** </br>
*To appear at Gecco 2024* </br>
Elia Cunegatti, Leonardo Lucio Custode, Giovanni Iacca <br>
University of Trento, Italy  <br>
[![arXiv](https://img.shields.io/badge/arXiv-2404.05621-b31b1b.svg)](https://arxiv.org/pdf/2403.18755.pdf)
 

```bibtex
@article{cunegatti2024many,
  title={Many-Objective Evolutionary Influence Maximization: Balancing Spread, Budget, Fairness, and Time},
  author={Cunegatti, Elia and Custode, Leonardo Lucio and Iacca, Giovanni},
  journal={arXiv preprint arXiv:2403.18755},
  year={2024}
}
```
## Setup

```
conda create -n myenv python=3.8.17
conda activate myenv    
conda install --file requirements.txt
```


## Usage
Here we provide the main commands to use our codebase as well as reproducing the original results.

### Experimental setting 1
To run MOEAIM (it runs on default with jazz dataset, to be changed using --dataset args):
```
python influence_maximization.py \
    --model WC \
    --experimental_setup setting1 \
    --version graph-aware \
```

To run MOEA (it runs on default with jazz dataset, to be changed using --dataset args):
```
python influence_maximization.py \
    --model WC \
    --experimental_setup setting1 \
    --version base \
```

To run CELF/GDD (it runs on default on all datasets):
```
python utils/heuristic.py /
     --model WC \
     --method CELF \
```
### Experimental setting 2
To run MOEAIM (it runs on default with jazz dataset, to be changed using --dataset args):
```
python influence_maximization.py \
    --method WC \
    --type setting2 \
    --version graph-aware \
```

This is what you see if you run ```python influence_maximization.py --help ```
```
usage: influence_maximization.py [-h] [--k K] [--p P] [--no_simulations NO_SIMULATIONS] [--model {LT,WC,IC}]
                                 [--obj_functions {['spread', 'seed'],['spread', 'seed', 'time'],['spread', 'seed', 'communities'],['spread', 'seed', 'budget'],['spread', 'seed', 'fairness']}]
                                 [--population_size POPULATION_SIZE] [--offspring_size OFFSPRING_SIZE] [--max_generations MAX_GENERATIONS] [--tournament_size TOURNAMENT_SIZE] [--num_elites NUM_ELITES]
                                 [--no_runs NO_RUNS] [--version {graph-aware,base}] [--experimental_setup {setting1,setting2}] [--smart_initialization_percentage SMART_INITIALIZATION_PERCENTAGE]
                                 [--search_space_size_min SEARCH_SPACE_SIZE_MIN] [--search_space_size_max SEARCH_SPACE_SIZE_MAX]
                                 [--graph {facebook_combined_un,email_di,soc-epinions_di,gnutella_di,wiki-vote_di,CA-HepTh_un,lastfm_un,power_grid_un,jazz_un,cora-ml_un}] [--out_dir OUT_DIR]

Evolutionary algorithm computation.

optional arguments:
  -h, --help            show this help message and exit
  --k K                 Seed set size as percentage of the whole network.
  --p P                 Probability of influence spread in the IC model.
  --no_simulations NO_SIMULATIONS
                        Number of simulations for spread calculation when the Monte Carlo fitness function is used.
  --model {LT,WC,IC}    Influence propagation model.
  --obj_functions {['spread', 'seed'],['spread', 'seed', 'time'],['spread', 'seed', 'communities'],['spread', 'seed', 'budget'],['spread', 'seed', 'fairness']}
                        Objective functions to be optimized.
  --population_size POPULATION_SIZE
                        EA population size.
  --offspring_size OFFSPRING_SIZE
                        EA offspring size.
  --max_generations MAX_GENERATIONS
                        Maximum number of generations.
  --tournament_size TOURNAMENT_SIZE
                        EA tournament size.
  --num_elites NUM_ELITES
                        EA number of elite individuals.
  --no_runs NO_RUNS     Number of runs of the EA.
  --version {graph-aware,base}
                        Smart Initialization and graph-aware operators or no Smart Initialization and base mutator operators.
  --experimental_setup {setting1,setting2}
                        Setting 1 and 2 of the experimental section.
  --smart_initialization_percentage SMART_INITIALIZATION_PERCENTAGE
                        Percentage of "smart" initial population, to be specified when multiple individuals technique is used.
  --search_space_size_min SEARCH_SPACE_SIZE_MIN
                        Lower bound on the number of combinations.
  --search_space_size_max SEARCH_SPACE_SIZE_MAX
                        Upper bound on the number of combinations.
  --graph {facebook_combined_un,email_di,soc-epinions_di,gnutella_di,wiki-vote_di,CA-HepTh_un,lastfm_un,power_grid_un,jazz_un,cora-ml_un}
                        Dataset name (un_ for undirected and di_ for directed).
  --out_dir OUT_DIR     Location of the output directory in case if outfile is preferred to have a default name.

```
## Paper Results
For the results displayed in the paper we provide the original files in ```data/results```. The folders are split w.r.t to the experimental setting and dataset.

## Dataset
The datasets used in the paper are available in ```data/graphs```. In details in ```data/graphs/standard``` are available the original datasets, in ```data/graphs/cleaned``` the datasets after pre-processing and in ```data/graphs/communities``` the communities ground truth for all the datasets.
To clean the datasets please use the following pipeline: ```utils/pre_processing.py``` preprocess the graphs as explained in the paper, then ```utils/relabel_graph.py``` rewire the nodes ids such as they are bound in [0,N] where N is the number of nodes in the graph. To conclude ```utils/community_detection.py``` computed the communities over the graphs. Please change the file path based on your settings. To run smart initialization at priori run ```python utils/smart_initialization.py ```, by default it computes the smart initialization for all datasets and propagation models.

## License 
This project is released under the MIT license.

## Contact
For any questions/doubt please feel free to contact me at: elia.cunegatti@unitn.it
