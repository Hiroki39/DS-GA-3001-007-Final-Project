# DS-GA-3001-007-Final-Project

## Project Description

The goal of this project is to train an RL agent to effectively plan the visit plan for tourists of DisneyLand based on the operation status and waiting time of each ride, weather condition, as well as the current time. The agent will be trained to maximize the total happiness (rewards) of the tourists.

## Usage

Before running the code, please run the following command to register the environment to gym:

```bash
pip install -e disneyenv
```

### Training

To train the agent, run the following command:

```bash
python train_agent.py --algo [ppo|a2c|dqn]
```

The training and evaluation process will be logged in `montor_logs/` folder. The evaluation results as well as the best model will be saved in `eval_results/` folder.

### Testing

To gain the detailed record of the best models' performance on evaluation data, run the following command:

```bash
python test_agent.py --algo [ppo|a2c|dqn|greedy|random]
```

The results will be logged in `test_logs/` folder.

### Visualization

To produce the plots of the testing results, run the following command:

```bash
python visualization.py
```

The plots will be saved in `images/` folder.

### Walking Time Matrix Generation

If you need to re-generate the walking time matrix, run the following command:

```bash
python generate_walking.py
```

The matrix will be saved in `/disneyenv/disneyenv/envs/data/` folder in the form of `walking_time.npy`. Be sure to create a `.env` file in the root directory of the project and set the `GOOGLE_MAP_API` variable to your own Google Map API key.