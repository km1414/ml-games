# ml-games

---

_Project, created to run reinforcement learning experiments in **gym** environments. 
The idea is to create universal framework, that could be used to run the games as 
well as comfortably test and compare different RL algorithms. Currently works with **Atari** 
envs from gym and uses raw pixels input to predict discrete actions as an output._




## Installation

### _Windows:_

**Download and install [Anaconda](https://www.anaconda.com/distribution/). 
Open _cmd.exe_ and run the following commands:**

**Create conda env:**

`conda create -n ml-games python=3.6 anaconda` 

**Activate conda env:**

`conda activate ml-games`

**Install/update required packages:**

`pip install -U numpy`

`pip install tensorflow`

`pip install gym`

`pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py`

**Clone the project:**

`git clone https://github.com/km1414/ml-games.git`


### _Linux:_
**TODO**


## Usage

**Open _cmd.exe_, navigate to project directory and activate previously created conda env. 
Pick the game from [Atari games list](https://gym.openai.com/envs/#atari) and run it with selected model. 
By default it will run 10k games, in order to change that, 
add **`-n_games`** and the number to the command.
More hyper-parameters for RL models could be changed directly in the scripts.**

**Examples:**

`python main.py -game Breakout-v0 -model RandomModel`

`python main.py -game Pong-v0 -model PolicyGradientsModel -n_games 20000`

**Performance statistics and game recordings will be saved in **_images/_** and **_videos/_**.**


## Repo overview


**gamer.py (Gamer)** - runs the games, tracks statistics and gets action from model. At the selected frequency plots game statistics 
to **_images/_** and saves game game play recordings to **_videos/_** .

**models/random_model.py (RandomModel)** - baseline model, that always makes random moves. Created to test if everything works and contains 
only mandatory methods:
- **predict_action(observation)**  - get the current game state from Gamer and return predicted action.
- **get_step_results(observation, reward, done, info)** - get the results after the action step was done.

**models/policy_gradients_model.py (PolicyGradientsModel)** - RL model, that predicts best action from existing observation and learns from his experience. 
Based on simple neural network and Policy Gradients approach. Contains same methods as RandomModel and many more. 

**main.py** Extracts command line arguments, loads initializes objects
and runs the projects according to user preferences.

## Results

**RandomModel** - base line results of different envs making only random moves (10k games).

<img src="/presentation/random_model/RandomModel_Bowling-v0_history.png" height="340"/><img src="/presentation/random_model/RandomModel_Boxing-v0_history.png" height="340"/>
<img src="/presentation/random_model/RandomModel_Breakout-v0_history.png" height="340"/><img src="presentation/random_model/RandomModel_Pong-v0_history.png" height="340"/>

## Contribution
New models could be created and added to **/models**. They should contain same methods as 
RandomModel and in order to run they should be added to **main.py**.






