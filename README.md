# ml-games

_Project, created to run reinforcement learning experiments in **gym** environments. 
The idea is to create universal framework, that could be used to run the games as 
well as comfortably test and compare different RL algorithms. Currently works with **Atari** 
envs from gym and uses raw pixels input to predict discrete actions as an output._




## Installation

---
### Windows:

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


## Usage

---
**Open **cmd.exe**, navigate to project directory and activate previously created conda env. 
Pick the game from **Atari** list and run it with selected model. 
By default it will run 10k games, if want to change that, 
add `-n_games` and the number to the command.
More hyper-parameters for RL models could be changed directly in the scripts.**

Examples:**

`python main.py -game Breakout-v0 -model RandomModel`

`python main.py -game Pong-v0 -model PolicyGradientsModel -n_games 20000`

Watch results in the screen, then check **_images/_** and **_videos/_** 
folders in the project directory to see the statistics and game recordings.



### Classes

---
**Gamer** - runs the games, tracks statistics and gets action from model. At the selected frequency plots game statistics 
to **_images/_** and saves game game play recordings to **_videos/_** .

**RandomModel** - baseline model, that always makes random moves. Created to test if everything works and contains 
only mandatory methods:
- **predict_action(observation)**  - get the current game state from Gamer and return predicted action.
- **get_step_results(observation, reward, done, info)** - get the results after the action step was done.

**PolicyGradientsModel** - RL model, that predicts best action from existing observation and learns from his experience. 
Based on simple neural network and Policy Gradients approach. Contains same methods as RandomModel and many more. 


### Results

---

TODO


### Contribution

---

TODO




