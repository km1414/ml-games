# ml-games

Project, created to run reinforcement learning experiments in **gym** environments. 
The idea is to separate processes of running the games and training the RL model in 
order test and compare different algorithms. Currently works with **Atari** games from gym 
and uses raw pixels input to predict discrete actions as an output.




### Installation

---
**Windows:**

The easiest way to set up the system is to use **Anaconda**. After installation, run the following commends in **cmd.exe**:

`conda create -n ml-games python=3.6 anaconda` 

`conda activate ml-games`

`pip install -U numpy`

`pip install tensorflow`

`pip install gym`

`pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py`



### Usage

---
Clone the repo: `git clone https://github.com/km1414/ml-games.git`

Enter cloned repo: `cd ml-games`

Run example with random decisions: `python main.py -game Breakout-v0 -model RandomModel`



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






