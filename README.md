# RL_Project
https://github.com/satwikkansal/q-learning-taxi-v3#q-learning-taxi-v3
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

## Q-learning taxi-v3
- State

![Reinforcement_Learning_Taxi_Env width-1200](https://github.com/ZeusKwon/RL_Project/assets/70212461/9b0f8ddc-c95f-40d6-998a-de5af902e3bf)

The State Space is the set of all possible situations our taxi could inhabit. The state should contain useful information the agent needs to make the right action.
R, G, Y, B or [(0,0), (0,4), (4,0), (4,3)] in (row, col) coordinates.
- action 
  - South
  - North
  - East
  - West
  - Pickup
  - Dropoff

- reward 
  - successful drop-off : +20
  - every time-step : -1
  - illegal pick-up and drop-off : -10

Table based q-learning implementation for taxi-v3 environment of Open AI gym.

## Instructions to run
```shell script
$ pip install -r requirements.txt
# or pip install gym==0.17.1 click
```

### Training
```shell script
$ python train.py --help
Usage: train.py [OPTIONS]

Options:
  --num-episodes INTEGER  Number of episodes to train on  [default: 100000]
  --save-path TEXT        Path to save the Q-table dump  [default:
                          q_table.pickle]
  --help                  Show this message and exit.
```

### Evaluation
```shell script
$ python evaluate.py --help
Usage: evaluate.py [OPTIONS]

Options:
  --num-episodes INTEGER  Number of episodes to train on  [default: 100]
  --q-path TEXT           Path to read the q-table values from  [default:
                          q_table.pickle]
  --help                  Show this message and exit.
```
