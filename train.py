from collections import defaultdict
import pickle
import random

import click
import gym

from utils import select_optimal_action

# hyperparameters
alpha = 0.1
gamma = 0.601
epsilon = 0.11

NUM_EPISODES = 50000

# Update Q_table
def update_q(running_Q_table, env, state):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = select_optimal_action(running_Q_table, state)

    # Step the environment by one timestep. Returns = observation, reward, done, info
    next_state, reward, _, _ = env.step(action)
    old_q_value = running_Q_table[state][action]

    # Check if next_state has q values already
    if not running_Q_table[next_state]:
        running_Q_table[next_state] = {action: 0 for action in range(env.action_space.n)}

    # Maximum q_value for the actions in next state
    next_max = max(running_Q_table[next_state].values())

    # Calculate the new q_value
    new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

    # Finally, update the q_value
    running_Q_table[state][action] = new_q_value

    return next_state, reward


def training_agent(running_Q_table, env, num_episodes):
    for i in range(num_episodes):
        # Resets the environment and returns a random initial state.
        state = env.reset()
        if not running_Q_table[state]:
            # 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup, 5 = dropoff
            running_Q_table[state] = {
                action: 0 for action in range(env.action_space.n)}
            
        # Reset Variable 
        epochs, num_penalties, reward, total_reward = 0, 0, 0, 0
        while reward != 20:
            # update q(state, reward)
            state, reward = update_q(running_Q_table, env, state)
            total_reward += reward

            if reward == -10:
                num_penalties += 1

            epochs += 1
        print(f"\nTraining Episode {i + 1}")
        print(f"Time Steps: {epochs}, Penalties: {num_penalties}, Reward: {total_reward}")

    print("Training Finished.\n")

    return running_Q_table


@click.command()
@click.option('--num-episodes', default=NUM_EPISODES, help='Number of Episodes to train on', show_default=True)
@click.option('--save-path', default="q_table.pickle", help='Path to save the Q-table dump', show_default=True)
def main(num_episodes, save_path):
    # load the game environment and render
    env = gym.make("Taxi-v3")
    # Define Dict
    running_Q_table = defaultdict(int, {})
    # Training Agent
    running_Q_table = training_agent(running_Q_table, env, num_episodes)
    # save the table for future use
    with open(save_path, "wb") as f:
        pickle.dump(dict(running_Q_table), f)


if __name__ == "__main__":
    main()
