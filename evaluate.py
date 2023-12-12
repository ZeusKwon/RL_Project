import pickle


import click
import gym

from utils import select_optimal_action


NUMBER_EPISODES = 100


def evaluate_agent(running_Q_table, env, number_trials):
    total_epochs, total_penalties = 0, 0

    print("Running episodes...")
    for _ in range(number_trials):
        state = env.reset()
        epochs, number_penalties, reward = 0, 0, 0

        while reward != 20:
            next_action = select_optimal_action(running_Q_table,state)
            state, reward, _, _ = env.step(next_action)

            if reward == -10:
                number_penalties += 1

            epochs += 1

        total_penalties += number_penalties
        total_epochs += epochs

    average_time = total_epochs / float(number_trials)
    average_penalties = total_penalties / float(number_trials)
    print(f"Evaluation results after {number_trials} trials")
    print(f"Average time steps taken: {average_time}")
    print(f"Average number of penalties incurred: {average_penalties}")


@click.command()
@click.option('--num-episodes', default=NUMBER_EPISODES, help='Number of episodes to train on', show_default=True)
@click.option('--q-path', default="q_table.pickle", help='Path to read the -table values from', show_default=True)
def main(num_episodes, q_path):
    env = gym.make("Taxi-v3")
    with open(q_path, 'rb') as f:
        running_Q_table = pickle.load(f)
    evaluate_agent(running_Q_table, env, num_episodes)


if __name__ == "__main__":
    main()
