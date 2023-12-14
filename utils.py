def select_optimal_action(q_table, state):
    max_q_value_action = None
    max_q_value = -100000
    
    # update max_q_value and set max_q_value_action
    if q_table[state]:
        for action, action_q_value in q_table[state].items():
            if action_q_value >= max_q_value:
                max_q_value = action_q_value
                max_q_value_action = action

    return max_q_value_action