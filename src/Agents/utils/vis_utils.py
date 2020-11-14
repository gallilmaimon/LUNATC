import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def log_results(dqn, handle_out, file_name, mean_n=100):
    if handle_out == 'save':
        res_df = pd.DataFrame({'text': dqn.init_states[:len(dqn.rewards)],
                               'new_text': dqn.final_states[:len(dqn.rewards)],
                               'score': dqn.rewards})
        res_df.to_csv(file_name, index=False)
    elif handle_out == 'plot':
        plt.plot(dqn.rewards)
        plt.plot(running_mean(dqn.rewards, mean_n))
        plt.show()
