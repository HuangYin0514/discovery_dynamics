import numpy as np


def plot_energy(ax, t, eng, potential_eng, kinetic_eng, title_label):
    ax.set_xlabel('Time step $(s)$')
    ax.set_ylabel('$E\;(J)$')
    ax.plot(t, potential_eng, 'y:', label='potential', linewidth=2)
    ax.plot(t, kinetic_eng, 'c-.', label='kinetic', linewidth=2)
    ax.plot(t, eng, 'g--', label='total', linewidth=2)
    ax.legend(fontsize=12)
    ax.set_title(title_label)


def plot_compare_energy(ax, t, true_eng, pred_eng, title_label):
    ax.set_xlabel('Time step $(s)$')
    ax.set_ylabel('$E\;(J)$')
    ax.set_yscale('log')
    # ax.plot(t, true_eng, 'g--', label='True Energy', linewidth=2)
    # ax.plot(t, pred_eng, 'r--', label='Prediction Energy', linewidth=2)
    ax.plot(t, np.abs(true_eng - pred_eng), 'b--', label='Energy Error', linewidth=2)
    ax.legend(fontsize=12)
    ax.set_title(title_label)


def plot_compare_state(ax, t, true_state, pred_state, title_label):
    ax.set_xlabel('Time step $(s)$')
    ax.set_ylabel('$State$')
    ax.plot(t, true_state, 'g--', label='True state', linewidth=2)
    ax.plot(t, pred_state, 'r--', label='Prediction state', linewidth=2)
    ax.legend(['True state', 'Prediction state'], fontsize=12)
    ax.set_title(title_label)


def plot_field(ax, t, state_q, state_p, title_label):
    ax.set_xlabel('state $q$')
    ax.set_ylabel('state $p$')
    ax.plot(state_q, state_p, 'g--', label='state', linewidth=2)
    ax.legend(['state'], fontsize=12)
    ax.set_title(title_label)
