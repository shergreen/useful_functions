import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper",font_scale=2.0)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
plt.rc('text', usetex=True)

def loglogplot(figsize=(7,5.8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.loglog()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', colors='black', width=1.0, length=3.0)
    return fig, ax

def plot(figsize=(7,5.8)): #figure out what the standard figure size is for journal articles...
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', colors='black', width=1.0, length=3.0)
    return fig, ax
