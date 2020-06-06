# -*- coding: utf-8 -*-
"""
This script performs statistical analysis on instabilities and outputs figures
and a html with stats

"""

import os
import sys
import itertools
import glob
import numpy as np
import pandas as pd
import math
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


print("Performing statistical analysis")

# =============================================================================
# Setup
# =============================================================================

# Run matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Define filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../..")) + "/"
WORKDIR = HOMEDIR + "data/instabilities/"
OUTDIR = HOMEDIR + "data/stats/"

# Settings
SAVE = 1  # Save outputs or not
MAX_TIME = 23  # Maximum number of snapshot pairs
TAU_CUTOFF = 20  # Maximum tau value that will be displayed
TAU_PICK = 1  # Tau to visualize on barplots

# Runs to include
subjects = [f"sub{int(subid):0>3}" for subid in sys.argv[1:]]  # Subject IDs
# from bash script
task = "rest"
boluses = ["BHB", "GLC"]
PrePost = ["pre", "post"]
taus = np.arange(1, TAU_CUTOFF+1)
times = np.arange(1, MAX_TIME+1)

N = len(subjects)

# HTML formatting settings
#- Color
cm = sns.cubehelix_palette(as_cmap=True, dark=0.5)

#- Set CSS properties for th elements in dataframe (table header)
th_props = [
  ('font-size', '16px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7f7f9'),
  ('border-witdth', '12px')
  ]
#- Set CSS properties for td elements in dataframe (table data)
td_props = [
  ('font-size', '14px'),
  ('color', '#000000'),
  ]
#- Set table styles
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

# Matplotlib settings
FORMAT = ".pdf"  # File format to save in

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['text.color'] = "black"
plt.rcParams['axes.labelcolor'] = "black"
plt.rcParams['xtick.color'] = "black"
plt.rcParams['ytick.color'] = "black"

# P value formatting
LIM1 = 0.05  # P value lim1
LIM2 = 0.01  # P value lim1
LIM3 = 0.001  # P value lim1

# This function is used for rounding p values for the plots
def round_p_value(pval):
    if pval >= 0.01:
        return round(pval, 2)
    else:
        return round(pval, 3)

# Initiate method for slicing multiindex pd dataframes
idx = pd.IndexSlice

# =========================================================================
# Load and refine instabilities
# =========================================================================

# Load computed instabilities
data_instabs = pd.read_csv(sorted(glob.glob(os.path.join(
        WORKDIR, "instabilities_*")))[-1],
                   header=0, index_col=0)

# Initiate big dataframe
data_long = []

# General labels for big dataframe
gen_labs = list(itertools.product(boluses, PrePost, taus, times))

# Loop through all
for s, sub in enumerate(subjects):


    # Cast subject and task indentifiers to lists
    subject_IV = [int(sub[-3:])]*len(gen_labs)
    task_IV = [task]*len(gen_labs)

    # Extract instabilities belonging to a specific (sub)network
    new_item = data_instabs \
            .query(f'(subject == "{sub}") & (tau <= {TAU_CUTOFF})') \
            .loc[:, "whole"] \
            .to_frame()

    # Construct indexes for dataframe
    indexes = pd.MultiIndex.from_tuples(list(zip(subject_IV,
                                                 task_IV, *(zip(*gen_labs)))),
                                        names=["subject", "task", "bolus",
                                               "PrePost", "tau", "time"])
    # Add indexes to dataframe
    new_item = new_item.set_index(indexes)

    # Add current instabilities to the big dataframe
    data_long.append(new_item)

# Finalize big dataframe
data_long = pd.concat(data_long, axis=0, ignore_index=False)

# Name column(s)
columns = ["instability"]
data_long.columns = columns

# =============================================================================
# Analysis with baseline subtraction
# =============================================================================

# Baseline correction
# -------------------

# Split Pre and Post
split_PP = [P for _, P in data_long.groupby("PrePost")]

# Perform baseline correction (Post-Pre bolus)
diff = (split_PP[0].instability.droplevel("PrePost") - \
        split_PP[1].instability.droplevel("PrePost"))

diff.name = "instability_diff"

# Descriptive stats
# -----------------
stats_des = diff \
    .groupby(["task", "bolus", "tau"]) \
    .agg(["mean", "sem"]) \
    .unstack(level="bolus")

# Statistical inference
# ---------------------

# Preallocate arrays for t scores and p values
tscores = np.zeros(len(taus))
pvals = np.zeros(len(taus))

# Loop through all tau values
for t, tau in enumerate(taus):

    # Extract values
    bhb_vals = diff.loc[(slice(None), task, "BHB", tau, slice(None))]
    glc_vals = diff.loc[(slice(None), task, "GLC", tau, slice(None))]

    # Perform paired t-test
    tscores[t], pvals[t] = stats.ttest_rel(bhb_vals, glc_vals,
            nan_policy="omit")

# Export stats into html
# ----------------------

# Unite data
stats_inf = pd.DataFrame(np.concatenate((tscores.flatten()[:, None],
                            pvals.flatten()[:, None]), axis=1),
                         index=stats_des.index , columns=(("T-score", ""),
                                                          ("p value", "")))

stats_comb = pd.concat((stats_des, stats_inf), axis=1)

# Apply styles
value_formats = {
        **{stats_comb.columns[i] :  "{:.2e}" for i in range(4)},
        **{stats_comb.columns[4]: "{:#.3g}",
           stats_comb.columns[5]: "{:#.3g}"}
                 }
stats_comb_html = stats_comb.style \
  .set_table_styles(styles) \
  .format(value_formats) \
  .background_gradient(cmap=cm, subset=[col for col \
                                               in stats_comb.columns]) \
  .set_caption(f"Statistics of baseline corrected bolus instabilities N={N}") \
      .render()

# Construct HTML
if SAVE:
    with open(OUTDIR + \
          f"bolus_instab_with_baseline_subtraction_N{N}.html", "w") as file:
        file.write(stats_comb_html)

# Visualize results
# -----------------

# Plotting settings
tau_max = np.max(taus)
corrfact = -2  # Rescaling factor for plotting

FS = 1  # Fontsize multiplier
MS = 1.375  # Markersize multiplier
CS = 1.375  # Capsize multiplier

colors = [[0, 0, 0, 1], [1, 0, 0, 1]]
legend_loc = 2
markers = ['o', 's']

title = f"POST-PRE BOLUS, {task.upper()}, N={N}"

# Create figure
plt.figure(figsize=(13.5, 10))
plt.suptitle(title, y=1, fontweight='bold', fontsize=1.8*FS*16, va="bottom")

### Subplot A: barplot

# Extract data for plotting
means = np.flip(np.array(stats_des.loc[idx[task, TAU_PICK], idx["mean", :]])) \
        *10**-corrfact

sems = np.flip(np.array(stats_des.loc[idx[task, TAU_PICK], idx["sem", :]])) \
        *10**-corrfact

pvals_pick = [round_p_value(pvals[TAU_PICK-1])]

xvals = np.arange(0, len(means), 1.0)

# Initiate subplot
plt.subplot(1, 2, 1)
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(2*MS)
    plt.gca().spines[sp].set_color("black")

# Barplotting
for i in range(len(means)):
    plt.bar(xvals[i],
            means[i],
            yerr=sems[i],
            width=0.8,
            linewidth=1*MS*4,
            edgecolor=colors[i],
            ecolor=colors[i],
            facecolor=colors[i][:3]+[0.5],
            capsize=CS*10,
            error_kw=dict(lw=2*MS, capsize=CS*6, capthick=2*MS),
            zorder=i+1)

# Limits
plt.xlim([min(xvals)-0.6, max(xvals)+0.6])

ymin = np.min(means - sems)  # Lowest point
ymax = np.max(means + sems)  # Highest point

yun = 0.01*(ymax - ymin)
plt.ylim([ymin - (5)*yun, ymax + 22*yun])

# Labels
labels = ["GLU-FAST ", " KET-FAST"]
ylabel = (("Brain Network Instability ($\it{r}$) at ${\\tau}$" + \
       "={0}".format(TAU_PICK)))
plt.tick_params(axis='x', which='major', labelsize=1.8*FS*16, length=0,
                pad=20)
plt.tick_params(axis='y', which='major', labelsize=1.5*FS*16, length=15,
                direction='in')
plt.annotate(r"$\times10^{%s}$" % corrfact, xy=(0, 1),
             xycoords='axes fraction', ha='left', va='bottom',
                 fontsize=1.5*FS*16, fontweight="bold")
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ytick_spacing = 0.2  # Number of labelticks on y axis

plt.xticks(xvals, labels)
plt.xlabel("")
plt.ylabel(ylabel, fontweight='bold', fontsize=FS*1.9*16)
plt.hlines(0, min(xvals)-1, max(xvals)+1, linewidth=5, zorder=0)

# Tick adjustment
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))

# Add astrixes
combos = list(itertools.combinations(np.arange(len(labels)), 2))

for i, combo in enumerate(combos):
    curr_pval = pvals_pick[i]
    astrix = '*' if (curr_pval <= LIM1) and (curr_pval > LIM2) \
             else "**" if (curr_pval <= LIM2) and (curr_pval > LIM3) \
             else "***" if (curr_pval <= LIM3) else "n.s."
    i = combo[0]
    j = combo[1]
    color = [0, 0, 0, 1]

    x1 = xvals[i]
    x2 = xvals[j]

    y = plt.gca().get_ylim()[1] - 5*yun if abs(i-j) > 1 \
        else plt.gca().get_ylim()[1] - 12*yun

    plt.plot([x1, x2],
              [y, y],
              color=color,
              linewidth=2*MS)

    plt.annotate(astrix, xy=((x1+x2)/2, y), ha='center', va='bottom',
                 fontsize=1.9*FS*16, color=color, fontweight="normal")

### Subplot B: line plot

# Extract data for plotting
means = np.flip(
        np.array(stats_des.loc[idx[task, :], idx["mean", :]]),
        axis=1) \
        *10**-corrfact

sems = np.flip(
        np.array(stats_des.loc[idx[task, :], idx["sem", :]]),
        axis=1) \
        *10**-corrfact

pvals_plot = [
        [round_p_value(pvals[i])] for i in range(len(pvals))
        ]

xvals = np.arange(1, tau_max+1)

# Initiate subplot
plt.subplot(1, 2, 2)
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(2*MS)
    plt.gca().spines[sp].set_color("black")

# Labels
labels = ["GLU-FAST", "KET-FAST"]
ylabel = ("Brain Network Instability ($\it{r}$)")
plt.tick_params(axis='both', which='major', labelsize=1.5*FS*16,
                length=15, direction='in')
plt.annotate(r"$\times10^{%s}$" % corrfact, xy=(0, 1),
             xycoords='axes fraction', ha='left', va='bottom',
                 fontsize=1.5*FS*16, fontweight="bold")

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
xtick_spacing = 5  # Number of labelticks on x axis
ytick_spacing = 0.2  # Number of labelticks on y axis

plt.xlabel(str("Time Delay ${\\tau}$\n[Each interval=24s]"),
           fontweight='bold', fontsize=1.8*FS*16, labelpad=20)
plt.ylabel(ylabel, fontweight='bold', fontsize=1.9*FS*16, labelpad=10)

# Plot with errorbars
for i in range(len(labels)):
    line = plt.errorbar(xvals, means[:, i],
        yerr=sems[:, i],
        marker=markers[i],
        capsize=CS*4,
        markersize=MS*9,
        linewidth=MS*1.75,
        elinewidth=MS*1.75,
        capthick=CS*1.25,
        color=colors[i],
        label=labels[i])

# Legend
plt.legend(prop={'size': 1.7*FS*16, 'weight': 'bold'},
           framealpha=0,
           loc=legend_loc)

# Limits
plt.xlim([0, tau_max + 1])

ymin = np.min(means - sems)  # Lowest point
ymax = np.max(means + sems)  # Highest point
minp = np.array(pvals_plot).flatten().min()
maxast = 3 if minp <= LIM3 else 2 if minp <= LIM2 \
    else 1 if minp <= LIM1 else 0  # Maximum number of astrixes
yun = 0.01*(ymax - ymin)

plt.ylim([ymin - (3*maxast + 3)*yun, ymax + 22*yun])

# Tick adjustment
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))

# Significance astrixes
for i, x in enumerate(xvals):
    y = plt.gca().get_ylim()[0] + (1.8 + 3*maxast)*yun
    curr_pval = pvals_plot[i][0]  # Current pval
    astrix = '*' if (curr_pval <= LIM1) and (curr_pval > LIM2) \
             else "*\n*" if (curr_pval <= LIM2) and (curr_pval > LIM3) \
             else "*\n*\n*" if (curr_pval <= LIM3) else ""
    plt.annotate(astrix,  xy=[x, y], ha='center', va='top',
                 fontsize=1.9*FS*16, linespacing=0.25,
                 fontweight="normal")

# Layout
plt.tight_layout(w_pad=3)
plt.subplots_adjust(wspace=0.4)

if SAVE:
    plt.savefig(
            OUTDIR + f"bolus_instab_with_baseline_subtraction_N{N}" + FORMAT,
            transparent=True, bbox_inches='tight',
            )

plt.close("all")

# =============================================================================
# Analysis without baseline subtraction
# =============================================================================

conds = ["GLC|pre", "GLC|post", "BHB|pre", "BHB|post"]  # Four conditions
combos = list(itertools.combinations(np.arange(len(conds)), 2))  # Combinations
target_comp = ["GLC|post", "BHB|post"]  # Highlighted comparison shown on plot

# Descriptive stats
# -----------------
stats_des = data_long \
    .query(f'task == "{task}"') \
    .reset_index("task", drop=True) \
    ["instability"] \
    .groupby(["subject", "bolus", "PrePost", "tau"]) \
    .agg(["mean"]) \
    .xs("mean", axis=1, drop_level=True) \
    .groupby(["bolus", "PrePost", "tau"]) \
    .agg(["mean", "sem"]) \
    .unstack(["bolus", "PrePost"])

# Statistical inference
# ---------------------

# List for storing statistical inference results
stats_res = []

# Loop through all tau values
for t, tau in enumerate(taus):

    # Extract data
    data_stats_pre = data_long \
        .query(f'task == "{task}"') \
        .reset_index("task", drop=True) \
        .instability \
        .to_frame() \
        .query('tau == {}'.format(tau)) \
        .reset_index("tau", drop=True) \

    # Construct new index object for conditions
    mtx = pd.MultiIndex.from_arrays([
            data_stats_pre.index.map(lambda x: '|'.join([str(i) \
                    for i in [x[j] for j in [1, 2]]])).to_list(), \
            data_stats_pre.index.map(lambda x: '|'.join([str(i) \
                    for i in [x[j] for j in [0]]])).to_list()],
        names=["cond", "sub"])

    # Add new indexes for conditions
    data_stats = data_stats_pre \
        .set_index(mtx) \
        .reset_index() \
        .dropna() \
        .sort_values(by=["cond", "sub"]) \
        .reset_index()

    data_stats["index"] = \
        np.array([list(range(int(data_stats["cond"].shape[0]/len(conds))))] \
                 *len(conds)).flatten()

    # Perform repeated measures anova
    ind_col = "index"
    lab_col = "cond"
    val_col = "instability"

    anova_output = pg.rm_anova(data=data_stats,
                              dv=val_col,
                              within=lab_col,
                              subject=ind_col,
                              detailed=True) \

    if anova_output["p-unc"][0] < 0.05:
        res_an = anova_output.loc[:, idx["F", "p-unc"]]
        res_pwc = pd.DataFrame(None, columns=["A", "B", "T", "p-corr"])

        # Protected post hoc t-test (LSD)
        degf = anova_output.loc[1, "DF"]
        SSE = anova_output.loc[1, "SS"]
        MSE = SSE/degf

        n = len(conds)

        combos_labels = list(itertools.combinations(conds, 2))

        for i, combo in enumerate(combos_labels):

            n_a = len(data_stats.query(f'cond ==  "{combo[0]}"'))
            n_b = len(data_stats.query(f'cond ==  "{combo[1]}"'))

            std_err = np.sqrt(MSE*(1/n_a + 1/n_b))

            y_a = np.mean(data_stats.query(f'cond ==  "{combo[0]}"')[val_col])
            y_b = np.mean(data_stats.query(f'cond ==  "{combo[1]}"')[val_col])

            dif = np.abs(y_a - y_b)

            t_val = dif/std_err
            p_val = stats.t.sf(t_val, degf)*2

            res_pwc = pd.concat((res_pwc, \
                     pd.DataFrame([combo[0], combo[1], t_val, p_val], \
                                  index=["A", "B", "T", "p-corr"]).T), axis=0)

    else:
        res_an = anova_output.loc[:, idx["F", "p-unc"]]
        res_pwc = None

    # Store statistical inference results (an: anova, pwc: pairwise comparisons)
    stats_res.append({"an": res_an, "pwc": res_pwc})


# Export stats into html
# ----------------------

# Unite data
fscores_an = [stats_res[i]["an"]["F"].values[0] for i in range(len(taus))]
pvals_an = [stats_res[i]["an"]["p-unc"].values[0] for i in range(len(taus))]

stats_inf_an = pd.DataFrame(zip(fscores_an, pvals_an),
                         index=stats_des.index , columns=(("F-score", ""),
                                                          ("p value", "")))

tscores_pwc = np.array([stats_res[i]["pwc"]["T"].values \
               if stats_res[i]["pwc"] is not None \
               else np.array([np.nan]*len(combos)) \
                   for i in range(len(taus))])

pvals_pwc = np.array([stats_res[i]["pwc"]["p-corr"].values \
               if stats_res[i]["pwc"] is not None \
               else np.array([np.nan]*len(combos)) \
                   for i in range(len(taus))])

combo_labels = [[conds[combo[0]], conds[combo[1]]] for combo in combos]
tscore_labels = ["T-score - " + " vs. ".join(combo_label) \
                 for combo_label in combo_labels]
pval_labels = ["p value - " + " vs. ".join(combo_label) \
               for combo_label in combo_labels]


stats_comb = pd.concat((stats_des,
                        stats_inf_an,
                        pd.DataFrame(tscores_pwc, columns=tscore_labels, index=taus),
                        pd.DataFrame(pvals_pwc, columns=pval_labels, index=taus)),
                       axis=1)


# Apply styles
value_formats = {
        **{stats_comb.columns[i] :  "{:.2e}" for i in range(2*len(conds))},
        **{stats_comb.columns[i]: "{:#.3g}" \
           for i in range(2*len(conds), 2*len(conds) + 2*len(combos) + 2)}
            }

stats_comb_html = stats_comb.style \
  .set_table_styles(styles) \
  .format(value_formats) \
  .background_gradient(cmap=cm, subset=[col for col \
                                               in stats_comb.columns]) \
  .set_caption(f"Statistics of bolus instabilities N={N}") \
      .render()

# Construct HTML
if SAVE:
    with open(OUTDIR + \
          f"bolus_instab_without_baseline_subtraction_N{N}.html", "w") as file:
        file.write(stats_comb_html)

# Visualize results
# -----------------

# Plotting settings
tau_max = np.max(taus)
corrfact = 0  # Rescaling factor for plotting

FS = 1  # Fontsize multiplier
MS = 1.375  # Markersize multiplier
CS = 1.375  # Capsize multiplier

colors = [[0.4, 0.4, 0.4, 1], [0, 0, 0, 1], [1, 0.5, 0, 1], [1, 0, 0, 1]]
legend_loc = 2
markers = ['o', 'D', 's', 'v']

title = f"BOLUS FOUR CONDITIONS, {task.upper()}, N={N}"

# Create figure
plt.figure(figsize=(13.5, 10))
plt.suptitle(title, y=1, fontweight='bold', fontsize=1.8*FS*16, va="bottom")

### Subplot A: barplot

# Extract data for plotting
means = np.flip(np.array(stats_des.loc[idx[TAU_PICK], idx["mean", :, :]])) \
        *10**-corrfact

sems = np.flip(np.array(stats_des.loc[idx[TAU_PICK], idx["sem", :, :]])) \
        *10**-corrfact

pvals_pick = stats_res[TAU_PICK-1]["pwc"] \
    if stats_res[TAU_PICK-1]["pwc"] is not None else None

xvals = np.arange(0, len(means), 1.0)

# Initiate subplot
plt.subplot(1, 2, 1)
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(2*MS)
    plt.gca().spines[sp].set_color("black")

# Bars
for i in range(len(means)):
    plt.bar(xvals[i],
            means[i],
            yerr=sems[i],
            width=0.8,
            linewidth=1*MS*4,
            edgecolor=colors[i],
            ecolor=colors[i],
            facecolor=colors[i][:3]+[0.5],
            capsize=CS*10,
            error_kw=dict(lw=2*MS, capsize=CS*6, capthick=2*MS),
            zorder=i+1)

# Labels
labels = ["FAST\n-ING\nPRE\nGLC", "GLC\nBOLUS",
          "FAST\n-ING\nPRE\nBHB", "BHB\nBOLUS"]
ylabel = (("Brain Network Instability ($\it{r}$) at ${\\tau}$" + \
        "={0}".format(TAU_PICK)))
plt.tick_params(axis='x', which='major', labelsize=1.4*FS*16, length=0,
                pad=20)
plt.tick_params(axis='y', which='major', labelsize=1.5*FS*16, length=15,
                direction='in')
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ytick_spacing = 0.01  # Number of labelticks on y axis

plt.xticks(xvals, labels)
plt.ylabel(ylabel, fontweight='bold', fontsize=FS*1.9*16)
plt.hlines(0, min(xvals)-1, max(xvals)+1, linewidth=5, zorder=0)

# Limits
plt.xlim([min(xvals)-0.6, max(xvals)+0.6])

ymin = np.min(means - sems)  # Lowest point
ymax = np.max(means + sems)  # Highest point
yun = 0.01*(ymax - ymin)

plt.ylim([ymin - 5*yun, ymax + 65*yun])

# Tick adjustment
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))

# Add astrixes
def label_diff(coords, yun, text):
    i = coords[0]
    j = coords[1]
    color = [0, 0, 0, 1]

    x1 = xvals[i]
    x2 = xvals[j]

    adjs = np.array([[0, 30.5, 14.5,  7.5],
                     [0,    0, 28.5, 21.5],
                     [0,    0,    0, 32.5],
                     [0,    0,    0,    0]])

    y = plt.gca().get_ylim()[1] - adjs[i,j]*yun


    plt.plot([x1, x2],
              [y, y],
              color=color,
              linewidth=1*MS)

    plt.annotate(text, xy=((x1+x2)/2, y), ha='center', va='bottom',
                  fontsize=1.9*FS*16, color=color, fontweight="normal")

yun = np.diff(plt.gca().get_ylim())[0]/100

for i, combo in enumerate(combos):
    curr_pval = round_p_value(pvals_pick.query(
        '(A =="{0}" & B =="{1}") or' \
        '(A =="{1}" & B =="{0}")' \
          .format(conds[combo[0]], conds[combo[1]])) \
          ["p-corr"].values[0]) if pvals_pick is not None else 1

    astrix = '*' if (curr_pval <= LIM1) and (curr_pval > LIM2) \
              else "**" if (curr_pval <= LIM2) and (curr_pval > LIM3) \
              else "***" if (curr_pval <= LIM3) else "n.s."
    label_diff(combo, yun, astrix)

### Subplot B: line plot

# Extract data for plotting
means = np.flip(
        np.array(stats_des.loc[:, idx["mean", :, :]]), axis=1)*10**-corrfact

sems = np.flip(
        np.array(stats_des.loc[:, idx["sem", :, :]]), axis=1)*10**-corrfact

pvals_plot = [stats_res[t]["pwc"].query( \
        '(A =="{0}" & B =="{1}") or' \
        '(A =="{1}" & B =="{0}")' \
        .format(*target_comp))["p-corr"].values[0] \
        if (stats_res[t]["an"].loc[0, "p-unc"] < 0.05) else 1 \
        for t in np.arange(tau_max)]

xvals = np.arange(1, tau_max+1)

# Initiate subplot
plt.subplot(1, 2, 2)
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(2*MS)
    plt.gca().spines[sp].set_color("black")

# Labels
labels = ["FASTING\nPRE GLC", "GLC BOLUS", "FASTING\nPRE BHB", "BHB BOLUS"]
ylabel = ("Brain Network Instability ($\it{r}$)")
plt.tick_params(axis='both', which='major', labelsize=1.3*FS*16,
                length=15, direction='in')

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
xtick_spacing = 5  # Number of labelticks on x axis
ytick_spacing = 0.01 # Number of labelticks on y axis

plt.xlabel(str("Time Delay ${\\tau}$\n[Each interval=24s]"),
            fontweight='bold', fontsize=1.8*FS*16, labelpad=20)
plt.ylabel(ylabel, fontweight='bold', fontsize=1.9*FS*16, labelpad=10)

# Plot with errorbars
for i in range(len(labels)):
    line = plt.errorbar(xvals, means[:, i],
        yerr=sems[:, i],
        marker=markers[i],
        capsize=CS*4,
        markersize=MS*9,
        linewidth=MS*1.75,
        elinewidth=MS*1.75,
        capthick=CS*1.25,
        color=colors[i],
        label=labels[i])

# Legend
plt.legend(prop={'size': 1.4*FS*16, 'weight': 'bold'},
            framealpha=0,
            loc=legend_loc)

# Limits
plt.xlim([0, tau_max + 1])

ymin = np.min(means - sems)  # Lowest point
ymax = np.max(means + sems)  # Highest point
minp = np.array(pvals_plot).min()
maxast = 3 if minp <= LIM3 else 2 if minp <= LIM2 \
    else 1 if minp <= LIM1 else 0  # Maximum number of astrixes
yun = 0.01*(ymax - ymin)

plt.ylim([ymin - (3*maxast + 3)*yun, ymax + 30*yun])

# Tick adjustment
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))

# Significance astrixes
for i, x in enumerate(xvals):
    y = plt.gca().get_ylim()[0] + (1.8 + 3*maxast)*yun
    curr_pval = pvals_plot[i]  # Current pval
    astrix = '*' if (curr_pval <= LIM1) and (curr_pval > LIM2) \
              else "*\n*" if (curr_pval <= LIM2) and (curr_pval > LIM3) \
              else "*\n*\n*" if (curr_pval <= LIM3) else ""
    plt.annotate(astrix,  xy=[x, y], ha='center', va='top',
                  fontsize=1.9*FS*16, linespacing=0.25,
                  fontweight="normal")

# Layout
plt.tight_layout(w_pad=3)


if SAVE:
    plt.savefig(
            OUTDIR + f"bolus_instab_without_baseline_subtraction_N{N}" \
            + FORMAT, transparent=True, bbox_inches='tight'
            )

plt.close("all")
