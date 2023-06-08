import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import scipy.stats as stats
from scipy.stats import levene

import seaborn as sns
import matplotlib.pyplot as plt


import os
import seaborn as sns

def plot_variable_pairs(train):
    """
    Generate pairwise scatter plots for the variables in the given DataFrame.

    Parameters:
        train (pandas DataFrame): The DataFrame containing the variables.

    The function uses seaborn's pairplot to create pairwise scatter plots for all combinations of variables in the DataFrame.

    The argument `kind="reg"` specifies that regression lines should be plotted on the scatter plots.

    The argument `corner=True` sets the corner plot to display only the lower triangle of the pairwise scatter plots,
    resulting in a more compact visualization.

    The argument `plot_kws={'line_kws': {'color': 'red'}}` is used to set the color of the regression lines to red.

    The function displays the plot using `plt.show()` and does not return any value.
    """
    sns.set(style="ticks")
    sns.pairplot(train, kind="reg", corner = True, plot_kws={'line_kws': {'color': 'red'}})
    plt.show()

def plot_categorical_and_continuous_vars(dataframe, categorical_var, continuous_var):
    """
    Generate multiple plots to visualize the relationship between a categorical variable and a continuous variable.

    Parameters:
        dataframe (pandas DataFrame): The DataFrame containing the data.
        categorical_var (str): The name of the categorical variable.
        continuous_var (str): The name of the continuous variable.

    The function generates three plots: a box plot, a strip plot, and a bar plot.

    The box plot displays the distribution of the continuous variable across different categories of the categorical variable.
    The x-axis represents the categorical variable, and the y-axis represents the continuous variable.

    The strip plot shows individual data points as scattered points, providing an overview of the distribution of the continuous variable for each category of the categorical variable.

    The bar plot displays the average value of the continuous variable for each category of the categorical variable.

    Each plot is displayed using `plt.show()`.

    The function does not return any value.
    """
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Box Plot of {continuous_var} vs {categorical_var}")
    plt.show()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Strip Plot of {continuous_var} vs {categorical_var}")
    plt.show()

    # Swarm plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Bar Plot of {continuous_var} vs {categorical_var}")
    plt.show()

def target_viz(train):
    """
    Visualizes the target variable distribution in the training dataset.

    Args:
        train (pd.DataFrame): The training dataset containing the target variable.

    The function creates a countplot to display the distribution of the target variable 'made_cut'.
    It provides a visual representation of the number of occurrences for each category of 'made_cut'.
    The plot is annotated with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        target_viz(train)
        # Displays the countplot for the 'made_cut' variable in the training dataset.
    """
    sns.countplot(data=train, x= 'made_cut')
    plt.title('Target visualization')
    
    # Customize x-axis labels
    plt.gca().set_xticklabels(['No', 'Yes'])

    # Customize legend labels
    plt.legend(['No', 'Yes'])
    plt.xlabel('Made cut')
    plt.show() 

def approach_viz(train):
    """
    Visualizes the distribution of strokes gained approach in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained approach ('sg_app_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's
    histplot with 'sg_app_prev' on the x-axis and 'made_cut' as the hue. The plot
    is annotated with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        approach_viz(train)
        # Displays a histogram comparing strokes gained approach between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained approach in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_app_prev', hue='made_cut')
    plt.title('Strokes gained approach between those who made cut and did not')
    plt.xlabel('Strokes gained approach')
    # Customize legend labels
    plt.legend(['Yes', 'No'])
    plt.show()

def approach_stats(train):
    """
    Performs statistical analysis on strokes gained approach between golfers who
    made the cut and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_app_prev' as the strokes gained approach values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained approach for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_app_prev' for the analysis to be performed.

    Example:
        approach_stats(train)
        # Performs statistical analysis on strokes gained approach between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained approach
    made_cut_yes = train[train.made_cut == 1].sg_app_prev
    made_cut_no = train[train.made_cut == 0].sg_app_prev

    # compare means of the two groups
    print('Mean strokes gained approach of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained approach of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")

def off_the_tee_viz(train):
    """
    Visualizes the distribution of strokes gained off-the-tee in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained off-the-tee ('sg_ott_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's histplot
    with 'sg_ott_prev' on the x-axis and 'made_cut' as the hue. The plot is annotated
    with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        off_the_tee_viz(train)
        # Displays a histogram comparing strokes gained off-the-tee between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained off-the-tee in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_ott_prev', hue='made_cut')
    plt.title('Strokes gained off-the-tee between those who made cut and did not')
    plt.xlabel('Strokes gained off-the-tee')
     # Customize legend labels
    plt.legend(['Yes','No'])
    plt.show()

def off_the_tee_stats(train):
    """
    Performs statistical analysis on strokes gained off-the-tee between golfers who
    made the cut and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_ott_prev' as the strokes gained off-the-tee values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained off-the-tee for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_ott_prev' for the analysis to be performed.

    Example:
        off_the_tee_stats(train)
        # Performs statistical analysis on strokes gained off-the-tee between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained off-the-tee
    made_cut_yes = train[train.made_cut == 1].sg_ott_prev
    made_cut_no = train[train.made_cut == 0].sg_ott_prev

    # compare means of the two groups
    print('Mean strokes gained off-the-tee of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained off-the-tee of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")

def t2g_viz(train):
    """
    Visualizes the distribution of strokes gained tee-to-green in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained tee-to-green ('sg_t2g_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's histplot
    with 'sg_t2g_prev' on the x-axis and 'made_cut' as the hue. The plot is annotated
    with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        t2g_viz(train)
        # Displays a histogram comparing strokes gained tee-to-green between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained tee-to-green in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_t2g_prev', hue='made_cut')
    plt.title('Strokes gained tee-to-green between those who made cut and did not')
    plt.xlabel('Strokes gained tee-to-green')
    plt.legend(['Yes','No'])
    plt.show()

def t2g_stats(train):
    """
    Performs statistical analysis on strokes gained tee-to-green between golfers who
    made the cut and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_t2g_prev' as the strokes gained tee-to-green values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained tee-to-green for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_t2g_prev' for the analysis to be performed.

    Example:
        t2g_stats(train)
        # Performs statistical analysis on strokes gained tee-to-green between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained tee-to-green
    made_cut_yes = train[train.made_cut == 1].sg_t2g_prev
    made_cut_no = train[train.made_cut == 0].sg_t2g_prev

    # compare means of the two groups
    print('Mean strokes gained tee-to-green of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained tee-to-green of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")

def total_viz(train):
    """
    Visualizes the distribution of strokes gained total in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained total ('sg_total_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's histplot
    with 'sg_total_prev' on the x-axis and 'made_cut' as the hue. The plot is annotated
    with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        total_viz(train)
        # Displays a histogram comparing strokes gained total between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained total in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_total_prev', hue='made_cut')
    plt.title('Strokes gained total between those who made cut and did not')
    plt.xlabel('Strokes gained total')
    plt.legend(['Yes','No'])
    plt.show()

def total_stats(train):
    """
    Performs statistical analysis on strokes gained total between golfers who made the cut
    and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_total_prev' as the strokes gained total values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained total for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_total_prev' for the analysis to be performed.

    Example:
        total_stats(train)
        # Performs statistical analysis on strokes gained total between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained total
    made_cut_yes = train[train.made_cut == 1].sg_total_prev
    made_cut_no = train[train.made_cut == 0].sg_total_prev

    # compare means of the two groups
    # compare means of the two groups
    print('Mean strokes gained total of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained total of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")

def approach_viz2(train):
    """
    Visualizes the distribution of strokes gained approach in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained approach ('sg_app_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's
    histplot with 'sg_app_prev' on the x-axis and 'made_cut' as the hue. The plot
    is annotated with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        approach_viz(train)
        # Displays a histogram comparing strokes gained approach between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained approach in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_app_2wk_avg', hue='made_cut')
    plt.title('Strokes gained approach between those who made cut and did not')
    plt.xlabel('Strokes gained approach')
    plt.legend(['Yes','No'])
    plt.show()

def approach_stats2(train):
    """
    Performs statistical analysis on strokes gained approach between golfers who
    made the cut and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_app_prev' as the strokes gained approach values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained approach for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_app_prev' for the analysis to be performed.

    Example:
        approach_stats(train)
        # Performs statistical analysis on strokes gained approach between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained approach
    made_cut_yes = train[train.made_cut == 1].sg_app_2wk_avg
    made_cut_no = train[train.made_cut == 0].sg_app_2wk_avg

    # compare means of the two groups
    print('Mean strokes gained approach of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained approach of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")

def off_the_tee_viz2(train):
    """
    Visualizes the distribution of strokes gained off-the-tee in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained off-the-tee ('sg_ott_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's histplot
    with 'sg_ott_prev' on the x-axis and 'made_cut' as the hue. The plot is annotated
    with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        off_the_tee_viz(train)
        # Displays a histogram comparing strokes gained off-the-tee between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained off-the-tee in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_ott_2wk_avg', hue='made_cut')
    plt.title('Strokes gained off-the-tee between those who made cut and did not')
    plt.xlabel('Strokes gained off-the-tee')
    plt.legend(['Yes','No'])
    plt.show()

def off_the_tee_stats2(train):
    """
    Performs statistical analysis on strokes gained off-the-tee between golfers who
    made the cut and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_ott_prev' as the strokes gained off-the-tee values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained off-the-tee for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_ott_prev' for the analysis to be performed.

    Example:
        off_the_tee_stats(train)
        # Performs statistical analysis on strokes gained off-the-tee between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained off-the-tee
    made_cut_yes = train[train.made_cut == 1].sg_ott_2wk_avg
    made_cut_no = train[train.made_cut == 0].sg_ott_2wk_avg

    # compare means of the two groups
    print('Mean strokes gained off-the-tee of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained off-the-tee of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")

def t2g_viz2(train):
    """
    Visualizes the distribution of strokes gained tee-to-green in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained tee-to-green ('sg_t2g_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's histplot
    with 'sg_t2g_prev' on the x-axis and 'made_cut' as the hue. The plot is annotated
    with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        t2g_viz(train)
        # Displays a histogram comparing strokes gained tee-to-green between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained tee-to-green in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_t2g_2wk_avg', hue='made_cut')
    plt.title('Strokes gained tee-to-green between those who made cut and did not')
    plt.xlabel('Strokes gained tee-to-green')
    plt.legend(['Yes','No'])
    plt.show()

def t2g_stats2(train):
    """
    Performs statistical analysis on strokes gained tee-to-green between golfers who
    made the cut and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_t2g_prev' as the strokes gained tee-to-green values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained tee-to-green for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_t2g_prev' for the analysis to be performed.

    Example:
        t2g_stats(train)
        # Performs statistical analysis on strokes gained tee-to-green between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained tee-to-green
    made_cut_yes = train[train.made_cut == 1].sg_t2g_2wk_avg
    made_cut_no = train[train.made_cut == 0].sg_t2g_2wk_avg

    # compare means of the two groups
    print('Mean strokes gained tee-to-green of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained tee-to-green of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")

def total_viz2(train):
    """
    Visualizes the distribution of strokes gained total in the previous week
    among golfers who made the cut and those who did not in the next week.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function creates a histogram with overlaid bars to compare the distribution
    of strokes gained total ('sg_total_prev') between two groups: golfers who made
    the cut and golfers who did not. The histogram is displayed using seaborn's histplot
    with 'sg_total_prev' on the x-axis and 'made_cut' as the hue. The plot is annotated
    with a title, x-label, and displayed using matplotlib.pyplot.show().

    Example:
        total_viz(train)
        # Displays a histogram comparing strokes gained total between golfers who
        # made the cut and those who did not in the next week.
    """
    # visualize strokes gained total in the previous week to making the cut in the next week
    sns.histplot(data=train, x='sg_total_2wk_avg', hue='made_cut')
    plt.title('Strokes gained total between those who made cut and did not')
    plt.xlabel('Strokes gained total')
    plt.legend(['Yes','No'])
    plt.show()

def total_stats2(train):
    """
    Performs statistical analysis on strokes gained total between golfers who made the cut
    and those who did not, and provides test results and conclusions.

    Args:
        train (pd.DataFrame): The training dataset containing relevant data.

    The function performs the following steps:
    1. Creates two subgroups based on 'made_cut' values: 'made_cut_yes' for golfers
       who made the cut (with 'sg_total_prev' as the strokes gained total values),
       and 'made_cut_no' for golfers who did not make the cut.
    2. Prints the average strokes gained total for both groups.
    3. Sets the significance level (alpha) for subsequent statistical tests.
    4. Performs a Levene test to verify the equality of variances between the two groups,
       and prints the result.
    5. Performs an independent two-sample t-test (assuming unequal variances) and prints
       the result, along with a conclusion based on the p-value and test statistic.

    Note:
        - The 'train' DataFrame is expected to contain the columns 'made_cut' and
          'sg_total_prev' for the analysis to be performed.

    Example:
        total_stats(train)
        # Performs statistical analysis on strokes gained total between golfers who
        # made the cut and those who did not, and provides the test results and conclusions.
    """
    # make sub groups of those who made or didn't make the cut based on strokes gained total
    made_cut_yes = train[train.made_cut == 1].sg_total_2wk_avg
    made_cut_no = train[train.made_cut == 0].sg_total_2wk_avg

    # compare means of the two groups
    # compare means of the two groups
    print('Mean strokes gained total of players who made cut:', round(made_cut_yes.mean(),2))
    print('Mean strokes gained total of players who did not make the cut:', round(made_cut_no.mean(),2))

    # set alpha for all following statistics tests
    alpha = 0.05

    #verify equal variance
    stat, p = levene(made_cut_yes, made_cut_no)
    if p < alpha:
        print("Variance is not equal (ttest_ind equal_var set to False)")
    else:
        print("Equal variances (ttest_ind equal_var set to True)")

    #run stats test
    t, p = stats.ttest_ind(made_cut_yes, made_cut_no, equal_var=False)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")