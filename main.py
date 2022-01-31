# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
#data modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import datetime
from datetime import datetime
from statannot import add_stat_annotation
import dataframe_image as dfi
from dateutil.relativedelta import relativedelta
from datetime import date


###############################################################################
# Functions/code for data analysis
label_size = 10
plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size 


# %%
#get path for data
pwd = os.getcwd()
dataset_path = pwd + '\\wbdata_hdr.csv'

#make folder for figures
try:
    os.makedirs(pwd + '\\figures')
except OSError as e:
    print('Folder allready exists')


# %%
#convert to mm-yyyy
def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    return date.replace(month=m, year=y)

#define dates for date dependant analysis
today = date.today() #current date
#calculate months
month1 = (date.today()+relativedelta(months=-4)).strftime('%m-%Y') #4 months ago
month2 = (date.today()+relativedelta(months=-3)).strftime('%m-%Y') #3 months ago
month3 = (date.today()+relativedelta(months=-2)).strftime('%m-%Y') #last month -2
month4 = (date.today()+relativedelta(months=-1)).strftime('%m-%Y') #last month


# %%
#import data
data = pd.read_csv(dataset_path)

#clean data from unwanted values
clean_data = data.drop(data[(data['groupName'] == 'General Public')|(data['groupName'] == 'Northumbria University')|(data['occupation'] == 'Student')|(data['occupation'] == 'Lecturer')].index)

#slider data
slider_data = pd.DataFrame(clean_data, columns= ['userID','gender','age','occupation','groupName','ethnicity','region','housing','livingarrangement','hadcovid','physical','emotional','mental','relational','meaning_purpose','critical_kind','fearfull_safe', 'torubled_actions','value_life','impulse_harm','fear_others','quality_sleep','drugs_alcohol','m_compassion','m_burnout','m_perception','m_flashbacks','m_avoidance','m_dissociation','m_bodilysymptoms','m_intrusivethoughts','anxious_worried','no_intrest','emotion_value','activation_value','diaryEntryDate'])
#rename
slider_data = slider_data.rename(columns={'groupName': 'staff_group',
                                          'fearfull_safe': 'fearful_safe',
                                          'torubled_actions': 'troubled_actions', 
                                          'm_avoidance': 'avoidance', 
                                          'm_bodilysymptoms': 'bodily_symptoms',
                                          'm_intrusivethoughts': 'intrusive_thoughts',
                                          'm_compassion': 'compassion', 
                                          'm_burnout': 'burnout', 
                                          'm_perception': 'perception',
                                          'm_flashbacks': 'flashbacks', 
                                          'm_dissociation': 'dissociation',
                                          'no_intrest': 'no_interest'})
#demographics dataframe
demographics_data = pd.DataFrame(slider_data, columns= ['userID','gender','age','occupation','staff_group','ethnicity','region','housing','livingarrangement','hadcovid'])
demographics_data.drop_duplicates(inplace=True)

#create df with reformatted date
newDateFormat = slider_data.copy() #copy
newDateFormat['diaryEntryDate'] = pd.to_datetime(newDateFormat['diaryEntryDate'], format="%d/%m/%Y") #convert to datetime
newDateFormat['entryMonth'] = newDateFormat['diaryEntryDate'].dt.strftime('%m-%Y') #convert to mm-yyyy


# %%
def steps_taken(df):
    steps_data = pd.DataFrame(df, columns= ['userID', 'stepsTaken'])
    steps_data.dropna(inplace=True)
    steps_data.columns = steps_data.columns.str.strip()
    #steps_data
    steps_takenGrouped = steps_data[['stepsTaken', 'userID']].groupby(['stepsTaken']).agg(['count'])
    steps_takenGrouped.columns = ['Count']
    steps_takenGrouped.sort_values('Count', ascending=False, inplace=True)
    #steps_takenGrouped
    plt.figure(figsize=(20, 15))
    intervention_plot = steps_takenGrouped.plot.barh(rot=0,fontsize=13)
    plt.ylabel("",fontsize=18)
    plt.xlabel("Number or responses",fontsize=18)
    plt.title("Interventions Taken",fontsize=20)
    plt.xticks(wrap = False)
    plt.grid(axis = 'x')
    plt.tight_layout()
    #save fig
    plt.savefig(pwd + '\\figures\\interventionData.png')


# %%
def entriesPerMonth(df):
    #group and count
    countMonthly = df.groupby('entryMonth').size().reset_index().rename(columns={0: 'count_entries'})
    #plot into bar chart
    plt.figure(figsize=(30, 20))
    entriesPerMonthBar = countMonthly.plot.bar(x='entryMonth',y='count_entries',rot=90)
    plt.ylabel("Number of entries",fontsize=18)
    plt.xlabel("Month",fontsize=18)
    plt.title("Number of entries per month",fontsize=20)
    plt.grid(axis = 'y')
    plt.tight_layout()
    plt.savefig(pwd + '\\figures\\entriesPerMonth.png')
    dfi.export(countMonthly.style.hide_index(), pwd + '\\figures\\monthlyEntriesCount.png')


# %%
#top 5 and lowest 5
def top5_low5(df):
    #prepare dataframe
    #get columns
    id_vars = list(df.columns)[:10]
    id_vars.append('entryMonth')
    value_vars = list(df.columns)[10:35]
    #melt dataframe
    dataset_melted = df.melt(id_vars=id_vars, value_vars = value_vars, var_name="Variable", value_name="Score")
    
    #get dates
    last_month = (date.today()+relativedelta(months=-1)).strftime('%m-%Y') #last month -2
    prev_month = (date.today()+relativedelta(months=-2)).strftime('%m-%Y') #last month

    ###################################
    #positive items
    ###################################
    #previous month
    countPositiveScores1 = dataset_melted.where((dataset_melted['Score'] > 0) & (dataset_melted['entryMonth'] < prev_month)).groupby('Variable').size().reset_index().rename(columns={0: 'prev_positive'})
    #sort and rank
    countPositiveScores1.sort_values(by=['prev_positive'], ascending=False, inplace=True)
    countPositiveScores1["prev_rank"] = countPositiveScores1["prev_positive"].rank(ascending=False)
    #latest month
    countPositiveScores = dataset_melted.where((dataset_melted['Score'] > 0) & (dataset_melted['entryMonth'] < last_month)).groupby('Variable').size().reset_index().rename(columns={0: 'latest_positive'})
    #sort and rank
    countPositiveScores.sort_values(by=['latest_positive'], ascending=False, inplace=True)
    countPositiveScores["latest_rank"] = countPositiveScores["latest_positive"].rank(ascending=False)

    #join positive old and latest
    positive_rankings = pd.merge(left=countPositiveScores1, right=countPositiveScores, how="left", left_on="Variable", right_on="Variable")
    positive_rankings.sort_values(by=['latest_positive'], ascending=False, inplace=True)
    positive_rankings["rank_change"] = positive_rankings["prev_rank"] - positive_rankings["latest_rank"]
    #select top 10
    positive_final = positive_rankings.head(10)
    #save df
    dfi.export(positive_final.style.hide_index(), pwd + '\\figures\\top10_positive.png')
    positive_final

    ###################################
    #negative items
    ###################################
    #previous month
    countNegativeScores1 = dataset_melted.where((dataset_melted['Score'] < 0) & (dataset_melted['entryMonth'] < prev_month)).groupby('Variable').size().reset_index().rename(columns={0: 'prev_negative'})
    countNegativeScores1.sort_values(by=['prev_negative'], ascending=False, inplace=True)
    countNegativeScores1["prev_rank"] = countNegativeScores1["prev_negative"].rank(ascending=False)
    #latest month
    countNegativeScores = dataset_melted.where((dataset_melted['Score'] < 0) & (dataset_melted['entryMonth'] < last_month)).groupby('Variable').size().reset_index().rename(columns={0: 'latest_negative'})
    countNegativeScores.sort_values(by=['latest_negative'], ascending=False, inplace=True)
    countNegativeScores["latest_rank"] = countNegativeScores["latest_negative"].rank(ascending=False)

    #join negative old and latest
    negative_rankings = pd.merge(left=countNegativeScores1, right=countNegativeScores, how="left", left_on="Variable", right_on="Variable")
    negative_rankings.sort_values(by=['latest_negative'], ascending=False, inplace=True)
    negative_rankings["rank_change"] = negative_rankings["prev_rank"] - negative_rankings["latest_rank"]
    #select top 10
    negative_final = negative_rankings.head(10)
    #save df
    dfi.export(negative_final.style.hide_index(), pwd + '\\figures\\top10_negative.png')
    #return positive_final, negative_final


# %%
# function to create demographic bar chart
def demographics_bar(var, rotation = 0):
    #demo_data - demographics data
    demo_data = demographics_data[[var, 'userID']].groupby([var]).agg(['count'])
    demo_data.columns = ['Count']

    #count num of groups in demographic to prevent overlap
    number_of_groups = len(demo_data.index)
    #plt.figure(figsize=(20, 15))

    #only show 15 largest
    if number_of_groups > 15:
        demo_data.sort_values('Count', ascending=False, inplace=True)
        bar1 = demo_data[0 : 15].plot.barh(rot=rotation)
    else:
        bar1 = demo_data.plot.barh(rot=rotation)
    
    plt.title(var.capitalize() + ' Demographics', fontsize=20)
    plt.xlabel('Number of users', fontsize=18)
    plt.ylabel('')
    plt.xticks(wrap = False)
    plt.grid(axis = 'x')
    plt.tight_layout()

    #save fig
    plt.savefig(pwd + '\\figures\\demographics_' + var +'.png')


# %%
def var_heatmap(df):
    #create df with only needed vars
    core_data = pd.DataFrame(slider_data, columns= ['physical','emotional','mental','relational','meaning_purpose','critical_kind','fearful_safe','troubled_actions','value_life',                                  'impulse_harm','fear_others','quality_sleep','drugs_alcohol','compassion','burnout','perception','flashbacks',                                      'avoidance','dissociation','bodily_symptoms','intrusive_thoughts','anxious_worried','no_interest',                                          'emotion_value','activation_value'])
    #clustermap for data
    sns.clustermap(core_data.corr('pearson'), method="complete", cmap='RdBu', annot=True,                                                    annot_kws={"size": 12}, vmin=-1, vmax=1, figsize=(20, 15))
    plt.title("Dendrograms")
    plt.tight_layout()
    plt.savefig(pwd + '\\figures\\clustermap.png')


# %%
##boxplot var comparion function
def basic_boxplot(cols, title = 'Box Plot'):
    basic_box_ds = pd.DataFrame(slider_data, columns=cols)
    plt.figure(figsize=(20, 6))
    ax = sns.boxplot(data=basic_box_ds, orient="v", palette="Set2", showfliers=True)
    ax = sns.stripplot(data=basic_box_ds, color=".25")
    plt.title(title,fontsize=25)
    ax.set_ylabel("Score",fontsize=20)
    ax.set_xlabel("Items",fontsize=20)
    ax.tick_params(labelsize=15)
    fig_title = title.replace(" ","")
    plt.tight_layout()
    plt.savefig(pwd + '\\figures\\' + fig_title + '.png')
    #descriptive statistics
    d_stats = basic_box_ds.describe()
    d_stats = d_stats.reindex(['max','75%','mean','25%','min','std'])
    d_stats = d_stats.round(2)
    dfi.export(d_stats, pwd + '\\figures\\' + fig_title + '_descriptives.png')


# %%
def monthly_boxplot(df, column):
    #get dates
    month1b = (date.today()+relativedelta(months=-4)).strftime('%m-%Y') #4 months ago
    month2b = (date.today()+relativedelta(months=-3)).strftime('%m-%Y') #3 months ago
    month3b = (date.today()+relativedelta(months=-2)).strftime('%m-%Y') #last month -2
    month4b = (date.today()+relativedelta(months=-1)).strftime('%m-%Y') #last month

    # Get data by month
    month1_data = df[df['entryMonth'] == month1b]
    month2_data = df[df['entryMonth'] == month2b]
    month3_data = df[df['entryMonth'] == month3b]
    month4_data = df[df['entryMonth'] == month4b]

    # Get data by month for the chosen item and rename columns so they can be used as labels in the plot and in
    # the Kruskal-Wallis tests
    month1_item = pd.DataFrame(month1_data, columns=[column])
    month1_item = month1_item.rename(columns={column: month1b})
    month2_item = pd.DataFrame(month2_data, columns=[column])
    month2_item = month2_item.rename(columns={column: month2b})
    month3_item = pd.DataFrame(month3_data, columns=[column])
    month3_item = month3_item.rename(columns={column: month3b})
    month4_item = pd.DataFrame(month4_data, columns=[column])
    month4_item = month4_item.rename(columns={column: month4b})

    # Concatenate the dataframes to give one that we can plot more easily
    months = pd.concat([month1_item, month2_item, month3_item, month4_item])

    # Plot by month
    item_plot = plt.figure(figsize=(20, 8))
    plt.title(column+" by month", fontsize=25)

    ax_p = sns.boxplot(data=months, orient="v",
                       palette="Set3", showfliers=True)
    ax_p = sns.stripplot(data=months, color=".25")

    # Add some Kruskal-Wallis tests to the plot
    test_results = add_stat_annotation(ax_p, data=months,
                                       box_pairs=[
                                           (month1b, month2b), (month2b, month3b), (month3b, month4b)
                                       ],
                                       test='Kruskal', text_format='full',
                                       loc='outside', verbose=2)
    ax_p.set_xlabel("Months", fontsize=20)
    ax_p.set_ylabel("Score", fontsize=20)
    ax_p.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(pwd + '\\figures\\' + column+"_by_month.png")
    #descriptive statistics
    d_stats = months.describe()
    d_stats = d_stats.reindex(['max','75%','mean','25%','min','std'])
    d_stats = d_stats.round(2)
    dfi.export(d_stats, pwd + '\\figures\\' + column+'_by_month_descriptives.png')


# %%
#boxplots to compare cntw, tewv, ....
def boxplotByGroup(df, column):
    # Make datetime field
    df['diaryEntryDate'] = pd.to_datetime(df['diaryEntryDate'])

    # Get data by month
    all_data = df
    tewv_data = df[df['staff_group'] == "TEWV NHS Foundation Trust"]
    cntw_data = df[df['staff_group'] == "CNTW Foundation NHS Trust"]
    other_data = df[df['staff_group'] == "Other"]
    gp_data = df[df['staff_group'] == "GP Surgery"]
    ncl_data = df[df['staff_group'] == "Newcastle Upon Tyne Hospitals NHS Foundation Trust"]
    southtees_data = df[df['staff_group'] == "South Tees Hospitals NHS Foundation Trust"]

    # Get data by month for the chosen item and rename columns so they can be used as labels in the plot and in
    # the Kruskal-Wallis tests
    all_item = pd.DataFrame(all_data, columns=[column])
    all_item = all_item.rename(columns={column: 'All Staff'})
    tewv_item = pd.DataFrame(tewv_data, columns=[column])
    tewv_item = tewv_item.rename(columns={column: 'TEWV'})
    cntw_item = pd.DataFrame(cntw_data, columns=[column])
    cntw_item = cntw_item.rename(columns={column: 'CNTW'})
    other_item = pd.DataFrame(other_data, columns=[column])
    other_item = other_item.rename(columns={column: 'Other'})
    gp_item = pd.DataFrame(gp_data, columns=[column])
    gp_item = gp_item.rename(columns={column: 'GP Surgery'})
    ncl_item = pd.DataFrame(ncl_data, columns=[column])
    ncl_item = ncl_item.rename(columns={column: 'Newcastle Upon Tyne Trust'})
    southtees_item = pd.DataFrame(southtees_data, columns=[column])
    southtees_item = southtees_item.rename(columns={column: 'South Tees NHS Trust'})

    

    # Concatenate the dataframes to give one that we can plot more easily
    boxplotGroups = pd.concat([all_item, tewv_item, cntw_item, ncl_item, southtees_item, gp_item, other_item])

    # Plot by month
    groups_plot = plt.figure(figsize=(20, 8))
    plt.title(column+" by staff groups", fontsize=25)

    ax_p = sns.boxplot(data=boxplotGroups, orient="v",
                       palette="Set3", showfliers=True)
    ax_p = sns.stripplot(data=boxplotGroups, color=".25")

    # Add some Kruskal-Wallis tests to the plot
    test_results = add_stat_annotation(ax_p, data=boxplotGroups,
                                       box_pairs=[
                                           ("All Staff", "TEWV"), ("All Staff", "CNTW"), ("All Staff", "Newcastle Upon Tyne Trust"), ("All Staff", "South Tees NHS Trust"), ("All Staff", "GP Surgery"), ("All Staff", "Other")
                                       ],
                                       test='Kruskal', text_format='full',
                                       loc='outside', verbose=2)
    ax_p.set_xlabel("Groups", fontsize=20)
    ax_p.set_ylabel("Score", fontsize=20)
    ax_p.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(pwd + '\\figures\\' + column+"_by_staffGroups.png")
    #descriptive statistics
    d_stats = boxplotGroups.describe()
    d_stats = d_stats.reindex(['max','75%','mean','25%','min','std'])
    d_stats = d_stats.round(2)
    dfi.export(d_stats, pwd + '\\figures\\' + column+'_by_staffGroups_descriptives.png')


# %%
#boxplots to compare cntw, tewv, ....
def boxplotByOccupation(df, column):
    # Make datetime field
    df['diaryEntryDate'] = pd.to_datetime(df['diaryEntryDate'])

    # Get data by month
    all_data = df
    nurse_data = df[df['occupation'] == "Nurse"]
    doctor_data = df[df['occupation'] == "Doctor"]
    manager_data = df[df['occupation'] == "Manager"]
    admin_data = df[df['occupation'] == "Administrator"]
    counselling_data = df[df['occupation'] == "Counselling and Therapy"]
    caresupp_data = df[df['occupation'] == "Care Support Worker"]

    # Get data by month for the chosen item and rename columns so they can be used as labels in the plot and in
    # the Kruskal-Wallis tests
    all_item = pd.DataFrame(all_data, columns=[column])
    all_item = all_item.rename(columns={column: 'All Staff'})
    nurse_item = pd.DataFrame(nurse_data, columns=[column])
    nurse_item = nurse_item.rename(columns={column: 'Nurse'})
    doctor_item = pd.DataFrame(doctor_data, columns=[column])
    doctor_item = doctor_item.rename(columns={column: 'Doctor'})
    manager_item = pd.DataFrame(manager_data, columns=[column])
    manager_item = manager_item.rename(columns={column: 'Manager'})
    admin_item = pd.DataFrame(admin_data, columns=[column])
    admin_item = admin_item.rename(columns={column: 'Administrator'})
    counselling_item = pd.DataFrame(counselling_data, columns=[column])
    counselling_item = counselling_item.rename(columns={column: 'Counselling and Therapy'})
    caresupp_item = pd.DataFrame(caresupp_data, columns=[column])
    caresupp_item = caresupp_item.rename(columns={column: 'Care Support Worker'})

    

    # Concatenate the dataframes to give one that we can plot more easily
    boxplotGroups = pd.concat([all_item, nurse_item, doctor_item, manager_item, admin_item, counselling_item, caresupp_item])

    # Plot by month
    groups_plot = plt.figure(figsize=(20, 8))
    plt.title(column+" by occupation groups", fontsize=25)

    ax_p = sns.boxplot(data=boxplotGroups, orient="v",
                       palette="Set3", showfliers=True)
    ax_p = sns.stripplot(data=boxplotGroups, color=".25")

    # Add some Kruskal-Wallis tests to the plot
    test_results = add_stat_annotation(ax_p, data=boxplotGroups,
                                       box_pairs=[
                                           ("All Staff", "Nurse"), ("All Staff", "Doctor"), ("All Staff", "Manager"), ("All Staff", "Administrator"), ("All Staff", "Counselling and Therapy"), ("All Staff", "Care Support Worker")
                                       ],
                                       test='Kruskal', text_format='full',
                                       loc='outside', verbose=2)
    ax_p.set_xlabel("Occupations", fontsize=20)
    ax_p.set_ylabel("Score", fontsize=20)
    ax_p.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(pwd + '\\figures\\' + column+"_by_occupationGroups.png")
    #descriptive statistics
    d_stats = boxplotGroups.describe()
    d_stats = d_stats.reindex(['max','75%','mean','25%','min','std'])
    d_stats = d_stats.round(2)
    dfi.export(d_stats, pwd + '\\figures\\' + column+'_by_occupationGroups_descriptives.png')


# %%
###############################################
#Box plots
###############################################
#columns
boxplot_cols = ['physical','emotional','mental','relational','meaning_purpose','critical_kind','fearful_safe','troubled_actions','value_life',                        'impulse_harm','fear_others','quality_sleep','drugs_alcohol','compassion','burnout','perception','flashbacks',                        'avoidance','dissociation','bodily_symptoms','intrusive_thoughts','anxious_worried','no_interest',                        'emotion_value','activation_value']


# %%
def monthlyAnlysis(boxplot_cols, newDateFormat):
    uniqueMonths = newDateFormat['entryMonth'].nunique()
    #only run if enough months
    if uniqueMonths >= 5:
        # monthly for each variable
        for plot_col in boxplot_cols:
            #plot monthly boxplot
            monthly_boxplot(newDateFormat, plot_col)
    #rankings need 3 months
    if uniqueMonths >= 3:
        #run top and low rankings
        top5_low5(newDateFormat)


# %%
def runAnalysis(boxplot_cols):
    steps_taken(clean_data)
    entriesPerMonth(newDateFormat)
    #user demographics
    demographics_bar('gender', 90)
    demographics_bar('age', 90)
    demographics_bar('occupation')
    demographics_bar('staff_group')
    demographics_bar('ethnicity')
    demographics_bar('region')
    demographics_bar('housing')
    demographics_bar('livingarrangement')
    demographics_bar('hadcovid')
    #heatmap
    var_heatmap(slider_data)
    #comparing variables
    positive1 = ['physical','emotional','mental','relational','meaning_purpose','critical_kind']
    positive2 = ['fearful_safe','troubled_actions','value_life','impulse_harm','fear_others','quality_sleep','drugs_alcohol']
    negative1 = ['compassion','burnout','perception','flashbacks','avoidance','dissociation']
    negative2 = ['bodily_symptoms','intrusive_thoughts','anxious_worried','no_interest','emotion_value','activation_value']
    #basic_boxplot(positive1, "Positive Items 1")
    #basic_boxplot(positive2, "Positive Items 2")
    #basic_boxplot(negative1, "Negative Items 1")
    #basic_boxplot(negative2, "Negative Items 2")
    ## each variable comparing staff groups eg. cntw and tewv
    # for plot_col in boxplot_cols:
    #     # plot monthly boxplot
    #     boxplotByGroup(slider_data, plot_col)
    # # each variable comparing occupations eg. doctor and nurse
    # for plot_col in boxplot_cols:
    #     # plot monthly boxplot
    #     boxplotByOccupation(slider_data, plot_col)

################################################################
# Run functions
################################################################
print('Data analysis satrted ==========>>>>>>>')
monthlyAnlysis(boxplot_cols, newDateFormat)
runAnalysis(boxplot_cols)
print('Analysis COMPLETE ========>>>')
