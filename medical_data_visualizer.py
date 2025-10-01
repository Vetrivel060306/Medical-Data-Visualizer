import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight']=0
df['BMI'] = df['weight'] / ((df['height'] / 100)**2)
df.loc[df['BMI'] > 25, 'overweight'] = 1



# 3
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )


    # 6
    
    

    # 7




    # 8
    fig = sns.catplot(
        data=df_cat,
        kind='count',
        x='variable',
        hue='value',
        col='cardio',
    ).set_axis_labels("Variable", "Total")



    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():

#11
    df_heat = df.loc[
                        (df['ap_lo'] <= df['ap_hi'])                   &
                        (df['height'] >= df['height'].quantile(0.025)) &
                        (df['height'] <= df['height'].quantile(0.975)) &
                        (df['weight'] >= df['weight'].quantile(0.025)) &
                        (df['weight'] <= df['weight'].quantile(0.975)),
                        ['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke',
                        'alco', 'active', 'cardio', 'overweight']
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    sns.heatmap(corr, annot=True, fmt=".1f", ax=ax, mask=mask, square=True, linewidths=.5, center=0)
    ax.set_title('Correlation Heatmap')


    # 16
    fig.savefig('heatmap.png')
    return fig
