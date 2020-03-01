#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:14:36 2019

@author: dharmit
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("heart.csv")

#Data Visualization
#graph for how each of the attributes has an effect on our end goal,
#that is the label which tells us whether the patient has heart disease or no (target).
print('''Target = 0 means that the patient doesn't have a heart disease,
Target = 1 means patient has a heart disease''')

# set the background colour of the plot to white
sns.set(style="whitegrid", color_codes=True)

# setting the plot size for all plots
sns.set(rc={'figure.figsize':(11.7,8.27)})

# countplot for age w.r.t. target
sns.countplot('age', data=data,hue = 'target')
plt.xlabel('Age of patients', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# plotting the violinplot for age v/s target
sns.violinplot(x="target",y="age", hue="target", data=data);
plt.xlabel('Whether patients have a heart disease or no(0=No , 1=Yes)', fontsize=14)
plt.ylabel('Age of patients', fontsize=14)
plt.show()


# countplot for sex w.r.t. target
sns.countplot('sex', data=data,hue = 'target')
plt.xlabel('Sex of patients (Male=1, Female=0)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# plotting the violinplot for sex w.r.t. target
sns.violinplot(x="target",y="sex", hue="target", data=data);
plt.xlabel('Whether patients have a heart disease or no(0=No , 1=Yes)', fontsize=14)
plt.ylabel('Sex of patients (Male=1, Female=0)', fontsize=14)
plt.show()


# countplot for chest pain type w.r.t. target
sns.countplot('cp', data=data,hue = 'target')
plt.xlabel('Chest pain type of patients (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# plotting the violinplot for chest pain type w.r.t. target
sns.violinplot(x="target",y="cp", hue="target", data=data);
plt.xlabel('Whether patients have a heart disease or no(0=No , 1=Yes)', fontsize=14)
plt.ylabel('Chest pain type of patients (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)', fontsize=14)
plt.show()


# countplot for resting blood pressure w.r.t. target
sns.countplot('trestbps', data=data,hue = 'target')
plt.xlabel('resting blood pressure (in mm Hg on admission to the hospital)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# plotting the violinplot for blood pressure w.r.t. target
sns.violinplot(x="target",y="trestbps", hue="target", data=data);
plt.xlabel('Whether patients have a heart disease or no(0=No , 1=Yes)', fontsize=14)
plt.ylabel('resting blood pressure (in mm Hg on admission to the hospital', fontsize=14)
plt.show()


# countplot for fasting-blood sugar v/s target
sns.countplot('fbs', data=data,hue = 'target')
plt.xlabel('fasting blood sugar > 120 mg/dl (1 = true; 0 = false)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# countplot for resting electrocardiographic results v/s target
sns.countplot('restecg', data=data,hue = 'target')
plt.xlabel('resting electrocardiographic results -- 0: normal, 1: having ST-T wave abnormality ,2:left ventricular hypertrophy', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# countplot for maximum heart rate achieved v/s target
sns.countplot('thalach', data=data,hue = 'target')
plt.xlabel('maximum heart rate achieved ', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# countplot for exercise induced angina v/s target
sns.countplot('exang', data=data,hue = 'target')
plt.xlabel('exercise induced angina (1 = yes; 0 = no) ', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# countplot for ST depression induced by exercise relative to rest v/s target
sns.countplot('oldpeak', data=data,hue = 'target')
plt.xlabel('ST depression induced by exercise relative to rest ', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# countplot for the slope of the peak exercise ST segment v/s target
sns.countplot('slope', data=data,hue = 'target')
plt.xlabel('the slope of the peak exercise ST segment(1=upsloping, 2=flat, 3=downsloping) ', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# countplot for thal v/s target
sns.countplot('thal', data=data,hue = 'target')
plt.xlabel('thal', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()
