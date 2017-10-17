# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13: 30: 32 2017

@author: Mahkah
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd



### Load Data
# Data retrieved from https: //dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910/DVN/GDF6Z0

cces_df = pd.read_stata('data/CCES16_Common_OUTPUT_Jul2017_VV.dta')
cat_columns = cces_df.select_dtypes(['category']).columns
cces_df[cat_columns] = cces_df[cat_columns].apply(lambda x: x.cat.codes)


### Recode
# Demographic and racial attitude coding adapted from http://svmiller.com/blog/2017/04/age-income-racism-partisanship-trump-vote-2016/#code

cces_recode = pd.DataFrame(index=cces_df.index)

def map_recode(var, recode_map, tabulate=True):
    cces_recode[var] = cces_df[var].map(recode_map)
    if tabulate == True:
        print(cces_recode[var].value_counts(sort=False))


## Gender
cces_recode['female'] = cces_df['gender']
print(cces_recode['female'].value_counts(sort=False))


## Race
cces_recode['white'] = np.where(cces_df['race'] == 0, 1, 0)
print(cces_recode['white'].value_counts(sort=False))

cces_recode['black'] = np.where(cces_df['race'] == 1, 1, 0)
print(cces_recode['black'].value_counts(sort=False))

cces_recode['hisp'] = np.where(cces_df['race'] == 2, 1, 0)
print(cces_recode['hisp'].value_counts(sort=False))

cces_recode['asian'] = np.where(cces_df['race'] == 3, 1, 0)
print(cces_recode['asian'].value_counts(sort=False))

cces_recode['race_other'] = np.where(np.isin(cces_df['race'], [4, 6, 7]), 1, 0)
print(cces_recode['race_other'].value_counts(sort=False))

cces_recode['mixed_race'] = np.where(cces_df['race'] == 5, 1, 0)
print(cces_recode['mixed_race'].value_counts(sort=False))

cces_recode['race_big_other'] = np.where(cces_df['race'] >= 4, 1, 0)
print(cces_recode['race_big_other'].value_counts(sort=False))


## Age
cces_recode['age'] = cces_df['birthyr'].apply(lambda x: 2016 - x)

cces_recode['age2'] = cces_recode['age'].apply(lambda x: x ** 2)

## Education
cces_recode['college'] = np.where(cces_df['educ'] > 3, 1, 0)
print(cces_recode['college'].value_counts(sort=False))


## Religiosity
# See: http: //www.pewresearch.org/fact-tank/2016/02/29/how-religious-is-your-state/
relig = pd.DataFrame(index=cces_df.index)

recode = {0: 1, 1: 0, 2: -1, 3: -1, 4: 0}
relig['importance'] = cces_df['pew_religimp'].map(recode)

recode = {0: 1, 1: 1, 2: 0, 3: 0, 4: -1, 5: -1, 6: 0, 7: 0}
relig['attend'] = cces_df['pew_churatd'].map(recode)

recode = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: -1, 6: -1, 7: 0, 8: 0}
relig['prayer'] = cces_df['pew_prayer'].map(recode)

relig.dropna(axis=0, how='any', inplace=True)
relig['religiosity'] = (relig['importance'] + relig['attend'] + relig['prayer']) / 3

cces_recode['religiosity'] = relig['religiosity']


## PID
recode = {0: 1, 1: 0.67, 2: 0.33, 3: 0, 4: -0.33, 5: -0.67, 6: -1}
map_recode('pid7', recode)


## Ideology
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1}
map_recode('ideo5', recode)


## Intercept
cces_recode = sm.add_constant(cces_recode, prepend=False)
cces_recode = cces_recode.rename(columns = {'const': 'intercept'})


### Attitudes
# See CCES Guide 2016.pdf, pg 56
attitude_meta = {}

## State of the country
recode_var = 'CC16_302'
attitude_meta[recode_var] = [
        "Over the past year the nation's economy has ...?", 
        "Over the past year the nation's economy has ...?", 
        '"Stayed about the same"', '"Gotten much better"',
        'State of the country',
        1]
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1, 5: 0, 6: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_303'
attitude_meta[recode_var] = [
        "Over the past four years, has your household's annual income ...?", 
        "Over the past four years, has your household's annual income ...?", 
        '"Stayed about the same"', '"Increased a lot"',
        'State of the country',
        1]
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1, 5: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_304'
attitude_meta[recode_var] = [
        "Over the next year, do you think the nation's economy will ...?", 
        "Over the next year, do you think the nation's economy will ...?", 
        '"Stay about the same"', '"Get much better"',
        'State of the country',
        1]
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1, 5: 0, 6: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_307'
attitude_meta[recode_var] = [
        "Do the police make you feel?", 
        "Do the police make you feel?", 
        'Neutral responses', '"Mostly safe"',
        'State of the country',
        1]
recode = {0: 1, 1: 0.5, 2: -0.5, 3: -1, 4: 0}
map_recode(recode_var, recode)


## Gun Regulation
recode_var = 'CC16_330a'
attitude_meta[recode_var] = [
        "Require background checks for all gun sales",
        "Do you support or oppose the following proposal? Background checks for all sales, including at gun shows and over the Internet?", 
        'Neutral responses', '"Support"',
        'Gun Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_330b'
attitude_meta[recode_var] = [
        "Prohibit local governments from doxxing gun owners",
        "Do you support or oppose the following proposal? Prohibit state and local governments from publishing the names and addresses of all gun owners", 
        'Neutral responses', '"Support"',
        'Gun Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_330d'
attitude_meta[recode_var] = [
        "Ban assault rifles",
        "Do you support or oppose the following proposal? Ban assault rifles", 
        'Neutral responses', '"Support"',
        'Gun Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_330e'
attitude_meta[recode_var] = [
        "Loosen concealed-carry requirements",
        "Do you support or oppose the following proposal? Make it easier for people to obtain concealed-carry permit", 
        'Neutral responses', '"Support"',
        'Gun Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


cces_recode['gun_cntrl_comb'] = (cces_recode['CC16_330a'] - cces_recode['CC16_330b'] + cces_recode['CC16_330d'] - cces_recode['CC16_330e']) / 4
print(cces_recode['gun_cntrl_comb'].value_counts(sort=False))
attitude_meta['gun_cntrl_comb'] = [
        "Gun regulation scale",
        "Gun regulation scale (from 4 elements)", 
        'Half pro-regulation/half anti-regulation', 'Completely pro-gun regulation',
        'Gun Regulation',
        1]


## Immigration
recode_var = 'CC16_331_1'
attitude_meta[recode_var] = [
        "Grant legal status to all employed, non-criminal illegal immigrants",
        "Do you think the U.S. government should implement the following immigration policy? Grant legal status to all illegal immigrants who have held jobs and paid taxes for at least 3 years, and not been convicted of any felony crimes", 
        'Neutral responses', '"Support"',
        'Immigration',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_331_2'
attitude_meta[recode_var] = [
        "Increase border patrols on the U.S.-Mexican border",
        "Do you think the U.S. government should implement the following immigration policy? Increase the number of border patrols on the U.S.-Mexican border", 
        'Neutral responses', '"Support"',
        'Immigration',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_331_3'
attitude_meta[recode_var] = [
        "Grant legal status to 'Dreamers'",
        "Do you think the U.S. government should implement the following immigration policy? Grant legal status to people who were brought to the US illegally as children, but who have graduated from a U.S. high school", 
        'Neutral responses', '"Support"',
        'Immigration',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_331_7'
attitude_meta[recode_var] = [
        'Identify and deport illegal immigrants', 
        "Do you think the U.S. government should implement the following immigration policy? Identify and deport illegal immigrants", 
        'Neutral responses', '"Support"',
        'Immigration',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


cces_recode['immigration_comb'] = (cces_recode['CC16_331_1'] - cces_recode['CC16_331_2'] + cces_recode['CC16_331_3'] - cces_recode['CC16_331_7']) / 4
print(cces_recode['immigration_comb'].value_counts(sort=False))
attitude_meta['immigration_comb'] = [
        "Immigration policy scale", 
        "Immigration policy scale (from 4 elements)", 
        'Half liberalize immigration/half tighten immigration', 'Completely liberalize immigration policy',
        'Immigration',
        1]


## Choice
recode_var = 'CC16_332a'
attitude_meta[recode_var] = [
        "Always allow a woman to obtain an abortion",
        "Do you support or oppoese the following proposals? Always allow a woman to obtain an abortion as a matter of choice", 
        'Neutral responses', '"Support"',
        'Choice/Abortion',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_332b'
attitude_meta[recode_var] = [
        "Permit abortion only in case of rape, incest or when the woman's life is in danger",
        "Do you support or oppoese the following proposals? Permit abortion only in case of rape, incest or when the woman's life is in danger", 
        'Neutral responses', '"Support"',
        'Choice/Abortion',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_332c'
attitude_meta[recode_var] = [
        "Prohibit all abortions after the 20th week of pregnancy",
        "Do you support or oppoese the following proposals? Prohibit all abortions after the 20th week of pregnancy", 
        'Neutral responses', '"Support"',
        'Choice/Abortion',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_332d'
attitude_meta[recode_var] = [
        "Allow employers to decline coverage of abortions in insurance plans",
        "Do you support or oppoese the following proposals? Allow employers to decline coverage of abortions in insurance plans", 
        'Neutral responses', '"Support"',
        'Choice/Abortion',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_332e'
attitude_meta[recode_var] = [
        "Prohibit the usage of federal funds for any abortion",
        "Do you support or oppoese the following proposals? Prohibit the expenditure of funds authorized or appropriated by federal law for any abortion", 
        'Neutral responses', '"Support"',
        'Choice/Abortion',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_332f'
attitude_meta[recode_var] = [
        "Make abortions illegal in all circumstances",
        "Do you support or oppoese the following proposals? Make abortions illegal in all circumstances", 
        'Neutral responses', '"Support"',
        'Choice/Abortion',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


cces_recode['abortion_comb'] = (cces_recode['CC16_332a'] - cces_recode['CC16_332b'] - cces_recode['CC16_332c'] - cces_recode['CC16_332d'] - cces_recode['CC16_332e'] - cces_recode['CC16_332f']) / 6
print(cces_recode['abortion_comb'].value_counts(sort=False))
attitude_meta['abortion_comb'] = [
        "Abortion policy scale",
        "Abortion policy scale (from 6 elements)", 
        'Half liberalize abortion/half tighten abortion', 'Completely liberalize abortion policy',
        'Choice/Abortion',
        1]


## Environmental Regulation
recode_var = 'CC16_333a'
attitude_meta[recode_var] = [
        "Allow the EPA to regulate CO2 emissions",
        "Do you support or oppoese the following proposals? Give Environmental Protection Agency power to regulate Carbon Dioxide emissions", 
        'Neutral responses', '"Support"',
        'Environmental Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_333b'
attitude_meta[recode_var] = [
        "Raise required fuel efficiency for standards from 25 mpg to 35 mpg",
        "Do you support or oppoese the following proposals? Raise required fuel efficiency for the average automobile from 25 mpg to 35 mpg", 
        'Neutral responses', '"Support"',
        'Environmental Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_333c'
attitude_meta[recode_var] = [
        "Require a minimum amount of renewable fuels for electricity generation even if prices increase",
        "Do you support or oppoese the following proposals? Require a minimum amount of renewable fuels (wind, solar, and hydroelectric) in the generation of electricity even if electricity prices increase somewhat", 
        'Neutral responses', '"Support"',
        'Environmental Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_333d'
attitude_meta[recode_var] = [
        "Strengthen enforcement of Clean Air and Clean Water Acts even if it costs jobs",
        "Do you support or oppoese the following proposals? Strengthen enforcement of the Clean Air Act and Clean Water Act even if it costs U.S. jobs", 
        'Neutral responses', '"Support"',
        'Environmental Regulation',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


cces_recode['enviro_reg_comb'] = (cces_recode['CC16_333a'] - cces_recode['CC16_333b'] + cces_recode['CC16_333c'] - cces_recode['CC16_333d']) / 4
print(cces_recode['enviro_reg_comb'].value_counts(sort=False))
attitude_meta['enviro_reg_comb'] = [
        "Environmental regulation scale", 
        "Environmental regulation scale (from 4 elements)", 
        'Support half/oppose half of regulatory proposals', 'Support all regulatory proposals',
        'Environmental Regulation',
        1]


## Criminal Justice
recode_var = 'CC16_334a'
attitude_meta[recode_var] = [
        "Eliminate mandatory minimum sentences for non-violent drug offenders", 
        "Do you support or oppoese the following proposals? Eliminate mandatory minimum sentences for non-violent drug offenders", 
        'Neutral responses', '"Support"',
        'Criminal Justice',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_334b'
attitude_meta[recode_var] = [
        "Require police officers to wear body cameras while on duty", 
        "Do you support or oppoese the following proposals? Require police officers to wear body cameras that record all of their activities while on duty", 
        'Neutral responses', '"Support"',
        'Criminal Justice',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_334c'
attitude_meta[recode_var] = [
        "Increase police on the street by 10%, even if it trades off with other public services", 
        "Do you support or oppoese the following proposals? Increase the number of police on the street by 10 percent, even if it means fewer funds for other public services", 
        'Neutral responses', '"Support"',
        'Criminal Justice',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_334d'
attitude_meta[recode_var] = [
        "Increase prison sentences for two strike felons", 
        "Do you support or oppoese the following proposals? Increase prison sentences for felons who have already committed two or more serious or violent crimes", 
        'Neutral responses', '"Support"',
        'Criminal Justice',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


cces_recode['cjr_comb'] = (cces_recode['CC16_334a'] + cces_recode['CC16_333b'] - cces_recode['CC16_333c'] - cces_recode['CC16_333d']) / 4
print(cces_recode['cjr_comb'].value_counts(sort=False))
attitude_meta['cjr_comb'] = [
        "Criminal justice scale", 
        "Criminal justice scale (from 4 elements)", 
        'Half liberalize criminal justice/half tighten criminal justice', 'Completely liberalize criminal justice',
        'Criminal Justice',
        1]


## Gay Marriage
recode_var = 'CC16_335'
attitude_meta[recode_var] = [
        "Do you favor or oppose allowing gays and lesbians to marry legally?", 
        "Do you favor or oppose allowing gays and lesbians to marry legally?", 
        'Neutral responses', '"Favor"',
        'Gay Marriage',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


## Government Spending
recode_var = 'CC16_337_1'
attitude_meta[recode_var] = [
        "Cut defense spending", 
        "Cut defense spending: Rank preference for cutting defense spending, cutting domestic spending (such as Medicare and Social Security), or raising taxes to cover the deficit.", 
        '"Least prefer"', '"Most prefer"',
        'Government Spending',
        1]
recode = {0: 1, 1: 0.5, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_337_2'
attitude_meta[recode_var] = [
        "Cut domestic spending",
        "Cut domestic spending: Rank preference for cutting defense spending, cutting domestic spending (such as Medicare and Social Security), or raising taxes to cover the deficit.", 
        '"Least prefer"', '"Most prefer"',
        'Government Spending',
        1]
recode = {0: 1, 1: 0.5, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_337_3'
attitude_meta[recode_var] = [
        "Raise taxes",
        "Raise taxes: Rank preference for cutting defense spending, cutting domestic spending (such as Medicare and Social Security), or raising taxes to cover the deficit.", 
        '"Least prefer"', '"Most prefer"',
        'Government Spending',
        1]
recode = {0: 1, 1: 0.5, 2: 0}
map_recode(recode_var, recode)


## Congressional Issues
recode_var = 'CC16_351B'
attitude_meta[recode_var] = [
        "Ratify Trans-Pacific Partnership", 
        "Would you vote for or against the following? Trans-Pacific Partnership Act. Free trade agreement among 12 Pacific nations (Australia, Brunei, Canada, Chile, Japan, Malaysia, Mexico, New Zealand, Peru, Singapore, and the US).", 
        'Neutral responses', '"Support"',
        'Specific Legislation',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_351E'
attitude_meta[recode_var] = [
        "Repeal the No Child Left Behind Act",
        "Would you vote for or against the following? Education Reform. Repeals the No Child Left Behind Act, which required testing of all students and penalized schools that fell below federal standards. Allows states to identify and improve poor performing schools.",
        'Neutral responses', '"Support"',
        'Specific Legislation',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_351F'
attitude_meta[recode_var] = [
        "Authorizes $305 billion for transportation infrastructure", 
        "Would you vote for or against the following? Highway and Transportation Funding Act. Authorizes $305 Billion to repair and expand highways, bridges, and transit over the next 5 years.",
        'Neutral responses', '"Support"',
        'Specific Legislation',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_351G'
attitude_meta[recode_var] = [
        "Imposes new sanctions on Iran if Iran doesn't agree to reduce its nuclear program", 
        "Would you vote for or against the following? Iran Sanctions Act. Imposes new sanctions on Iran, if Iran does not agree to reduce its nuclear program by June 30.",
        'Neutral responses', '"Support"',
        'Specific Legislation',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_351H'
attitude_meta[recode_var] = [
        "Reform Medicare through the Accountability and Cost Reform Act",
        "Would you vote for or against the following? Accountability and Cost Reform Act. Shifts Medicare from fee-for-serviceto pay-for-performance. Ties Medicare payments to doctors to quality of care measures. Requires higher premiums for seniors who make more than $134,000. Renews the Children Health Insurance Program (CHIP).",
        'Neutral responses', '"Support"',
        'Specific Legislation',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_351I'
attitude_meta[recode_var] = [
        "Repeal Affordable Care Act",
        "Would you vote for or against the following? Repeal Affordable Care Act. Would repeal the Affordable Care Act of 2009 (also known as Obamacare).",
        'Neutral responses', '"Support"',
        'Specific Legislation',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_351K'
attitude_meta[recode_var] = [
        "Raise the federal minimum wage to $12 an hour by 2020",
        "Would you vote for or against the following? Minimum wage. Raises the federal minimum wage to $12 an hour by 2020.",
        'Neutral responses', '"Support"',
        'Specific Legislation',
        1]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


## U.S. Military Deployment
recode_var = 'CC16_414_1'
attitude_meta[recode_var] = [
        "Would deploy troops to ensure the supply of oil",
        "Would you approve the use of U.S. military troops in order to ensure the supply of oil?",
        'Neutral responses', '"Approve"',
        'Military Force Usage',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_414_2'
attitude_meta[recode_var] = [
        "Would deploy troops to destroy a terrorist camp",
        "Would you approve the use of U.S. military troops in order to destroy a terrorist camp",
        'Neutral responses', '"Approve"',
        'Military Force Usage',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_414_3'
attitude_meta[recode_var] = [
        "Would deploy troops to intervene in a region where there is genocide or a civil war",
        "Would you approve the use of U.S. military troops in order to intervene in a region where there is genocide or a civil war?",
        'Neutral responses', '"Approve"',
        'Military Force Usage',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_414_4'
attitude_meta[recode_var] = [
        "Would deploy troops to assist the spread of democracy",
        "Would you approve the use of U.S. military troops in order to assist the spread of democracy",
        'Neutral responses', '"Approve"',
        'Military Force Usage',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_414_5'
attitude_meta[recode_var] = [
        "Would deploy troops to protect American allies under attack by foreign nations",
        "Would you approve the use of U.S. military troops in order to protect American allies under attack by foreign nations",
        'Neutral responses', '"Approve"',
        'Military Force Usage',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


recode_var = 'CC16_414_6'
attitude_meta[recode_var] = [
        "Would deploy troops to help the UN uphold international law",
        "Would you approve the use of U.S. military troops in order to help the United Nations uphold international law",
        'Neutral responses', '"Approve"',
        'Military Force Usage',
        0]
recode = {0: 1, 1: -1, 2: 0}
map_recode(recode_var, recode)


cces_recode['aumf_comb'] = (cces_recode['CC16_414_1'] + cces_recode['CC16_414_2'] + cces_recode['CC16_414_3'] + cces_recode['CC16_414_4'] + cces_recode['CC16_414_5'] + cces_recode['CC16_414_6']) / 6
print(cces_recode['aumf_comb'].value_counts(sort=False))
attitude_meta['aumf_comb'] = [
        "Military force usage scale", 
        "Military force usage scale (from 6 elements)", 
        'Half approve/half disapprove of military force usage', 'Approve of all military force usage',
        'Military Force Usage',
        1]


## Racial Attitudes
recode_var = 'CC16_422c'
attitude_meta[recode_var] = [
        "I am angry that racism exists.",
        "Do you agree or disagree with the following statement? I am angry that racism exists.",
        '"Neither agree nor disagree"', '"Strongly agree"',
        'Racial Attitudes',
        0]
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1}
map_recode(recode_var, recode)


recode_var = 'CC16_422d'
attitude_meta[recode_var] = [
        "White people in the U.S. have certain advantages because of the color of their skin.",
        "Do you agree or disagree with the following statement? White people in the U.S. have certain advantages because of the color of their skin.",
        '"Neither agree nor disagree"', '"Strongly agree"',
        'Racial Attitudes',
        0]
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1}
map_recode(recode_var, recode)


recode_var = 'CC16_422e'
attitude_meta[recode_var] = [
        "I often find myself fearful of people of other races.",
        "Do you agree or disagree with the following statement? I often find myself fearful of people of other races.",
        '"Neither agree nor disagree"', '"Strongly agree"',
        'Racial Attitudes',
        0]
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1}
map_recode(recode_var, recode)


recode_var = 'CC16_422f'
attitude_meta[recode_var] = [
        "Racial problems in the U.S. are rare, isolated situations.",
        "Do you agree or disagree with the following statement? Racial problems in the U.S. are rare, isolated situations.",
        '"Neither agree nor disagree"', '"Strongly agree"',
        'Racial Attitudes',
        0]
recode = {0: 1, 1: 0.5, 2: 0, 3: -0.5, 4: -1}
map_recode(recode_var, recode)


cces_recode['cognative_racism'] = -(cces_recode['CC16_422d'] - cces_recode['CC16_422f']) / 2
print(cces_recode['cognative_racism'].value_counts(sort=False))
attitude_meta['cognative_racism'] = [
        "Cognative racism scale  (a proxy for awareness of racism)", 
        "Cognative racism scale (from 2 elements)", 
        'neutral cognative racism scale', 'max cognative racism scale',
        'Racial Attitudes',
        1]


cces_recode['empathetic_racism'] = -(cces_recode['CC16_422c'] - cces_recode['CC16_422e']) / 2
print(cces_recode['empathetic_racism'].value_counts(sort=False))
attitude_meta['empathetic_racism'] = [
        "Empathetic racism scale (a proxy for sympathy towards the experiences of racial minorities)", 
        "Empathetic racism scale (from 2 elements)", 
        'neutral cognative racism scale', 'max empathetic racism scale',
        'Racial Attitudes',
        1]


## DV
cces_recode = pd.concat([cces_recode, cces_df['CC16_326'], cces_df['CC16_410a']], axis=1)
cces_recode=cces_recode.rename(columns = {'CC16_326': 'vote12', 'CC16_410a': 'vote16'})

recode = {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 7: 0}
cces_recode['vote for Hillary Clinton'] = cces_recode['vote16'].map(recode)
print(cces_recode['vote for Hillary Clinton'].value_counts(sort=False))

recode = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 7: 0}
cces_recode['vote for Donald Trump'] = cces_recode['vote16'].map(recode)
print(cces_recode['vote for Donald Trump'].value_counts(sort=False))


## Population
cces_recode['voters'] = pd.Series(1, index=cces_recode.index)

cces_recode['2012 Obama voters'] = np.where(cces_recode['vote12'] == 0, 1, 0)
print(cces_recode['2012 Obama voters'].value_counts(sort=False))

cces_recode['2012 Romney voters'] = np.where(cces_recode['vote12'] == 1, 1, 0)
print(cces_recode['2012 Romney voters'].value_counts(sort=False))

cces_recode['Democrats'] = np.where(cces_recode['pid7'] >= 0.5, 1, 0)
print(cces_recode['Democrats'].value_counts(sort=False))

cces_recode['Republicans'] = np.where(cces_recode['pid7'] <= -0.5, 1, 0)
print(cces_recode['Republicans'].value_counts(sort=False))

cces_recode['Independents'] = np.where((cces_recode['pid7'] >= -0.5) & (cces_recode['pid7'] <= 0.5), 1, 0)
print(cces_recode['Independents'].value_counts(sort=False))


## Weight
cces_recode = pd.concat([cces_recode, cces_df['commonweight_vv_post']], axis=1)



### Model
def cces_glm(Y, X, W, summarize):
    
    logit = sm.GLM(Y, X, family=sm.families.Binomial(), freq_weights=W, missing='drop')
    
    result = logit.fit()
    
    if summarize == True:
        print(result.summary())
    
    params = result.params
    conf_int = result.conf_int()
    conf_int['coef'] = params
    result_df = pd.concat([conf_int, np.exp(conf_int)], axis=1)
    result_df.columns = ['coef_2.5%', 'coef_97.5%', 'coef', 'or_2.5%', 'or_97.5%', 'or']
    result_df['factor'] = result_df.index
    return result_df




### Define model parameters
#Use white as a base
controls = ['intercept', 'female', 'black', 'hisp', 'asian', 'race_other', 'mixed_race', 'age', 'age2', 'college', 'religiosity', 'pid7', 'ideo5']
#controls = ['intercept', 'female', 'black', 'hisp', 'race_big_other', 'age', 'college', 'religiosity', 'pid7', 'ideo5']
xvars = attitude_meta.keys()
outcomes = ['vote for Hillary Clinton', 'vote for Donald Trump']
weight = ['commonweight_vv_post']
populations = ['voters', '2012 Obama voters', '2012 Romney voters', 'Democrats', 'Republicans', 'Independents']

### Run models and produce output for Tableau
result = pd.DataFrame()


for population in populations:
    
    cces_model = cces_recode[cces_recode[population] == 1]
    cces_weight = np.array(cces_model[weight]).reshape(cces_model[weight].shape[0])
    
    for xvar in xvars:
        
        for outcome in outcomes:
            
            factors = [xvar] + controls
            indiv_result = cces_glm(cces_model[outcome], cces_model[factors], cces_weight, summarize=True)
            
            indiv_result['ind_var'] = pd.Series(xvar, index=indiv_result.index)
            indiv_result['ind_var_qtext'] = pd.Series(attitude_meta.get(xvar)[0], index=indiv_result.index)
            indiv_result['ind_var_qtext_full'] = pd.Series(attitude_meta.get(xvar)[1], index=indiv_result.index)
            indiv_result['ind_var_orshift0'] = pd.Series(attitude_meta.get(xvar)[2], index=indiv_result.index)
            indiv_result['ind_var_orshift1'] = pd.Series(attitude_meta.get(xvar)[3], index=indiv_result.index)
            indiv_result['ind_var_cat'] = pd.Series(attitude_meta.get(xvar)[4], index=indiv_result.index)
            indiv_result['ind_var_concise'] = pd.Series(attitude_meta.get(xvar)[5], index=indiv_result.index)
            indiv_result['dep_var'] = pd.Series(outcome, index=indiv_result.index)
            indiv_result['population'] = pd.Series(population, index=indiv_result.index)
            
            result = pd.concat([result, indiv_result], ignore_index=True)


result.to_csv('data\logit_results.csv')


