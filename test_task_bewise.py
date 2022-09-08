#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from yargy import (
    Parser,
    rule, or_, and_
)
from yargy.predicates import (
    caseless, gram, in_caseless
)
from yargy.pipelines import (
    morph_pipeline
)
from yargy.interpretation import (
    fact,
    attribute
)
from yargy.relations import gnc_relation

file_path = input('Enter .csv file path to extract from\n')
df = pd.read_csv(file_path, sep=',')
df['insight'] = [{} for _ in range(len(df))]

gnc = gnc_relation()

#setting yargy rules for different entities
GREETING_MORPH = morph_pipeline([
    'привет',
    'здравствовать',
    'приветствовать'
])
GOOD = morph_pipeline(['добрый']).match(gnc)
DAY_TIME = morph_pipeline([
    'утро',
    'день',
    'вечер',
    'время суток'
]).match(gnc)
GREETING = or_(
    GREETING_MORPH,
    rule(GOOD, DAY_TIME),
    rule(DAY_TIME, GOOD)
)

FAREWELL_MORPH = morph_pipeline([
    'до свидания',
    'до встречи',
    'до завтра',
    'до завтрашнего дня'
]).match(gnc)
ALL = caseless('всего')
THE_BEST = morph_pipeline([
    'хороший',
    'добрый'
])
FAREWELL_CASELESS = rule(ALL, THE_BEST)
FAREWELL = or_(FAREWELL_MORPH, FAREWELL_CASELESS)

Company = fact('Company', ['company_name'])
COMPANY_PREFIX = morph_pipeline([
    'компания',
    'организация',
    'предприятие'
])
COMPANY_NAME = or_(
    rule(
        gram('NOUN'),
        gram('ADJF').optional()
    ).repeatable(max=3),
    rule(
        gram('ADJF').optional(),
        gram('NOUN')
    ).repeatable(max=3)
)
COMPANY = or_(
    rule(COMPANY_PREFIX, COMPANY_NAME.interpretation(
        Company.company_name
)),
    rule(COMPANY_NAME.interpretation(
        Company.company_name
), COMPANY_PREFIX)
).interpretation(Company)

FIRST_NAME = gram('Name')
IT_IS = in_caseless({'это', 'я'})
ME = morph_pipeline([
    'я',
])
NAMED = morph_pipeline(['звать'])
MY = morph_pipeline([
    'мой'
]).match(gnc)
NAME = morph_pipeline([
    'имя'
]).match(gnc)
INTRODUCE = or_(
    rule(IT_IS, FIRST_NAME),
    rule(FIRST_NAME, IT_IS),
    rule(ME, NAMED, FIRST_NAME),
    rule(ME, FIRST_NAME, NAMED),
    rule(NAMED, ME, FIRST_NAME),
    rule(MY, NAME, FIRST_NAME),
    rule(FIRST_NAME, MY, NAME)
)

#define extraction funcs
def get_greetings(dataset):
    parser = Parser(GREETING)
    for index, row in dataset.iterrows():
        for match in parser.findall(row['text']):
            dataset.at[index, 'insight']['greeting'] = True

def get_farewell(dataset):
    parser = Parser(FAREWELL)
    for index, row in dataset.iterrows():
        for match in parser.findall(row['text']):
            dataset.at[index, 'insight']['farewell'] = True

def get_introduce(dataset):
    parser = Parser(INTRODUCE)
    for index, row in dataset.iterrows():
        for match in parser.findall(row['text']):
            for token in match.tokens:
                if 'Name'in token.forms[0].grams:
                    start, finish = match.tokens[0].span
                    if (finish - start > 1):    #checking names' length, drop names consisting 1 letter
                        dataset.at[index, 'insight']['introduce'] = True
                        dataset.at[index, 'insight']['manager_name'] = token.value.capitalize()

def get_company(dataset):
    parser = Parser(COMPANY)
    for index, row in dataset.iterrows():
        for match in parser.findall(row['text']):
            dataset.at[index, 'insight']['company_name'] = ' '.join(list(map(lambda x: x.capitalize(), match.fact.company_name.split(' '))))

def check_manager_requirement(dataset):
    for name, group in dataset.groupby('dlg_id'):
        greeting_present = False
        farewell_present = False
        for row in group['insight']:
            if 'greeting' in row:
                greeting_present = True
            if 'farewell' in row:
                farewell_present = True
        if greeting_present and farewell_present:
            dataset.at[dataset.groupby('dlg_id').groups[name][0], 'insight']['manager_is_ok'] = True
        else:
            dataset.at[dataset.groupby('dlg_id').groups[name][0], 'insight']['manager_is_ok'] = False

print('Starting extraction... ')
print('Results will be saved in out.csv in the same folder')
get_greetings(df.loc[df['role'] == 'manager'])
get_farewell(df.loc[df['role'] == 'manager'])
get_introduce(df.loc[df['role'] == 'manager'])
get_company(df.loc[df['role'] == 'manager'])
check_manager_requirement(df)

df.to_csv('out.csv')
