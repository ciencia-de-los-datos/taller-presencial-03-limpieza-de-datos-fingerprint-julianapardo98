"""GitHub Classroom autograding script."""

import os

import pandas as pd
import clean_data

clean_data.main("input.txt", "output.txt")

if not os.path.exists("test.csv"):
    raise FileNotFoundError("File 'test.csv' not found")

test = pd.read_csv("test.csv", index_col=None)

assert test.loc[0, "key"] == "adhoc queri"
assert test.loc[6, "key"] == "agricultur product"
assert test.loc[11, "key"] == "airlin"
assert test.loc[12, "key"] == "airlin compani"
assert test.loc[16, "key"] == "analyt applic"
assert test.loc[25, "key"] == "analyt model"

#
# Retorna error si la carpeta output/ no existe
if not os.path.exists("output.txt"):
    raise FileNotFoundError("File 'output.txt' not found")

#
# Lee el contenido del archivo output.txt
dataframe = pd.read_csv("output.txt")
count = dataframe.groupby("text").size()


assert count.loc["AD-HOC QUERIES"] == 6
assert count.loc["AGRICULTURAL PRODUCTION"] == 5
assert count.loc["AIRLINE COMPANIES"] == 4
assert count.loc["AIRLINES"] == 1
assert count.loc["ANALYTIC APPLICATIONS"] == 9
assert count.loc["ANALYTIC MODEL"] == 10
