# open csv file
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Get the folder path where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to export.csv
csv_path = os.path.join(script_dir, "export.csv")

# Vacation dates and durations in days
csv_path_vac = os.path.join(script_dir, "Feiertage_Semesterstart.xlsx")
df_vacation = pd.read_excel(csv_path_vac)

df_vacation['Startdatum'] = pd.to_datetime(df_vacation['Startdatum']).dt.strftime('%Y-%m-%d')

vacation_dates = dict(zip(df_vacation['Startdatum'], df_vacation['Länge in Tage']))


# read csv file
df = pd.read_csv(csv_path, sep=';', encoding='utf-8')

# generate table for current school semester starting 20.01.2025
start_date = pd.to_datetime(df_vacation['Semesterstart']).dt.strftime('%Y-%m-%d')[0]
# end date is two weeks before today
end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)

# convert 'Abw. von' column to datetime format Mi, 04.06.2025 
absence_date = np.hstack(df['Abw. von'])
df['Abw. von']= np.hstack([datetime.strptime(date_str.split(', ')[1], '%d.%m.%Y') for date_str in absence_date])

# filter the dataframe for the current semester
df_current_semester = df[(df['Abw. von'] >= start_date) & (df['Abw. von']<= end_date)].reset_index(drop=True)

# filter duplicate absences
df_current_semester = df_current_semester.drop_duplicates(subset=['Name', 'Abw. von'])

# Sort by name and date
df_current_semester_sorted = df_current_semester.sort_values(['Name', 'Abw. von'])


# count the number of absences for each student. 
# next day absence is replaced with NaN
names = df_current_semester_sorted['Name'].unique()
df_current_semester_sorted= df_current_semester_sorted[df_current_semester_sorted["Typ"]=="zeugnisrelevant"]
valid_absences = []
for name in names:
    df_name = df_current_semester_sorted[df_current_semester_sorted['Name']== name].reset_index()
    day_diff = df_name['Abw. von'].diff().dt.days.fillna(99)
    df_name['Folgetag'] = df_name['Abw. von'].where(day_diff > 1, np.nan)

    # check for weekends in between absences
    fr = df_name['Abw. von'].dt.dayofweek.isin([4]).shift(1)  # Friday
    # remove if day_diff is 2 and weekend is True
    weekend = (day_diff == 3.0) & (fr)
    df_name['Wochenende'] = df_name['Abw. von'].where(~weekend, np.nan)

    # remove if vacation day is in between absences
    is_nan = []
    for date, duration in vacation_dates.items():
        vacation_start = pd.to_datetime(date)
        vacation_end = vacation_start + pd.Timedelta(days=duration - 1)
        before_or_after =(df_name['Abw. von'] >= (vacation_start - pd.Timedelta(days = 1))) & (df_name['Abw. von'] <= (vacation_end + pd.Timedelta(days = 1)))
        if np.sum(before_or_after) >= 2:
            # remove the final absence
            after_vac = (df_name['Abw. von'] == (vacation_end + pd.Timedelta(days = 1)))
            is_nan.append(after_vac)
        else:
            is_nan.append(np.zeros(len(df_name), dtype=bool))

    is_nan = np.vstack(is_nan)
    is_true = np.sum(is_nan, axis=0) >= 1
    df_name['Ferien'] = df_name['Abw. von'].where(~is_true, np.nan)

    # keep only absences that are not nan
    df_name = df_name.dropna(subset=['Folgetag', 'Wochenende', 'Ferien'])
    valid_absences.append(df_name)
df_valid_absences = pd.concat(valid_absences, ignore_index=True)
df_absences = df_valid_absences.groupby('Name')['Abw. von'].count().reset_index()

df_absences.columns = ['Name', 'Anzahl Absenzen dieses Semester']

# return how many unexcused absences each student has
df_unexcused_absences = df_valid_absences[df_valid_absences['Entsch.']=="Nein"].groupby('Name')['Abw. von'].count().reset_index()
df_unexcused_absences.columns = ['Name', 'Unentschuldigte Absenzen dieses Semester']

# export both tables to one csv file
csv_path_out_a = os.path.join(script_dir, r".\Absenzen_Export\absenzen.csv")
csv_path_out_ua = os.path.join(script_dir, r".\Absenzen_Export\unentschuldigte_absenzen.csv")
df_absences.to_csv(csv_path_out_a, sep=';', encoding='utf-8', index=False)
df_unexcused_absences.to_csv(csv_path_out_ua, sep=';', encoding='utf-8', index=False)

