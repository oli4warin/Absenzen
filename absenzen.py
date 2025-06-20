# open csv file
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pdfplumber



def is_valid_absence(df_tobe_filtered) -> pd.DataFrame: 
    """
    Function to filter valid absences based on the following criteria:
    1. The absence must not be preceeded by a previous day (Folgetag).
    2. There must not be a weekend in between the absences (Wochenende).
    3. There must not be a vacation day in between the absences (Ferien).
    In those cases the absence is discarded.
    Parameters:
    df_tobe_filtered (DataFrame): DataFrame containing the absences to be filtered.
    Returns:
    DataFrame: Filtered DataFrame containing only valid absences.
    """
    valid_absences = []
    for name in names:
        if name == "Azra Nehir Donat":
            x = 1
        df_name = df_tobe_filtered[df_tobe_filtered['Name']== name].reset_index()
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
        #df_name = df_name.dropna(subset=['Folgetag', 'Wochenende', 'Ferien'])
        valid_absences.append(df_name)
    df_valid_absences = pd.concat(valid_absences, ignore_index=True)
    df_valid_absences = df_valid_absences.dropna(subset=['Folgetag', 'Wochenende', 'Ferien'])
    return df_valid_absences


def extract_table_from_pdf(pdf_path) -> pd.DataFrame:
    """
    Extracts a table from the first page of a PDF file and returns it as a DataFrame.
    
    Parameters:
    pdf_path (str): Path to the PDF file.
    
    Returns:
    pd.DataFrame: DataFrame containing the extracted table.
    """
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        table = first_page.extract_table()
        if table:
            return pd.DataFrame(table[:], columns=['Datum', 'Wochentag', 'Start Uhrzeit', 'Ende Uhrzeit', 'Info', 'Lehrperson', 'Raum', 'Beschreibung'])
        else:
            raise ValueError("No table found in the PDF.")

# Get the folder path where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to export.csv
csv_path = os.path.join(script_dir, "export.csv")
            
exam_info = extract_table_from_pdf(os.path.join(script_dir, "Pruefungsplan.pdf"))
exam_info['Datum'] = pd.to_datetime(exam_info['Datum']).dt.strftime('%Y-%m-%d')

# Vacation dates and durations in days
year = datetime.today().year
csv_path_vac = os.path.join(script_dir, "Feiertage_Semesterstart.xlsx")
df_vacation = pd.read_excel(csv_path_vac)
df_vacation['Startdatum'] = pd.to_datetime(df_vacation['Startdatum']).dt.strftime('%Y-%m-%d')

tagderarbeit= datetime.strptime(f'{year}-05-01', '%Y-%m-%d').strftime('%Y-%m-%d')
auffahrt = datetime.fromisocalendar(year, 20, 4).strftime('%Y-%m-%d')
pfingsten = datetime.fromisocalendar(year, 22, 1).strftime('%Y-%m-%d')
vacation_dates = dict(zip([tagderarbeit, auffahrt, pfingsten], [1,2,1]))  

# read csv file
df = pd.read_csv(csv_path, sep=';', encoding='utf-8')

# generate table for current school semester starting 20.01.2025
start_date = datetime.fromisocalendar(year, 4, 1).strftime('%Y-%m-%d')
end_date = pd.Timestamp.today().normalize()

# convert 'Abw. von' column to datetime format Mi, 04.06.2025 
absence_date = np.hstack(df['Abw. von'])
df['Abw. von']= np.hstack([datetime.strptime(date_str.split(', ')[1], '%d.%m.%Y') for date_str in absence_date])

# filter the dataframe for the current semester
df_current_semester = df[(df['Abw. von'] >= start_date) & (df['Abw. von']<= end_date)].reset_index(drop=True)

# filter duplicate absences
df_current_semester = df_current_semester.drop_duplicates(subset=['Name', 'Abw. von'])

# Sort by name and date
df_current_semester_sorted = df_current_semester.sort_values(['Name', 'Abw. von'])

# join first and last name
df_current_semester_sorted['Name'] = df_current_semester_sorted['Vorname'] + ' ' + df_current_semester_sorted['Name']

# count the number of absences for each student. only zeugnisrelevant and non Jokertag absences are counted
names = df_current_semester_sorted['Name'].unique()
df_current_semester_sorted= df_current_semester_sorted[df_current_semester_sorted["Typ"]=="zeugnisrelevant"]
df_current_semester_sorted= df_current_semester_sorted[df_current_semester_sorted["Grund"]!="Jokertag"]

# Split table unexcused absences and valid absences
df_current_semester_sorted_excused = df_current_semester_sorted[df_current_semester_sorted['Entsch.'] == "Ja"].reset_index(drop=True)
df_current_semester_sorted_unexcused = df_current_semester_sorted[df_current_semester_sorted['Entsch.'] == "Nein"].reset_index(drop=True)

# return how many excused absences each student has
df_excused_absences = is_valid_absence(df_current_semester_sorted_excused)
df_excused_absences_counted = df_excused_absences.groupby('Name')['Abw. von'].count().reset_index()
df_excused_absences_counted.columns = ['Name', 'Entschuldigte Absenzen dieses Semester']

# return how many unexcused absences each student has
df_unexcused_absences = is_valid_absence(df_current_semester_sorted_unexcused)
df_unexcused_absences_counted = df_unexcused_absences.groupby('Name')['Abw. von'].count().reset_index()
df_unexcused_absences_counted.columns = ['Name', 'Unentschuldigte Absenzen dieses Semester']

# merge the two dataframes into one
df_absences_excused_unexcused = pd.merge(df_excused_absences, df_unexcused_absences, on='Name', how='outer')

# merge the two columns unexcused and excused absences into one table
df_filtered_absences = pd.merge(df_excused_absences_counted, df_unexcused_absences_counted, on='Name', how='outer')

# add a column with the number of still excusable absences (absences that are not yet 7 days old)
end_date_seven = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
df_unexcused_past_seven = df_unexcused_absences[(df_unexcused_absences['Abw. von'] >= end_date_seven) & (df_unexcused_absences['Abw. von']<= end_date)].reset_index(drop=True)
df_unexcused_past_seven_counted = df_unexcused_past_seven.groupby('Name')['Abw. von'].count().reset_index()
df_unexcused_past_seven_counted.columns = ['Name', 'davon noch entschuldbar']

# merge the third column into one
df_filtered_absences_seven = pd.merge(df_filtered_absences, df_unexcused_past_seven_counted, on='Name', how='outer')

# add a column with the total number of absences
df_filtered_absences_seven['Total Absenzen dieses Semester'] = df_filtered_absences_seven['Entschuldigte Absenzen dieses Semester'] + df_filtered_absences_seven['Unentschuldigte Absenzen dieses Semester'].fillna(0)

# Return all dates for excused and unexcused absences
excused_dates = df_excused_absences.groupby('Name')['Abw. von'].apply(
    lambda x: ', '.join(x.dt.strftime('%d.%m.%Y'))
)
unexcused_dates = df_unexcused_absences.groupby('Name')['Abw. von'].apply(
    lambda x: ', '.join(x.dt.strftime('%d.%m.%Y'))
)

# Add column for absences, which overlap with exam days
absence_date_values = df_current_semester_sorted['Abw. von'].values 
exam_date_values = pd.to_datetime(exam_info['Datum']).values
mask = absence_date_values[:, None] == exam_date_values[None, :]
overlaps_array = mask.any(1)
dates_colliding = absence_date_values[overlaps_array]

number_absences_names = df_current_semester_sorted['Name'].value_counts(sort = False)
number_absences = number_absences_names.values
block_starts = np.cumsum(np.pad(number_absences[:-1], (1, 0)))
labels = np.repeat(np.arange(len(number_absences)), number_absences)
result = np.bincount(labels, weights=overlaps_array)
overlap_number = result.astype(int).tolist()

#return the colliding dates per student
ends = np.cumsum(overlap_number)
starts = np.concatenate(([0], ends[:-1]))
dates_colliding_per_sus = np.split(dates_colliding.astype('datetime64[D]'), ends[:-1])
number_absences_names = pd.DataFrame({'Name': number_absences_names.index, 'Anzahl Absenzen': number_absences_names.values, 'Überlapp Daten': dates_colliding_per_sus})
number_absences_names['Überlapp mit Prüfungen'] = overlap_number

df_filtered_absences_seven['Daten entschuldigt'] =df_filtered_absences_seven['Name'].map(excused_dates)
df_filtered_absences_seven['Daten unentschuldigt'] = df_filtered_absences_seven['Name'].map(unexcused_dates)
df_filtered_absences_seven_overlap = pd.merge(df_filtered_absences_seven, number_absences_names[['Überlapp mit Prüfungen', 'Überlapp Daten', 'Name']], on='Name', how='outer')

# export both tables to one csv file
csv_path_out_a = os.path.join(script_dir, r".\Absenzen_Export\absenzen_output.csv")
df_filtered_absences_seven_overlap.to_csv(csv_path_out_a, sep=';', encoding='utf-8', index=False)


