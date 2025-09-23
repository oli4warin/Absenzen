import pandas as pd
import numpy as np
from datetime import datetime
import re
import streamlit as st
import pymupdf


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



# Get the folder path where the script is located
#script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to export.csv
#csv_path = os.path.join(script_dir, "export.csv")
csv_path = "export.csv"

# Vacation dates and durations in days
year = datetime.today().year
tagderarbeit= datetime.strptime(f'{year}-05-01', '%Y-%m-%d').strftime('%Y-%m-%d')
auffahrt = datetime.fromisocalendar(year, 20, 4).strftime('%Y-%m-%d')
pfingsten = datetime.fromisocalendar(year, 22, 1).strftime('%Y-%m-%d')
vacation_dates = dict(zip([tagderarbeit, auffahrt, pfingsten], [1,2,1]))  

st.title("Absenzenübersicht Ermittler")
# read csv file
uploaded_file = st.file_uploader("Lade die Absenzen Datei export.csv hoch")
st.text("Die hochgeladene Datei")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
  st.write(df)
else:
    st.error("Bitte lade die Absenzen Datei hoch.")
    st.stop()
#df = pd.read_csv(csv_path, sep=';', encoding='utf-8') # to debug

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
df_unexcused_past_seven_counted.columns = ['Name', 'Davon noch entschuldbar']

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

df_filtered_absences_seven['Daten entschuldigt'] =df_filtered_absences_seven['Name'].map(excused_dates)
df_filtered_absences_seven['Daten unentschuldigt'] = df_filtered_absences_seven['Name'].map(unexcused_dates)

#uploaded_file_exam = extract_table_from_pdf(Pruefungsplan.pdf)?? # to debug
if st.checkbox("Zeige Überlapp mit Prüfungen"):

    
    uploaded_file_exam = st.file_uploader("Lade die Prüfungsplan Datei Pruefungsplan.pdf hoch")
    if uploaded_file_exam is None:
        st.error("Bitte lade die Prüfungsplan Datei hoch.")
        st.stop()
    else:
        doc = pymupdf.open(stream=uploaded_file_exam.getvalue(), filetype="pdf")
        
        date_pattern = r"\b\d{2}\.\d{2}\.\d{4}\b"

        matches = []

        for page in doc:
            text = page.get_text("text")
            found_dates = re.findall(date_pattern, text)
            if found_dates:
                matches.extend(found_dates)  # fügt alle gefundenen Daten hinzu

        # In ein NumPy Array umwandeln
        date_array = np.array(matches)
        #exam_info = extract_table_from_pdf(uploaded_file_exam)
        ##drop nan rows
        #exam_info
        #exam_info = exam_info.dropna(subset=['Datum'])
        #exam_info['Datum'] = pd.to_datetime(exam_info['Datum']).dt.strftime('%Y-%m-%d')
        st.write("Die hochgeladene Prüfungsplan Datei")
        st.write(date_array)

        # Add column for absences, which overlap with exam days
        absence_date_values = df_current_semester_sorted['Abw. von'].values 
        exam_date_values = pd.to_datetime(date_array).values
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
        dates_colliding_per_sus = np.split(dates_colliding.astype('datetime64[D]').astype(str), ends[:-1])
        number_absences_names = pd.DataFrame({'Name': number_absences_names.index, 'Anzahl Absenzen': number_absences_names.values, 'Verpasste Prüfungen Daten': dates_colliding_per_sus})
        number_absences_names['Verpasste Prüfungen'] = overlap_number
        # Indicate, if absence stops right before the exam
        df_filtered_absences_seven_overlap = pd.merge(df_filtered_absences_seven, number_absences_names[['Verpasste Prüfungen', 'Verpasste Prüfungen Daten', 'Name']], on='Name', how='outer')

else:
    df_filtered_absences_seven_overlap = df_filtered_absences_seven.copy()
# export both tables to one csv file
#csv_path_out_a = os.path.join(script_dir, r".\Absenzen_Export\absenzen_output.csv")
csv_path_out_a= "Absenzen_Export\absenzen_output.csv"
#df_filtered_absences_seven_overlap.to_csv(csv_path_out_a, sep=';', encoding='utf-8', index=False)

st.subheader("Absenzenübersicht")
st.text("Klicke auf die Spaltenüberschrift, um die Tabelle zu sortieren und lade die Datei herunter.")
st.write(df_filtered_absences_seven_overlap)
st.download_button(
    label="Download Absenzenübersicht",
    data=df_filtered_absences_seven_overlap.to_csv(sep=';', encoding='utf-8', index=False).encode('utf-8'),
    file_name='Absenzenübersicht.csv',
    mime='text/csv'
)

st.markdown("Das Programm befindet sich in der Beta Phase. Bitte meldet Fehler oder Verbesserungsvorschläge an julia.nguyen@sbl.ch")
