#web app
from flask import Flask, render_template, flash, redirect, url_for, request, make_response, send_file

from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand


app = Flask(__name__)
app.config.from_object('config')
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1 #cache

db = SQLAlchemy(app)
migrate = Migrate(app, db)

manager = Manager(app)
manager.add_command('db', MigrateCommand)


#====== routes =========
import os
import os.path
from io import BytesIO
import pandas as pd
from datetime import date
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import json
import glob
import statistics
import seaborn as sns


import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.dfg import algorithm as dfg_miner
from pm4py.visualization.dfg import visualizer as dfg_visualizer

from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter

import sweetviz as sv
import pandas_profiling
from pandas_profiling import ProfileReport
#import webbrowser

#basedir = os.path.abspath(os.path.dirname(__file__))
ds_folder = './'#os.path.dirname(os.path.abspath(__file__))
models_folder = './static/images'
#reports_folder = './templates/reports'
reports_folder = './static'
# === loads ====
def datasetList():
    datasets = [x.split('.')[0] for f in [os.path.join(ds_folder, 'datasets')] for x in os.listdir(f)]
    extensions = [x.split('.')[1] for f in [os.path.join(ds_folder, 'datasets')] for x in os.listdir(f)]
    folders = [f for f in [os.path.join(ds_folder, 'datasets')] for x in os.listdir(f)]
    return datasets, extensions, folders

#Load columns of the dataset
def loadColumns(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'), nrows=0)
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'), nrows=0)
        return df.columns

#Load Dataset    
def loadDataset(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
        return df
#===============

def create_xes(df, ca, ac, ti):
    find_ds = [x.split('.')[0] for f in [os.path.join(ds_folder, 'datasets')] for x in os.listdir(f)]
    path_ds = './datasets'
    if df in find_ds:
        df = pd.read_csv(os.path.join(path_ds, df + '.csv'))
    
    #======= find column datetime type =======
    col_ts = 0
    col_names = dict(df.loc[0])#list(df.columns.values)
    
    col_names = {str(key): str(value) for key, value in col_names.items()}

    for k in col_names:
        for v in col_names[k]:
            if ('/' in v) and ('/' in v):
                col_ts = 1
                att_ts = k
    
    if col_ts == 1:
        df = dataframe_utils.convert_timestamp_columns_in_df(df, timest_format='%d/%m/%Y %H:%M', timest_columns=att_ts)
    
    df.rename(columns={ca: 'case:concept:name'}, inplace=True)
    df.rename(columns={ac: 'concept:name'}, inplace=True)
    df.rename(columns={ti: 'time:timestamp'}, inplace=True)

    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
    events_log = log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    xes_exporter.apply(events_log, os.path.join(ds_folder+"datasets",'file.xes'))

def clear_files():
    if os.path.exists(os.path.join(models_folder,'epm_model.jpg')):
        os.remove(os.path.join(models_folder,'epm_model.jpg'))
    if os.path.exists(os.path.join(models_folder,'epm_dotted.jpg')):
        os.remove(os.path.join(models_folder,'epm_dotted.jpg'))
    if os.path.exists(os.path.join(ds_folder+"datasets",'file.xes')):
        os.remove(os.path.join(ds_folder+"datasets",'file.xes'))
    if os.path.exists(os.path.join(reports_folder,'df_report.html')):
        os.remove(os.path.join(reports_folder,'df_report.html'))
    
def clear_dataset_user():
    files = glob.glob("./datasets/*")
    print(files)
    for f in files:
        if f != './datasets/log_test.csv':# and f != './datasets/log_test2.csv':
            os.remove(f)


    #for f in os.listdir(ds_folder+"datasets"):
    #    os.remove(os.path.join(ds_folder+"datasets", f))


def create_plot(df, c):
    x = df[c].value_counts()#np.linspace(0, 1, N)
    y = df[c].value_counts().index#np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
    
    #fig = go.Figure([go.Bar(x=x, y=y, orientation='h')])
    #fig.write_image(os.path.join(models_folder,'plot_attribute.jpg'))
    #fig.write_html(os.path.join(models_folder,'plot_attribute.html'))
    
    data = [
        go.Pie(
            #===== orientation = h ==========
            values=df['x'], # assign x as the dataframe column 'x'
            labels = df['y']
            #title='Activity(ies)'
        )              
    ]    
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

def create_plot_students(df, c):
    x = df[c].value_counts()#np.linspace(0, 1, N)
    y = df[c].value_counts().index#np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
    
    #fig = go.Figure([go.Bar(x=x, y=y, orientation='h')])
    #fig.write_image(os.path.join(models_folder,'plot_attribute.jpg'))
    #fig.write_html(os.path.join(models_folder,'plot_attribute.html'))
    
    data = [
        go.Pie(
            #===== orientation = h ==========
            values=df['x'], # assign x as the dataframe column 'x'
            labels = df['y']
            #title='Student(s)'            
        )              
    ]    
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

def create_plot2(df, c):
    df[c] = dataframe_utils.convert_timestamp_columns_in_df(df, timest_format='%d/%m/%Y %H:%M', timest_columns='c')
    #print(df)
    x = df[c].value_counts(sort=False)#np.linspace(0, 1, N)
    y = df[c].value_counts(sort=False).index#np.random.randn(N)
    
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
    #df = dataframe_utils.convert_timestamp_columns_in_df(df, timest_format='%d/%m/%Y %H:%M', timest_columns='y')
    
    
    #fig = go.Figure([go.Bar(x=x, y=y, orientation='h')])
    #fig.write_image(os.path.join(models_folder,'plot_attribute.jpg'))
    #fig.write_html(os.path.join(models_folder,'plot_attribute.html'))
    
    data = [
        go.Bar(
            #===== orientation = h ==========
            x=df['y'], # assign x as the dataframe column 'x'
            y=df['x'],            
            marker=dict(color='#123456')         
        )              
    ]
    
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

def create_report_sweet(df):
    df_report = sv.analyze(df)
    df_report.show_html(os.path.join(reports_folder,'df_report.html'), open_browser=False, layout='vertical')
    #df_report.show_html(filepath=os.path.join(reports_folder,'df_report.html'), open_browser=True, layout='vertical')

def create_report_profiling(df):
    profile = ProfileReport(df, title='Report event log', html={'style': {'full_width': True}}, minimal=True)
    profile.to_file(output_file=os.path.join(reports_folder,'df_report.html'))

def getIndexes(dfObj, value, listOfPos):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    #listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

#=============================

@app.route('/index')
@app.route('/')
def index():
    clear_files()
    clear_dataset_user()
    return render_template('index.html')

#==== show datasets ====
@app.route('/viewds', methods = ['GET', 'POST'])
def viewds():
    clear_files()
    datasets,_,folders = datasetList()
    originalds = []
    featuresds = []
    for i in range(len(datasets)):
        if folders[i] == 'datasets': originalds += [datasets[i]]
        else: featuresds += [datasets[i]]
    if request.method == 'POST':
            f = request.files['file']
            f.save(os.path.join(ds_folder+'datasets', f.filename))
            return redirect(url_for('viewds'))
    return render_template('viewds.html', originalds = originalds, featuresds = featuresds)

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/publications')
def publications():
    return render_template('publications.html')


@app.route('/datasets/')
def datasets():
    return redirect('/')

@app.route('/datasets/<dataset>', methods=['GET', 'POST'])
#def dataset(description = None, head = None, ev = None, att = None, missing_v = None, att_ts = None, dataset = None):
def dataset(ev = None, head = None, tail = None, att = None, att_ts = None, dataset = None, columns = None, bar = None, df = None, description = None):
    df = loadDataset(dataset)
    columns = loadColumns(dataset)

    #==== update df correct order - LMS-Moodle
    #df_aux = pd.DataFrame(columns=columns)
    #len_df = len(df)
    #j = len_df - 1
    #for i in range(len_df):
    #    df_aux = df_aux.append(df.iloc[j], ignore_index=True)
    #    j = j - 1
    #df = df_aux

    full_report = request.form.get('radio_report')
    create_report_profiling(df)
    if full_report == 'radio_sweet':
        create_report_sweet(df)
    elif full_report == 'radio_profiling':
        create_report_profiling(df)
        #new = 2 # open in a new tab, if possible
        #url = os.path.join(reports_folder,'df_report.html')
        #webbrowser.open(url,new=new)

    #========plot some attribute=======
    #attrib_plot = request.form.get('att-plot')
    #if attrib_plot != None:
    #    bar = create_plot(df, attrib_plot)
    
    #df_filter = df.groupby(["Component", "Event Context", "Event Name"], sort=True, as_index=True)["Event Name"].count().reset_index(name="count")
    #len_df = len(df_filter)
    #for i in range(len_df):
    #  for j in range(i):
    #    if df_filter.iloc[i, 0] == df_filter.iloc[j, 0]:
    #        df_filter.iloc[i, 0] = ''
    #    if df_filter.iloc[i, 1] == df_filter.iloc[j, 1]:
    #        df_filter.iloc[i, 1] = ''
    
    df.reset_index(drop=True, inplace=True)
       
    #if os.path.exists(os.path.join(models_folder,'epm_model.jpg')):
    #    os.remove(os.path.join(models_folder,'epm_model.jpg'))

    #=========================================
    try:
        #missing_v = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
        description = df.describe().round(2)
        description = description.rename(index={'count': 'Occurrences', 'unique': 'Unique values', 'top': 'Majority', 'freq': 'Majority-occurrences'})
        head = df.head(10)
        tail = df.tail(10)
        ev = df.shape[0]
        att = df.shape[1]        
    except: pass
    
    #==== update df correct order - LMS-Moodle
    

    return render_template('dataset.html',
                           ev = ev,
                           att = att,
                           att_ts = att_ts,
                           #missing_v = missing_v,
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           tail = tail.to_html(index=False, classes='table table-striped table-hover'),                           
                           dataset = dataset,
                           columns=columns,
                           bar=bar,
                           df = df.to_html(index=False, classes='table table-striped table-hover'),                           
                           description = description.to_html(index=True, classes='table table-striped table-hover')
                           )

@app.route('/pm/<dataset>', methods=['GET', 'POST'])
def pm(dataset):
    df = loadDataset(dataset)
    columns = loadColumns(dataset)
    return render_template('pm.html', df=df, columns=columns, dataset=dataset)

@app.route('/modelprocess/<dataset>', methods=['GET', 'POST'])
def modelprocess(dataset):          
    df = dataset
    df_plot = loadDataset(dataset)
    ca = request.form.get('case')
    ac = request.form.get('activity')
    ti = request.form.get('timestamp')
    
    #len(df_plot[ac].value_counts().index)
    
    '''
    f = open(os.path.join(ds_folder+"datasets", df+".csv"), 'r')
    log_file = dict()
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        parts = line.split(',')        
        timestamp_log = parts[0]        
        case_log = parts[1]        
        context_log = parts[2]
        comp_log = parts[3]
        name_log = parts[4]
        if case_log not in log_file:
            log_file[case_log] = []
        event = (timestamp_log, case_log, context_log, comp_log, name_log)
        log_file[case_log].append(event)
    f.close()        
        
    #for case_log in log_file:        
    #    for (timestamp_log, case_log, context_log, comp_log, name_log) in log_file[case_log]:
    #        print(timestamp_log, case_log, context_log, comp_log, name_log)

    F = dict()    
    for case_log in log_file:
        for i in range(0, len(log_file[case_log])-1):
            ai = log_file[case_log][i][4]#indice do activity name
            aj = log_file[case_log][i+1][4]
            if ai not in F:
                F[ai] = dict()                
            if aj not in F[ai]:
                F[ai][aj] = 0                
            F[ai][aj] += 1            
    '''            
      
    #pm alg options
    pm_alg = request.form.get('pm_method')
       
    create_xes(df, ca, ac, ti)
    
    bar = create_plot(df_plot, ac)
    bar2 = create_plot_students(df_plot, ca)
    bar3 = create_plot2(df_plot, ti)

    log = xes_importer.apply(os.path.join(ds_folder+"datasets",'file.xes'))
    
    start_activities = pm4py.get_start_activities(log)    
    end_activities = pm4py.get_end_activities(log)

    students = []
    students_events = []
    students_traces = []
    
    for case_index, case in enumerate(log):
        students.append(case.attributes["concept:name"])
        students_events.append(len(case))
        create_trace = ''      
        for event_index, event in enumerate(case):
            if event_index == 0:
                create_trace = create_trace+' '+event["concept:name"]
            else:
                create_trace = create_trace+' > '+event["concept:name"]
        students_traces.append(create_trace)
    
    
    dotted = sns.scatterplot(x=df_plot[ca].index, y=df_plot[ca], hue=df_plot[ac])
    dotted.tick_params(axis='x', rotation=90)
    dotted.set_xticklabels([])
    fig = dotted.get_figure()
    fig.set_size_inches(10, 6)
    fig.savefig(os.path.join(models_folder,'epm_dotted.jpg'))
    #fig = plt.Figure()

    total_events = df_plot.shape[0]
    first_event = df_plot[ti].iloc[0]
    last_event = df_plot[ti].iloc[total_events-1]
    classes_events = len(df_plot[ac].value_counts().index)
    max_events = max(students_events, key=int)
    min_events = min(students_events, key=int)
    mean_events = round(sum(students_events) / len(students_events), 2)
    median_events = statistics.median(students_events)
    mode_events = statistics.mode(students_events)
    
    dfg = dfg_miner.apply(log)#control-flow pairs-values
    
    count_students = len(students)
    if pm_alg == 'hm':
        heu_net = heuristics_miner.apply_heu(log, parameters={"dependency_thresh": -1, "and_measure_thresh": 1, "dfg_pre_cleaning_noise_thresh":0.0})        
        gviz = hn_visualizer.apply(heu_net)    
        hn_visualizer.save(gviz, os.path.join(models_folder,'epm_model.jpg'))
    else:        
        #dfg = dfg_miner.apply(log)            
        gviz = dfg_visualizer.apply(dfg, log=log, variant=dfg_visualizer.Variants.FREQUENCY)
        dfg_visualizer.save(gviz, os.path.join(models_folder,'epm_model.jpg'))
    
    return render_template('graphs.html', dfg = dfg, ca=ca, ac=ac, ti=ti, count_students=count_students, students=students, students_traces=students_traces, students_events=students_events, bar=bar, bar2 = bar2, bar3 = bar3, dataset=dataset, start_activities=start_activities, end_activities=end_activities, total_events=total_events,max_events=max_events, min_events=min_events, mean_events=mean_events, classes_events=classes_events, first_event=first_event,last_event=last_event, median_events=median_events, mode_events=mode_events)


@app.route('/preprocessing/<dataset>', methods=['GET', 'POST'])
def preprocessing(dataset = dataset):
    columns = loadColumns(dataset)
    df_prep = loadDataset(dataset)
    return render_template('preprocessing.html', dataset = dataset, columns=columns, df_prep = df_prep)

@app.route('/datasets/<dataset>/preprocessed_dataset/', methods=['POST'])
def preprocessed_dataset(dataset):    
    manualFeatures = request.form.getlist('manualfeatures')
    datasetName = request.form.get('newdataset')
    manualRows = request.form.getlist('manual_rows')
    mode_preprocessing = request.form.get('radio_preprocessing')
    sort_log = request.form.get('radio_sort')
    anonymize_log = request.form.get('radio_anonymize')
    anonymize_student = request.form.get('anonymize_names')

    columns = loadColumns(dataset)    
    df = loadDataset(dataset)
    filename = dataset + '_'
    
    #==== update df reverse order - LMS-Moodle =====#
    if sort_log == 'radio_sort_yes':        
        df_aux = pd.DataFrame(columns=columns)
        df_aux = df.iloc[::-1]
        df = df_aux
        
        '''
        df_aux = pd.DataFrame(columns=columns)
        len_df = len(df)
        j = len_df - 1
        for i in range(len_df):
            df_aux = df_aux.append(df.iloc[j], ignore_index=True)
            j = j - 1
        df = df_aux
        '''
    #===============================================#
    if anonymize_log == 'radio_anonymize_yes':
        df_aux = pd.DataFrame(columns=columns)        
        names = df[anonymize_student].unique()
        df_aux = df
        len_names = len(names)
        n = 1        
        for key in names:
            df_aux[anonymize_student] = df[anonymize_student].replace([key],'Anonymous'+str(n))
            n = n + 1
        df = df_aux
    #===============================================#

    listOfPos = list()
    if mode_preprocessing == 'radio_keep':
        if not manualFeatures:
            df = df[columns]
        else:
            df = df[manualFeatures]

        if len(manualRows) > 0:
            df_rows = pd.DataFrame(columns=columns)     
            #listOfPos = list()
            resultDict = {}
            
            # Iterate over the list of elements one by one
            for elem in manualRows:        
                # Check if the element exists in dataframe values
                if elem in df.values:
                    listOfPositions = getIndexes(df, elem, listOfPos)
                    
            for i in range(len(listOfPositions)):
                if listOfPositions[i][0] in df.index:                
                    df_rows = df_rows.append(df.iloc[listOfPositions[i][0]], ignore_index=True)
            df = df_rows
    elif mode_preprocessing == 'radio_delete':    
        df = df.drop(manualFeatures, axis=1)
        if len(manualRows) > 0:
            #listOfPos = list()                
            resultDict = {}
            # Iterate over the list of elements one by one
            for elem in manualRows:        
                # Check if the element exists in dataframe values
                if elem in df.values:
                    listOfPositions = getIndexes(df, elem, listOfPos)
                    
            for i in range(len(listOfPositions)):
                if listOfPositions[i][0] in df.index:
                    df = df.drop(listOfPositions[i][0])
                    #print('Position ', i, ' (Row index , Column Name) : ', listOfPositions[i])

    filename += str(datasetName) + '.csv'
    df.to_csv(os.path.join(ds_folder+"datasets", filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])

#=======================
# No caching at all for API endpoints.
'''
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response
'''

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


#=======================

if __name__ == "__main__":
    #app.run(debug=TRUE)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
