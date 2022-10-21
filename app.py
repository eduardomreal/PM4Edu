#web app
from flask import Flask, session, render_template, flash, redirect, url_for, request, make_response, send_file

from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from flask_wtf import Form
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import DataRequired

from flask_login import LoginManager
from flask_login import login_user, logout_user
from flask_login import login_required, current_user

import time

from random import randint

from operator import itemgetter

#==== form ======
class LoginForm(Form):
    username = StringField("username", validators=[DataRequired()])
    password = PasswordField("password", validators=[DataRequired()])
    remember_me = BooleanField("remeber_me")


app = Flask(__name__)
app.config.from_object('config')
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1 #cache

db = SQLAlchemy(app)
migrate = Migrate(app, db)


#==== tables ======
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True)
    password = db.Column(db.String)
    name = db.Column(db.String)
    email = db.Column(db.String, unique=True)
    token = db.Column(db.String(20))

    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_active(self):
        return True
    
    @property
    def is_anonymous(self):
        return False
    
    def get_id(self): #is a method
        return str(self.id) #it is using unicode, converter string

    def __init__(self, username, password, name, email):
        self.username = username
        self.password = password
        self.name = name
        self.email = email
        self.token = token
    
    def __repr__(self):
        return "<User %r>" % self.username
#====== end tables ======


manager = Manager(app)
manager.add_command('db', MigrateCommand)

lm = LoginManager()
lm.init_app(app)


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

import shutil

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

from datetime import datetime

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

def datasetList_users():
    datasets = [x.split('.')[0] for f in [os.path.join('./datasets_temp/'+current_user.username+'/')] for x in os.listdir(f)]
    extensions = [x.split('.')[1] for f in [os.path.join('./datasets_temp/'+current_user.username+'/')] for x in os.listdir(f)]
    folders = [f for f in [os.path.join('./datasets_temp/'+current_user.username+'/')] for x in os.listdir(f)]
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
        #elif extension == 'xlsx':
            #df = pd.read_excel(os.path.join(folders[datasets.index(dataset)], dataset + '.xlsx'), nrows=0)
        return df.columns

def loadColumns_users(dataset):
    datasets, extensions, folders = datasetList_users()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'csv':
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
        #elif extension == 'xlsx':
            #df = pd.DataFrame(pd.read_excel(os.path.join(folders[datasets.index(dataset)], dataset + '.xlsx')))
            #df = pd.read_excel(os.path.join(folders[datasets.index(dataset)], dataset + '.xlsx'))
        return df

def loadDataset_users(dataset):
    datasets, extensions, folders = datasetList_users()    
    if dataset in datasets:        
        extension = extensions[datasets.index(dataset)]
        if extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
        return df
#===============

def create_xes(df, ca, ac, ti):
    df_name = df
    if df_name == 'log_test':        
        find_ds = [x.split('.')[0] for f in [os.path.join(ds_folder, 'datasets')] for x in os.listdir(f)]
        path_ds = './datasets'
        if df_name in find_ds:
            df = pd.read_csv(os.path.join(path_ds, df + '.csv'))
    else:        
        find_ds = [x.split('.')[0] for f in [os.path.join(ds_folder, 'datasets_temp/'+current_user.username)] for x in os.listdir(f)]
        path_ds = './datasets_temp/'+current_user.username
        if df_name in find_ds:
            df = pd.read_csv(os.path.join(path_ds, df + '.csv'))
    
    #find_ds = [x.split('.')[0] for f in [os.path.join(ds_folder, 'datasets')] for x in os.listdir(f)]
    #path_ds = './datasets'
    #if df in find_ds:
    #    df = pd.read_csv(os.path.join(path_ds, df + '.csv'))
    
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

    xes_exporter.apply(events_log, os.path.join(ds_folder+"datasets_temp/"+current_user.username, df_name+'.xes'))
    
    #xes_exporter.apply(events_log, os.path.join(ds_folder+"datasets",'file.xes'))

def clear_files():
    if os.path.exists(os.path.join(models_folder,'epm_model.jpg')):
        os.remove(os.path.join(models_folder,'epm_model.jpg'))
    if os.path.exists(os.path.join(models_folder,'epm_dotted.jpg')):
        os.remove(os.path.join(models_folder,'epm_dotted.jpg'))
    if os.path.exists(os.path.join(ds_folder+"datasets",'file.xes')):
        os.remove(os.path.join(ds_folder+"datasets",'file.xes'))
    if os.path.exists(os.path.join(reports_folder,'df_report.html')):
        os.remove(os.path.join(reports_folder,'df_report.html'))
    
    if os.path.exists(os.path.join('./datasets_temp/', current_user.username)):
        filelist = glob.glob(os.path.join('./datasets_temp/', current_user.username, "*.xes"))
        for f in filelist:
            os.remove(f)   
    
def clear_dataset_user():
    files = glob.glob("./datasets/*")
    #print(files)
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

'''
def list_bag(ds):
    c = loadColumns(ds)
    p = loadDataset(ds)
    data = []    
    for l in range(len(c)):
        list_col = []
        for i, j in p[c].items():            
            list_col.append(j)
        data.append(list_col) #list of list
    return data
'''
#=============================

@lm.user_loader
def load_user(id):    
    return User.query.filter_by(id=id).first()

@lm.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('login'))

@app.route('/index')
@app.route('/')
@login_required
def index():
    clear_files()
    #clear_dataset_user()
    return render_template('index.html')

@app.route("/login", methods=["GET","POST"])
def login():    
    if current_user.is_authenticated:
        return redirect('/index')    
    form = LoginForm()    
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()        
        '''
        if user.username in session:
            session['username'] = session.get('username') + 1
        else
            session['username'] = 1
        '''
        if user.token == "False":
            if user and user.password == form.password.data:                
                login_user(user)                
                user.token = "True"
                #db.session.commit()
                #session['username'] = current_user.username
                if os.path.exists(os.path.join('./datasets_temp/', current_user.username)):                    
                    shutil.rmtree("./datasets_temp/"+current_user.username)                                       
                path = os.path.join('./datasets_temp/', current_user.username)
                #user_now = randint(1,200)
                '''
                path_xes = os.path.join('./datasets_temp/', current_user.username+"/xes/")
                path_prep = os.path.join('./datasets_temp/', current_user.username+"/prep/")
                path_models = os.path.join('./datasets_temp/', current_user.username+"/models/")
                '''
                os.mkdir(path, mode=0o777)
                '''
                os.mkdir(path_xes, mode=0o777)
                os.mkdir(path_prep, mode=0o777)
                os.mkdir(path_models, mode=0o777)
                '''
                return redirect(url_for("index"))
                flash("Logged in.")
            else:
                flash("Invalid login.")
        else:
            flash("User already logged!.")
    return render_template("login.html", form=form)

@app.route('/logout')
@login_required
def logout():    
    shutil.rmtree("./datasets_temp/"+current_user.username)
    '''
    files = glob.glob("./datasets_temp/"+current_user.username+"/*")    
    for f in files:        
        os.remove(f)
    os.rmdir(os.path.join('./datasets_temp/', current_user.username))
    '''
    user = User.query.filter_by(username=current_user.username).first()
    user.token = "False"
    #db.session.commit()
    #session.pop('username', None)
    logout_user()    
    flash("Logged out.")
    return redirect(url_for('index'))


'''
@app.route("/teste/<info>")
@app.route("/teste", defaults={"info": None})
def teste(info):
    
    #=== select ====
    #r = User.query.filter_by(username="eduardomreal").all()
    #r = User.query.filter_by(password="1234").all()

    #r = User.query.filter_by(username="eduardomreal").first()
    
    #print(r)
    #print(r.username, r.name)

    #return "Ok!"
    
    #==== insert =====
    #i = User("eduardomreal", "1234", "Edu Real", "eduardomreal@gmail.com")
    #db.session.add(i)
    #db.session.commit()
    return "Ok!"
'''

#==== show datasets ====
@app.route('/viewds', methods = ['GET', 'POST'])
@login_required
def viewds():
    clear_files()
    datasets,_,folders= datasetList()
    originalds = []    
    featuresds = []

    #====users=====
    datasets_users,_users,folders_users= datasetList_users()
    originalds_users = []    
    featuresds_users = []
    #==============

    logs = []
    d_file = request.form.get('download_yes')   
    #arquivos = dict((arquivo, os.path.getsize('./datasets/'+arquivo)/1e+6) for arquivo in os.listdir('./datasets/'))
    
    
    for i in range(len(datasets)):
        if folders[i] == 'datasets': originalds += [datasets[i]]
        else: featuresds += [datasets[i]]
    
    for i in range(len(datasets_users)):
        if folders_users[i] == 'datasets_temp/'+current_user.username: originalds_users += [datasets_users[i]]
        else: featuresds_users += [datasets_users[i]]
    
    featuresds_size = dict((arquivo, os.path.getsize('./datasets/'+arquivo+'.csv')/1e+3) for arquivo in featuresds)
    featuresds_size_users = dict((arquivo_users, os.path.getsize('./datasets_temp/'+current_user.username+'/'+arquivo_users+'.csv')/1e+3) for arquivo_users in featuresds_users)
        
    '''
    for i in range(len(datasets)):
        if _[i] == 'xlsx':
            read_file = pd.read_excel('./datasets/'+datasets[i]+'.xlsx', engine='openpyxl')
            read_file.to_csv('./datasets/'+datasets[i]+".csv", index = None, header=True)
            os.remove('./datasets/'+datasets[i]+".xlsx")
        logs.append(datasets[i]+"."+_[i])
    '''    
    if request.method == 'POST':
            f = request.files['file']            
            #f.save(os.path.join(ds_folder+'datasets', f.filename))            
            path_user = os.path.join('./datasets_temp/'+current_user.username)
            f.save(os.path.join(path_user, f.filename))
            return redirect(url_for('viewds'))
    return render_template('viewds.html', originalds = originalds, featuresds = featuresds, featuresds_size = featuresds_size, originalds_users = originalds_users, featuresds_users = featuresds_users, featuresds_size_users = featuresds_size_users)

@app.route('/tutorial')
@login_required
def tutorial():
    return render_template('tutorial.html')

@app.route('/publications')
@login_required
def publications():
    return render_template('publications.html')


@app.route('/datasets/')
@login_required
def datasets():
    return redirect('/')

@app.route('/datasets/<dataset>', methods=['GET', 'POST'])
@login_required
#def dataset(description = None, head = None, ev = None, att = None, missing_v = None, att_ts = None, dataset = None):
def dataset(ev = None, head = None, tail = None, att = None, att_ts = None, dataset = None, columns = None, bar = None, df = None, description = None):
    print(dataset)
    if dataset == 'log_test':
        df = loadDataset(dataset)
        columns = loadColumns(dataset)
    else:
        df = loadDataset_users(dataset)
        columns = loadColumns_users(dataset)
    
    #print(df.memory_usage(deep = True).sum())
    
    '''
    if 'Usuário afetado' in columns:
        df.drop('Usuário afetado', inplace=True, axis=1)
    if 'Descrição' in columns:
        df.drop('Descrição', inplace=True, axis=1)
    if 'Origem' in columns:
        df.drop('Origem', inplace=True, axis=1)
    if 'endereço IP' in columns:
        df.drop('endereço IP', inplace=True, axis=1)

    if 'Componente' in columns:
        df.drop(df.loc[df['Componente']=='Sistema'].index, inplace=True)
        df.drop(df.loc[df['Componente']=='Logs'].index, inplace=True)
    if 'Component' in columns:
        df.drop(df.loc[df['Component']=='Sistema'].index, inplace=True)
        df.drop(df.loc[df['Component']=='Logs'].index, inplace=True)
    #print(df.memory_usage(deep = True).sum())
    '''
    if dataset == 'log_test':        
        df.to_csv(os.path.join(ds_folder+'datasets', dataset+'.csv'), index=False)#atualiza csv
    else:        
        df.to_csv(os.path.join('./datasets_temp/'+current_user.username, dataset+'.csv'), index=False)#atualiza csv

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
@login_required
def pm(dataset):
    if dataset == 'log_test':
        df = loadDataset(dataset)
        columns = loadColumns(dataset)
    else:
        df = loadDataset_users(dataset)
        columns = loadColumns_users(dataset)
    #df = loadDataset(dataset)
    #columns = loadColumns(dataset)
    return render_template('pm.html', df=df, columns=columns, dataset=dataset)

@app.route('/modelprocess/<dataset>', methods=['GET', 'POST'])
@login_required
def modelprocess(dataset):    
    df = dataset
    if dataset == 'log_test':
        df_plot = loadDataset(dataset)
    else:
        df_plot = loadDataset_users(dataset)
    #df_plot = loadDataset(dataset)
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

    #if dataset == 'log_test':
    #    log = xes_importer.apply(os.path.join(ds_folder+"datasets", dataset+'.xes'))
    #else:
    log = xes_importer.apply(os.path.join('./datasets_temp/', current_user.username, dataset+'.xes'))

    start_activities = pm4py.get_start_activities(log)
    end_activities = pm4py.get_end_activities(log)

    start_activities = {k: v for k, v in sorted(start_activities.items(), key=lambda item: item[1], reverse=True)}
    end_activities = {k: v for k, v in sorted(end_activities.items(), key=lambda item: item[1], reverse=True)}
    #end_activities = end_activities_sorted

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
    #median_events = statistics.median(students_events)
    #mode_events = statistics.mode(students_events)
    
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
    return render_template('graphs.html', dfg = dfg, ca=ca, ac=ac, ti=ti, count_students=count_students, students=students, students_traces=students_traces, students_events=students_events, bar=bar, bar2 = bar2, bar3 = bar3, dataset=dataset, start_activities=start_activities, end_activities=end_activities, total_events=total_events,max_events=max_events, min_events=min_events, mean_events=mean_events, classes_events=classes_events, first_event=first_event,last_event=last_event)


@app.route('/preprocessing/<dataset>', methods=['GET', 'POST'])
@login_required
def preprocessing(dataset = dataset):
    #columns = loadColumns(dataset)
    if dataset == 'log_test':
        df_prep = loadDataset(dataset)
    else:
        df_prep = loadDataset_users(dataset)
    
    #======= find column datetime type =======
    col_ts = 0
    col_names = dict(df_prep.loc[0])#list(df.columns.values)
    col_names = {str(key): str(value) for key, value in col_names.items()}
    for k in col_names:
        for v in col_names[k]:
            if ('/' in v) and ('/' in v):
                col_ts = 1
                att_ts = k
    #==== get date part timestamp ====
    ts = []
    #row_df = 0
    for i in df_prep[att_ts]:
        i = i.split(" ")[0]
        #df_prep[att_ts].loc[row_df] = i
        #row_df+=1
        ts.append(i)
    df_ts = pd.DataFrame(ts) #list to dataframe    
    df_prep[att_ts] = df_ts
    #=======================================
    #df_prep.drop(att_ts, axis=1, inplace=True) #excluir coluna timestamp    
    columns = df_prep.columns.values #nomes das colunas
    
    return render_template('preprocessing.html', dataset = dataset, columns=columns, df_prep = df_prep)

@app.route('/preprocess_names_choice_attribute/<dataset>', methods=['GET', 'POST'])
@login_required
def preprocess_names_choice_attribute(dataset = dataset):
    #choice_names_attribute = request.form.get('names_attribute')    
    if dataset == 'log_test':
        df_prep = loadDataset(dataset)
    else:
        df_prep = loadDataset_users(dataset)
    columns = df_prep.columns.values #nomes das colunas

    return render_template('names_choice-attribute.html', dataset = dataset, df_prep = df_prep, columns=columns)

@app.route('/attributes_delete/<dataset>', methods=['GET', 'POST'])
@login_required
def attributes_delete(dataset = dataset):
    if dataset == 'log_test':
        df_prep = loadDataset(dataset)
    else:
        df_prep = loadDataset_users(dataset)
    
    columns = df_prep.columns.values #nomes das colunas
    
    return render_template('attributes_delete.html', dataset = dataset, columns=columns, df_prep = df_prep)

@app.route('/names_delete/<dataset>', methods=['GET', 'POST'])
@login_required
def names_delete(dataset = dataset):
    choice_names_attribute = request.form.get('names_attribute')
    if dataset == 'log_test':
        df_prep = loadDataset(dataset)
    else:
        df_prep = loadDataset_users(dataset)     
    columns = df_prep.columns.values #nomes das colunas
    
    return render_template('names_delete.html', dataset = dataset, columns=columns, df_prep = df_prep, choice_names_attribute = choice_names_attribute)

@app.route('/elements_delete/<dataset>', methods=['GET', 'POST'])
@login_required
def elements_delete(dataset = dataset):
    if dataset == 'log_test':
        df_prep = loadDataset(dataset)
    else:
        df_prep = loadDataset_users(dataset)
    
    #======= find column datetime type =======
    col_ts = 0
    col_names = dict(df_prep.loc[0])#list(df.columns.values)
    col_names = {str(key): str(value) for key, value in col_names.items()}
    for k in col_names:
        for v in col_names[k]:
            if ('/' in v) and ('/' in v):
                col_ts = 1
                att_ts = k
    #==== get date part timestamp ====
    ts = []
    #row_df = 0
    for i in df_prep[att_ts]:
        i = i.split(" ")[0]        
        ts.append(i)
    df_ts = pd.DataFrame(ts) #list to dataframe    
    df_prep[att_ts] = df_ts
    #=======================================
    columns = df_prep.columns.values #nomes das colunas
    
    return render_template('elements_delete.html', dataset = dataset, columns=columns, df_prep = df_prep)

@app.route('/anonymize/<dataset>', methods=['GET', 'POST'])
@login_required
def anonymize(dataset = dataset):      
    if dataset == 'log_test':
        df_prep = loadDataset(dataset)
    else:
        df_prep = loadDataset_users(dataset)
    columns = df_prep.columns.values #nomes das colunas
    return render_template('anonymize.html', dataset = dataset, df_prep = df_prep, columns=columns)

@app.route('/inverse_order/<dataset>', methods=['GET', 'POST'])
@login_required
def inverse_order(dataset = dataset):      
    if dataset == 'log_test':
        df_prep = loadDataset(dataset)
    else:
        df_prep = loadDataset_users(dataset)
    columns = df_prep.columns.values #nomes das colunas    
    return render_template('inverse-order.html', dataset = dataset, df_prep = df_prep, columns=columns)

@app.route('/datasets/<dataset>/preprocessed_attribute_deleted/', methods=['POST'])
@login_required
def preprocessed_attribute_deleted(dataset):    
    manualFeatures = request.form.getlist('manualfeatures')
    datasetName = request.form.get('newdataset')
    mode_preprocessing = request.form.get('radio_preprocessing')
    
    #====== create log/information about exclusions (or keep) =====
    if manualFeatures:
        textfile = open(os.path.join(ds_folder+"datasets_temp", "log-columns_file-"+datasetName+".txt"), "w")
        for element in manualFeatures:
            textfile.write(element + "\n")
        textfile.close()        
        data = ""        
        if manualFeatures:
            with open(os.path.join(ds_folder+"datasets_temp", "log-columns_file-"+datasetName+".txt")) as fp:
                data = fp.read()
        if os.path.exists(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt")):
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'a') as fp:
                fp.write(data)
        else:
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'w') as fp:
                fp.write(data)

    
    if dataset == 'log_test':
        '''
        if manualFeatures:
            textfile = open(os.path.join(ds_folder+"datasets", "log-columns_file-"+datasetName+".txt"), "w")
            for element in manualFeatures:
                textfile.write(element + "\n")
            textfile.close()    
        data = ""
        if manualFeatures:
            with open(os.path.join(ds_folder+"datasets", "log-columns_file-"+datasetName+".txt")) as fp:
                data = fp.read()        
        if os.path.exists(os.path.join(ds_folder+"datasets", "log-joined_file.txt")):
            with open(os.path.join(ds_folder+"datasets", "log-joined_file.txt"), 'a') as fp:
                fp.write(data)
        else:
            with open(os.path.join(ds_folder+"datasets", "log-joined_file.txt"), 'w') as fp:
                fp.write(data)
    '''
        columns = loadColumns(dataset)    
        df = loadDataset(dataset)
    
    else:#se log uploaded   
        #dataset = datasetName
        columns = loadColumns_users(dataset)    
        df = loadDataset_users(dataset)
        '''
        if manualFeatures:
            textfile = open(os.path.join(ds_folder+"datasets_temp", "log-columns_file-"+datasetName+".txt"), "w")
            for element in manualFeatures:
                textfile.write(element + "\n")
            textfile.close()        
        data = ""        
        if manualFeatures:
            with open(os.path.join(ds_folder+"datasets_temp", "log-columns_file-"+datasetName+".txt")) as fp:
                data = fp.read()
        if os.path.exists(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt")):
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'a') as fp:
                fp.write(data)
        else:
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'w') as fp:
                fp.write(data)
        '''
    #===============================================

    filename = dataset + '_'
        
    listOfPos = list()
    if mode_preprocessing == 'radio_keep':
        if not manualFeatures:
            df = df[columns]
        else:
            df = df[manualFeatures]        
    elif mode_preprocessing == 'radio_delete':    
        df = df.drop(manualFeatures, axis=1)

    filename += str(datasetName) + '.csv'
    #if dataset == 'log_test':
    #    df.to_csv(os.path.join(ds_folder+"datasets", filename), index=False)        
    #else:
    df.to_csv(os.path.join('./datasets_temp/'+current_user.username, filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])

@app.route('/datasets/<dataset>/preprocessed_names_deleted/', methods=['POST'])
@login_required
def preprocessed_names_deleted(dataset):    
    datasetName = request.form.get('newdataset')
    mode_preprocessing = request.form.get('radio_preprocessing')
    names_list = request.form.getlist('names_rows')    
    
    #====== create log/information about exclusions (or keep) =====
    if names_list:
        textfile = open(os.path.join(ds_folder+"datasets_temp", "log-rows_file-"+datasetName+".txt"), "w")
        for element in names_list:
            textfile.write(element + "\n")
        textfile.close()
        data = ""
        if names_list:  
            with open(os.path.join(ds_folder+"datasets_temp", "log-rows_file-"+datasetName+".txt")) as fp:
                data = fp.read()        
        if os.path.exists(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt")):
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'a') as fp:
                fp.write(data)
        else:
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'w') as fp:
                fp.write(data)
        
    if dataset == 'log_test':        
        columns = loadColumns(dataset)    
        df = loadDataset(dataset)
    
    else:#se log uploaded        
        columns = loadColumns_users(dataset)
        df = loadDataset_users(dataset)
    #===============================================

    filename = dataset + '_'
        
    #listOfPos = list()
    if mode_preprocessing == 'radio_keep':
        listOfPos = list()
        if len(names_list) > 0:
            df_rows = pd.DataFrame(columns=columns)     
            #listOfPos = list()
            resultDict = {}
            
            # Iterate over the list of elements one by one
            for elem in names_list:        
                # Check if the element exists in dataframe values
                if elem in df.values:
                    listOfPositions = getIndexes(df, elem, listOfPos)
                    
            for i in range(len(listOfPositions)):
                if listOfPositions[i][0] in df.index:                
                    df_rows = df_rows.append(df.iloc[listOfPositions[i][0]], ignore_index=True)
            df = df_rows
    elif mode_preprocessing == 'radio_delete':
        listOfPos = list()  
        if len(names_list) > 0:            
            #listOfPos = list()                
            resultDict = {}
            # Iterate over the list of elements one by one
            
            for elem in names_list:        
                if elem in df.values:
                    listOfPositions = getIndexes(df, elem, listOfPos)
                    
            for i in range(len(listOfPositions)):
                if listOfPositions[i][0] in df.index:
                    df = df.drop(listOfPositions[i][0])
                    #print('Position ', i, ' (Row index , Column Name) : ', listOfPositions[i])

    filename += str(datasetName) + '.csv'    
    df.to_csv(os.path.join('./datasets_temp/'+current_user.username, filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])

@app.route('/datasets/<dataset>/preprocessed_elements_deleted/', methods=['POST'])
@login_required
def preprocessed_elements_deleted(dataset):    
    datasetName = request.form.get('newdataset')
    mode_preprocessing = request.form.get('radio_preprocessing')
    manualRows = request.form.getlist('manual_rows')
    
    #====== create log/information about exclusions (or keep) =====
    if manualRows:
        textfile = open(os.path.join(ds_folder+"datasets_temp", "log-rows_file-"+datasetName+".txt"), "w")
        for element in manualRows:
            textfile.write(element + "\n")
        textfile.close()
        data = ""
        if manualRows:  
            with open(os.path.join(ds_folder+"datasets_temp", "log-rows_file-"+datasetName+".txt")) as fp:
                data = fp.read()        
        if os.path.exists(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt")):
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'a') as fp:
                fp.write(data)
        else:
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'w') as fp:
                fp.write(data)
        
    if dataset == 'log_test':        
        columns = loadColumns(dataset)    
        df = loadDataset(dataset)
    
    else:#se log uploaded        
        columns = loadColumns_users(dataset)
        df = loadDataset_users(dataset)
    #===============================================

    filename = dataset + '_'
        
    #listOfPos = list()
    if mode_preprocessing == 'radio_keep':
        listOfPos = list()
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
        listOfPos = list()  
        if len(manualRows) > 0:            
            #listOfPos = list()                
            resultDict = {}
            # Iterate over the list of elements one by one
            
            for elem in manualRows:        
                if elem in df.values:
                    listOfPositions = getIndexes(df, elem, listOfPos)
                    
            for i in range(len(listOfPositions)):
                if listOfPositions[i][0] in df.index:
                    df = df.drop(listOfPositions[i][0])
                    #print('Position ', i, ' (Row index , Column Name) : ', listOfPositions[i])

    filename += str(datasetName) + '.csv'
    #if dataset == 'log_test':
    #    df.to_csv(os.path.join(ds_folder+"datasets", filename), index=False)        
    #else:
    df.to_csv(os.path.join('./datasets_temp/'+current_user.username, filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])

@app.route('/datasets/<dataset>/preprocessed_anonymize/', methods=['POST'])
@login_required
def preprocessed_anonymize(dataset):    
    datasetName = request.form.get('newdataset')    
    anonymize_student = request.form.get('anonymize_names')
    
    if dataset == 'log_test':
        columns = loadColumns(dataset)    
        df = loadDataset(dataset)
    else:
        columns = loadColumns_users(dataset)    
        df = loadDataset_users(dataset)

    filename = dataset + '_'
    
    #===== Anonymize ==========================#    
    if anonymize != '':
        df_aux = pd.DataFrame(columns=columns)        
        names = df[anonymize_student].unique()
        df_aux = df
        len_names = len(names)
        n = 1        
        for key in names:
            df_aux[anonymize_student] = df[anonymize_student].replace([key],'Anonymous'+str(n))
            n = n + 1
        df = df_aux
    #===========================================#
    
    filename += str(datasetName) + '.csv'    
    df.to_csv(os.path.join('./datasets_temp/'+current_user.username, filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])

@app.route('/datasets/<dataset>/preprocessed_inverse_order/', methods=['POST'])
@login_required
def preprocessed_inverse_order(dataset):    
    datasetName = request.form.get('newdataset')

    if dataset == 'log_test':
        columns = loadColumns(dataset)    
        df = loadDataset(dataset)
    else:
        columns = loadColumns_users(dataset)    
        df = loadDataset_users(dataset)

    filename = dataset + '_'
    
    #==== update df reverse order - LMS-Moodle =====#
    #if sort_log == 'radio_sort_yes':        
    df_aux = pd.DataFrame(columns=columns)
    df_aux = df.iloc[::-1]
    df = df_aux
    #================================================
    
    filename += str(datasetName) + '.csv'    
    df.to_csv(os.path.join('./datasets_temp/'+current_user.username, filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])

@app.route('/datasets/<dataset>/preprocessed_dataset/', methods=['POST'])
@login_required
def preprocessed_dataset(dataset):    
    manualFeatures = request.form.getlist('manualfeatures')
    datasetName = request.form.get('newdataset')
    manualRows = request.form.getlist('manual_rows')
    mode_preprocessing = request.form.get('radio_preprocessing')
    sort_log = request.form.get('radio_sort')
    anonymize_log = request.form.get('radio_anonymize')
    anonymize_student = request.form.get('anonymize_names')
    
    #====== create log/information about exclusions (or keep) =====
    if dataset == 'log_test':
        if manualRows:
            textfile = open(os.path.join(ds_folder+"datasets", "log-rows_file-"+datasetName+".txt"), "w")
            for element in manualRows:
                textfile.write(element + "\n")
            textfile.close()
        if manualFeatures:
            textfile = open(os.path.join(ds_folder+"datasets", "log-columns_file-"+datasetName+".txt"), "w")
            for element in manualFeatures:
                textfile.write(element + "\n")
            textfile.close()    
        
        data = data2 = ""
        if manualRows:  
            with open(os.path.join(ds_folder+"datasets", "log-rows_file-"+datasetName+".txt")) as fp:
                data2 = fp.read()
        if manualFeatures:
            with open(os.path.join(ds_folder+"datasets", "log-columns_file-"+datasetName+".txt")) as fp:
                data = fp.read()
        data += "\n"
        data += data2
        if os.path.exists(os.path.join(ds_folder+"datasets", "log-joined_file.txt")):
            with open(os.path.join(ds_folder+"datasets", "log-joined_file.txt"), 'a') as fp:
                fp.write(data)
        else:
            with open(os.path.join(ds_folder+"datasets", "log-joined_file.txt"), 'w') as fp:
                fp.write(data)

        #i_date = request.form.get('start_date')
        #f_date = request.form.get('end_date')

        columns = loadColumns(dataset)    
        df = loadDataset(dataset)
    else:#se log uploaded
        if manualRows:
            textfile = open(os.path.join(ds_folder+"datasets_temp", "log-rows_file-"+datasetName+".txt"), "w")
            for element in manualRows:
                textfile.write(element + "\n")
            textfile.close()
        if manualFeatures:
            textfile = open(os.path.join(ds_folder+"datasets_temp", "log-columns_file-"+datasetName+".txt"), "w")
            for element in manualFeatures:
                textfile.write(element + "\n")
            textfile.close()    
        
        data = data2 = ""
        if manualRows:  
            with open(os.path.join(ds_folder+"datasets_temp", "log-rows_file-"+datasetName+".txt")) as fp:
                data2 = fp.read()
        if manualFeatures:
            with open(os.path.join(ds_folder+"datasets_temp", "log-columns_file-"+datasetName+".txt")) as fp:
                data = fp.read()
        data += "\n"
        data += data2
        if os.path.exists(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt")):
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'a') as fp:
                fp.write(data)
        else:
            with open(os.path.join(ds_folder+"datasets_temp", "log-joined_file.txt"), 'w') as fp:
                fp.write(data)

        #i_date = request.form.get('start_date')
        #f_date = request.form.get('end_date')

        columns = loadColumns_users(dataset)    
        df = loadDataset_users(dataset)
    #===============================================

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
    #===== Anonymize ==========================#
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
    #===========================================#

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
            for th in columns: #verificar se a coluna é Hora ou Timestamp (guambiarra a corrigir)
                if th == 'Hora':
                    th_name = 'Hora'
                if th == 'Timestamp':
                    th_name = 'Timestamp'
            print(th_name)

            for elem in manualRows:        
                #verificar se é data
                cont_bar = 0
                for bar in elem:
                    if bar == '/':
                        cont_bar += 1
                # Check if the element exists in dataframe values
                index_end = len(elem.split(" ")[0])                
                if cont_bar == 2:
                    for i in range(len(df)):
                        df_date = str(df[th_name].loc[i]) #Hora ou Timestamp                       
                        if elem == df_date[0:index_end]:
                            #print(df_date)
                            #print(df_date[0:index_end])
                            listOfPositions = getIndexes(df, df_date, listOfPos)    
                else:    
                    if elem in df.values:
                        listOfPositions = getIndexes(df, elem, listOfPos)
                    
            for i in range(len(listOfPositions)):
                if listOfPositions[i][0] in df.index:
                    df = df.drop(listOfPositions[i][0])
                    #print('Position ', i, ' (Row index , Column Name) : ', listOfPositions[i])
    
    '''
    if i_date and f_date:
        df_timestamp =  df        
        #i_date = i_date+" 00:00"
        #f_date = f_date+" 23:59"
        i_date = datetime.strptime(i_date, '%Y-%m-%d')
        f_date = datetime.strptime(f_date, '%Y-%m-%d')
        i_d = i_date.strftime("%d")+"/"+i_date.strftime("%m")+"/"+i_date.strftime("%Y")+" 00:00"
        f_d = f_date.strftime("%d")+"/"+f_date.strftime("%m")+"/"+f_date.strftime("%Y")+" 23:59"
        print(i_d)
        print(f_d)
        #i_d = datetime.strptime(i_d, '%d/%m/%Y %H:%M')
        #f_d = datetime.strptime(f_d, '%d/%m/%Y %H:%M')
        #print(i_d)
        #print(f_d)
        df_timestamp['Timestamp'] = pd.to_datetime(df_timestamp['Timestamp'], format='%d/%m/%Y %H:%M')
        print(df_timestamp[(df_timestamp['Timestamp'] > i_d) & (df_timestamp['Timestamp'] < f_d)])
    '''
    
    filename += str(datasetName) + '.csv'
    if dataset == 'log_test':
        df.to_csv(os.path.join(ds_folder+"datasets", filename), index=False)        
    else:
        df.to_csv(os.path.join('./datasets_temp/'+current_user.username, filename), index=False)

    #df.to_csv(os.path.join(ds_folder+"datasets", filename), index=False)
    #return render_template('list_deleted.html', dataset = dataset, manualRows = manualRows)
    return redirect('/datasets/' + filename.split('.')[0])


@app.route('/get_csv/<dataset>')
@login_required
def get_csv(dataset):
    csv_dir  = "./datasets"
    csv_file = dataset
    csv_path = os.path.join(csv_dir, csv_file)
    return send_file(csv_path, as_attachment=True, attachment_filename=csv_file)

@app.route('/get_csv_user/<dataset>')
@login_required
def get_csv_user(dataset):
    csv_dir  = "./datasets_temp/"+current_user.username
    csv_file = dataset
    csv_path = os.path.join(csv_dir, csv_file)
    return send_file(csv_path, as_attachment=True, attachment_filename=csv_file)

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
