import os.path
basedir = os.path.abspath(os.path.dirname(__file__))

#ds_folder = os.path.dirname(os.path.abspath(__file__))
#my_file = os.path.join(THIS_FOLDER, 'myfile.txt')

DEBUG = True

SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'storage.db')
SQLALCHEMY_TRACK_MODIFICATIONS = True

SEND_FILE_MAX_AGE_DEFAULT = 1 #cache

SECRET_KEY = 'emr'
