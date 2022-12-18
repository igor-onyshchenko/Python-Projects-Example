# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:51:13 2017

@author: igoro
"""

from subprocess import Popen
import pandas as pd
import copy
import pandasql as sql
import pickle
import cx_Oracle
from sqlalchemy import create_engine
import configparser
import datetime
import time
import sys
import smtplib
from email.mime.text import MIMEText
import logging 


def send_email(server_addr_port, fromaddr, to_addr_list, subject, text):
    try:
        # Prepare actual message
        msg = MIMEText("; \n".join(text))
        message = """From: %s\nTo: %s\nSubject: %s\n\n%s
        """ % (fromaddr, ", ".join(to_addr_list), subject, msg)
        
        #connection to server
        server = smtplib.SMTP(server_addr_port)
        
        #extended HELLO command to server to enable sending mails
        server.ehlo()
        
        #Transport Layer Security - start protected channel for mailing
        #server.starttls()
        
        #logging in / authorization 
        #server.login(username,password)
        
        #sending mails
        server.sendmail(fromaddr, to_addr_list, message)
        
        #close server
        server.quit()
        print('E-mail successfully sent.')
    except Exception as e:
        print(str(e))



print('START', datetime.datetime.now())
#========================================================================

# create a configparser
try:
    config = configparser.ConfigParser()
    #config.read(sys.argv[1])
    config.read('C:/BI_work/WH_Churn_Model/config_WH_churn.ini') #!!!!!! Change initial directory here
except:
    print('READING INI-FILE FAILED')

# logging
try:
    logger = logging.getLogger('WH_CHURN_LOG')
    hdlr = logging.FileHandler(config['LOG']['LOG_FILE'])
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)  
    error_text = list()
except:
    print('CREATING LOG FAILED')

logger.info(' logging: success')

# connecting to Oracle
try:
    login = config['DB']['USER'] #state login here
    password = config['DB']['PASSWORD'] #state the password here
    TNS = config['DB']['NAME'] #state TNS here
    ip = config['DB']['IP']
    port = config['DB']['PORT']
    db_name = config['DB']['NAME']
    #con = cx_Oracle.connect(login + '/' + password + '@' +TNS)
    dsnStr = cx_Oracle.makedsn(ip, port, db_name)
    con = cx_Oracle.connect(user=login, password=password, dsn=dsnStr)    
    cur = con.cursor()
    logger.info(' connecting to Oracle: success')
except Exception as e:
    logger.error(' connecting to Oracle: ' + str(e))
    error_text.append(' connecting to Oracle: ' + str(e))


#checking if all the tables in Oracle are ready
i = 0
while True:
    i = i + 1
    try:
        print('TABLES REDINESS CHECK', datetime.datetime.now())
        logger.info('TABLES REDINESS CHECK')
        TABLES_REDINESS_CHECK = pd.read_sql("""
        select t1.ims_source_id
        from 
        (
                select *
                from playtech_dwh.dwh_fact_ims_daily
                where ims_source_id = 18
                and stats_date = trunc(sysdate) - 1
                                      and rownum = 1
                                      ) t1
                inner join 
                (
                        select *
                        from playtech_dwh.dwh_fact_casino_sessions
                        where ims_source_id = 18
                        and game_date_id = trunc(sysdate) - 1
                                                and rownum = 1
                                                ) t2 on t1.ims_source_id = t2.ims_source_id
                        inner join 
                        (
                                select *
                                from playtech_dwh.dwh_fact_casinos
                                where ims_source_id = 18
                                and end_date_id = trunc(sysdate) - 1
                                                       and rownum = 1
                                                       ) t3 on t1.ims_source_id = t3.ims_source_id""", con)
        if TABLES_REDINESS_CHECK.shape[0] > 0:
            logger.info(' TABLES REDINESS CHECK: success')
            break
        else:
            time.sleep(1800) #wait for 30 minutes and then check again
    except Exception as e:
        logger.error(' TABLES REDINESS CHECK: ' + str(e))
        error_text.append(' TABLES REDINESS CHECK: ' + str(e))
    if i >= 24:
        logger.error(' Oracle tables have not been updated ')
        error_text.append(' Oracle tables have not been updated ')
        break #script is trying to run for 24 hours; most probably there was a problem with database update
    
# preparing panel
try:
    print('MAKING PANEL', datetime.datetime.now())
    cur.execute(open(config['SQL']['SQL_SCRIPT']).read())
    logger.info(' preparing panel: success')
except Exception as e:
    logger.error(' preparing panel: ' + str(e))
    error_text.append(' preparing panel: ' + str(e))

# data to score
try:
    print('GETTING DATA TO SCORE', datetime.datetime.now())
    data_to_score = pd.read_sql("""
                                select c.* 
                                  from casino_churn_panel c left join casino_churn_daily f on c.IMS_SOURCE_ID = f.IMS_SOURCE_ID and c.PLAYER_ID = f.PLAYER_ID and c.SNAPSHOT_POINT = f.SNAPSHOT_POINT
                                 where f.CHURN_PROB is NULL
                                       and c.ims_source_id = 18 and c.brand_id = 108
                                """, con)
    logger.info(' data to score: success')
except Exception as e:
    logger.error(' data to score: ' + str(e))
    error_text.append(' data to score: ' + str(e))

################### ONETIMERS

# import model
try:
    print('IMPORTING MODEL ONETIMERS', datetime.datetime.now())
    model = pickle.load(open(config['MODEL']['CURRENT_MODEL_ONETIMERS'], 'rb'))
    logger.info(' import model ONETIMERS: success')
except Exception as e:
    logger.error(' import model ONETIMERS: ' + str(e))
    error_text.append(' import model ONETIMERS: ' + str(e))
    
# score data
try:
    print('SCORING ONETIMERS', datetime.datetime.now())
    data = copy.deepcopy(data_to_score)
    #====== categorial variables to WoE
    CV = pd.read_csv(config['MODEL']['CATEGORIAL_VARS_ONETIMERS'])
    data = sql.sqldf("""
                     select d.*,
                            c.PLATFORM_WOE
                       from data d left join CV c on d.PLATFORM = c.PLATFORM
                     """)
    data['SNAPSHOT_POINT'] = data_to_score['SNAPSHOT_POINT']
    data = data[data['SENIORITY'] == 1]
    
    data = data.fillna(-1)
    Vars_to_Drop = ['CHURN_PROB', 'SNAPSHOT_POINT', 'IMS_SOURCE_ID', 'BRAND_ID', 'PLAYER_ID','CHURN10D', 'PLATFORM_CODE', 'LAST_BET_DATE', 'CASINO_NGR_LAST14_RT', 'CASINO_NGR_LAST30_RT', 'CASINO_NGR_LAST60_RT', 'CASINO_NGR_LAST7_RT', 'CASINO_NGR_LAST_DAY_RT', 'PLATFORM', 'USERNAME']
    model_predicted = model.predict_proba(data.drop(Vars_to_Drop, 1))[:,1]
    #da = AL.data_audit_cust(data.drop(Vars_to_Drop, 1))
    data['CHURN_PROB'] = model_predicted
    data = data[['SNAPSHOT_POINT', 'IMS_SOURCE_ID', 'PLAYER_ID', 'CHURN_PROB', 'BRAND_ID', 'USERNAME']]
    data['UPDATE_DATE'] = datetime.datetime.now()#str(datetime.datetime.now())
    logger.info(' score data ONETIMERS: success')
except Exception as e:
    logger.error(' score data ONETIMERS: ' + str(e))
    error_text.append(' score data ONETIMERS: ' + str(e))

# upload data
try:
    print('UPLOADING DATA ONETIMERS', datetime.datetime.now())
    engine = create_engine(config['DB']['ENGINE'], echo=True) #echo = True - if you want to see the process in python colsole
    data.to_sql('casino_churn_daily', 
                engine, 
                if_exists = 'append', 
                index=False,
                flavor = 'oracle'
                )
    cur.execute("""commit""")
    logger.info(' upload data ONETIMERS: success')
except Exception as e:
    logger.error(' upload data ONETIMERS: ' + str(e))
    error_text.append(' upload data ONETIMERS: ' + str(e))
###################
    
################### NON-ONETIMERS

# import model
try:
    print('IMPORTING MODEL NON-ONETIMERS', datetime.datetime.now())
    model = pickle.load(open(config['MODEL']['CURRENT_MODEL_NONONETIMERS'], 'rb'))
    logger.info(' import model NON-ONETIMERS: success')
except Exception as e:
    logger.error(' import model NON-ONETIMERS: ' + str(e))
    error_text.append(' import model NON-ONETIMERS: ' + str(e))
    
# score data
try:
    print('SCORING', datetime.datetime.now())
    data = copy.deepcopy(data_to_score)
    #====== categorial variables to WoE
    CV = pd.read_csv(config['MODEL']['CATEGORIAL_VARS_NONONETIMERS'])
    data = sql.sqldf("""
                     select d.*,
                            c.PLATFORM_WOE
                       from data d left join CV c on d.PLATFORM = c.PLATFORM
                     """)
    data['SNAPSHOT_POINT'] = data_to_score['SNAPSHOT_POINT']
    data = data[data['SENIORITY'] > 1]
    
    data = data.fillna(-1)
    Vars_to_Drop = ['CHURN_PROB', 'SNAPSHOT_POINT', 'IMS_SOURCE_ID', 'BRAND_ID', 'PLAYER_ID','CHURN10D', 'PLATFORM_CODE', 'LAST_BET_DATE', 'CASINO_NGR_LAST14_RT', 'CASINO_NGR_LAST30_RT', 'CASINO_NGR_LAST60_RT', 'CASINO_NGR_LAST7_RT', 'CASINO_NGR_LAST_DAY_RT', 'PLATFORM', 'USERNAME']
    model_predicted = model.predict_proba(data.drop(Vars_to_Drop, 1))[:,1]    
    #da = AL.data_audit_cust(data.drop(Vars_to_Drop, 1))
    data['CHURN_PROB'] = model_predicted
    data = data[['SNAPSHOT_POINT', 'IMS_SOURCE_ID', 'PLAYER_ID', 'CHURN_PROB', 'BRAND_ID', 'USERNAME']]
    data['UPDATE_DATE'] = datetime.datetime.now()#str(datetime.datetime.now())
    logger.info(' score data NON-ONETIMERS: success')
except Exception as e:
    logger.error(' score data NON-ONETIMERS: ' + str(e))
    error_text.append(' score data NON-ONETIMERS: ' + str(e))

# upload data
try:
    print('UPLOADING DATA NON-ONETIMERS', datetime.datetime.now())
    engine = create_engine(config['DB']['ENGINE'], echo=True) #echo = True - if you want to see the process in python colsole
    data.to_sql('casino_churn_daily', 
                engine, 
                if_exists = 'append', 
                index=False,
                flavor = 'oracle'
                )
    cur.execute("""commit""")
    logger.info(' upload data NON-ONETIMERS: success')
except Exception as e:
    logger.error(' upload data NON-ONETIMERS: ' + str(e))
    error_text.append(' upload data NON-ONETIMERS: ' + str(e))
###################

# close connection
try:
    con.close()
    logger.info(' close connection: success')
except Exception as e:
    logger.error(' close connection: ' + str(e))
    error_text.append(' close connection: ' + str(e))
    
#launching IMS update
try:
    if len(error_text)>0:
        print('There were errors in scoring process; IMS update denied')
        logger.error(' There were errors in scoring process; IMS update denied ')
        error_text.append(' There were errors in scoring process; IMS update denied ')
    else:
        try:
            Popen("C:/scripts/Churn/who_integrated/bin/start_who.bat")
        except Exception as e:
            print('IMS update failed')
            logger.error(str(e))
            error_text.append(' Opening .bat file for IMS update failed ')
except Exception as e:
    print('IMS Update failed')
    logger.error(str(e)) 
    error_text.append(' launching IMS update: ' + str(e))    

# Sending Mail
try:
    if len(error_text)>0:
        result = 'WH Churn - failure'
        text = error_text
    else:
        result = 'WH Churn - success'
        text = ['Daily WH churn scoring finished']
        
    send_email('mail.playtech.corp','cugp@playtech.com', ['Artem.Lytvyn@playtech.com', 'Sergey.Goichman@playtech.com'], result, text)
            
    logger.info('Email conposed and sent successfully')
except Exception as e:
    logger.error('Email sending fail!')
    logger.error( str(e))

# close log
try:
    logging.shutdown()
except:
    print('LOGGING CLOSURE HAS FAILED')

print('FINISH', datetime.datetime.now())
# data.dtypes                          
# data['SNAPSHOT_POINT'].unique()                           
# data_to_score.dtypes                           
                           
                           
                           

