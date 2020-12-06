
from wwo_hist import retrieve_hist_data
import os
import pandas as pd

from datetime import datetime
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


from sklearn.metrics import mean_squared_error, r2_score
from numpy import sqrt



#Set directory for csv file 
os.chdir(r"C:\Users\Shefali Saini\Desktop\Spyder")

# parameters
frequency=1
start_date = '01-JAN-2019'
end_date = '31-DEC-2019'
api_key = '233329d0652145418c7150443200710'
location_list = ['delhi']


# To retrieve historical data
hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)


# To read csv file to create dataframe
d1 = pd.read_csv("delhi.csv") 
d2 = pd.read_csv("final_data.csv")

# To concate dataframes & to drop columns  
data=pd.concat([d2,d1],axis=1)  

# Dataframe for date_time column
d4 = pd.DataFrame(data['date_time'])


# To drop columns
data = data.drop(['date_time','totalSnow_cm','sunHour','uvIndex','moon_illumination','moonrise',
               'moonset','sunrise','sunset','FeelsLikeC','HeatIndexC','WindChillC',
               'WindGustKmph','humidity','maxtempC','mintempC','pressure','Unnamed: 18',
               'tempC','windspeedKmph','location','DewPointC','cloudcover','Surface Albedo','Dew Point',
               'Year','Minute','Clearsky DHI','Clearsky GHI','Clearsky DNI'],axis=1)


#data.to_csv('DATASET2.csv')

# Set negative values to 0, show sum negative values.
data[data < 0] = 0
data[data < 0].sum()

# check null values
data.isnull().sum()

# Data points with DNI > 1000. Recieved 0 rows 
d3 = data[data['DNI'] > 1000]

# Feature correlation for all features
plt.figure(figsize=(21, 13))
sns.heatmap(data.corr(), cmap='Greens', annot=True)
plt.title('Feature correlation',fontsize = 14, fontweight ='bold');

# Feature correlation for irradiance metrics
sns.heatmap(data.corr()[['GHI', 'DNI', 'DHI']], cmap='Greens', annot=True)
plt.title('GHI/DNI/DHI feature correlation',fontsize = 14, fontweight ='bold');

# Sum DNI/GHI/DHI by days of year
data.groupby(['Month','Day']).sum()[['GHI', 'DNI', 'DHI']].plot()
plt.title('Sum GHI/DNI/DHI by day of year',fontsize = 14, fontweight ='bold');
plt.ylabel('Irr W/M2')


# Mean DNI/GHI/DHI over time.
data.groupby('Hour').mean()[['GHI', 'DNI', 'DHI']].plot()
plt.title('Mean GHI/DNI/DHI by time of day',fontsize = 14, fontweight ='bold');
plt.ylabel('Irr W/M2')



# Scaling 
feat_count = len(data.columns)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
scaled = pd.DataFrame(scaled, index=data.index, columns=data.columns)

scaled = pd.concat([scaled,d4],axis=1)
pd.to_datetime(scaled['date_time'])


# Get features
features = ['Day','Hour','Temperature','DHI','DNI','GHI','Relative Humidity','Solar Zenith Angle',
            'Pressure','Wind Speed','precipMM','visibility','winddirDegree']
label = ['DNI']

#Create df of all lagged features

lags = [24, 168]
lstm_data = pd.DataFrame()
lagged_feat_dict = {}

for x in lags:
    
    # shift
    lagged_data = scaled[features].shift(periods=x).dropna() # drop after looping
    
    # rename lagged features
    lagged_col_names = {}
    for y in lagged_data.columns:
        lagged_col_names[y] = y + '_' + str(x)
    lagged_data.rename(columns=lagged_col_names, inplace=True)
    
    # make dict of lagged_col_names (to reference for later)
    lagged_feat_dict[x] = lagged_col_names.values()
        
    # make lstm_data the lagged data
    lstm_data = pd.concat([lstm_data, lagged_data], axis=1)
    
# Add DNI target
lstm_data = pd.concat([lstm_data, scaled[label]], axis=1)

lstm_data = pd.concat([lstm_data,d4],axis=1)

month = 7

tts = len(lstm_data.loc[pd.DatetimeIndex(lstm_data['date_time']).month < month])


model_results = []

epochs_ = 5
batch_size_ = 12
dropout_ = .3

lstm_data.set_index('date_time',drop = False,inplace = True)

for x in lags:
    
    print('Training w/ lag', x)
    
    feats = list(lagged_feat_dict[x])
    
    lstm_temp = lstm_data[feats + ['DNI','date_time']].dropna()
    lstm_train = lstm_temp.loc[pd.DatetimeIndex(lstm_temp['date_time']).month < month]
    lstm_test = lstm_temp.loc[pd.DatetimeIndex(lstm_temp['date_time']).month >= month]
    
    X_train = lstm_train[feats].values
    X_test = lstm_test[feats].values
    y_train = lstm_train['DNI'].values
    y_test = lstm_test['DNI'].values
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # design network                                                            
    model = Sequential()                                                        
    model.add(LSTM(int(x), input_shape=(X_train.shape[1], X_train.shape[2])))   
    model.add(Dropout(dropout_))                                                
    model.add(Dense(1))                                                         
    model.compile(loss='mean_squared_error', optimizer='adam')                  
                                                                                
    # fit network                                                               
    history = model.fit(X_train,                                                
                        y_train,                                                
                        epochs=epochs_,                                         
                        batch_size=batch_size_,                                 
                        validation_data=(X_test, y_test),                       
                        verbose=1,                                              
                        shuffle=False)     
    
    
   

   # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title('Training lag ' + str(x))
    plt.show()   
    
    # make a prediction
    yhat = model.predict(X_test)
    
    # sets floor for yhat
    yhat = [x[0] if x > 0 else 0 for x in yhat]
    
    # Concat y_hat to dataframe for later
    lstm_test = pd.concat([lstm_test, pd.DataFrame(yhat, columns=['yhat'], index=lstm_test.index)], axis=1)
    
    # Invert scaling (so we can get our forecast and evaluation in terms of DNI (W/m2))

    # Get yhat and y_test
    yhat = lstm_test['yhat']
    original = lstm_test['DNI']
    
    # Copy lstm_test
    lstm_test_yhat = lstm_test.iloc[:, :feat_count]
    lstm_test_original = lstm_test.iloc[:, :feat_count]

    # Substitute yhat and y_test into lstm_test
    lstm_test_yhat.iloc[:, 1] = yhat
    lstm_test_original.iloc[:, 1] = original

    # Inverse yhat
    inv_yhat = scaler.inverse_transform(lstm_test_yhat)
    inv_yhat = pd.DataFrame(inv_yhat).iloc[:, 1]

    # Inverse y_test
    inv_y_test = scaler.inverse_transform(lstm_test_original)
    inv_y_test = pd.DataFrame(inv_y_test).iloc[:, 1]
    
    hours = 300

    plt.plot(yhat[-hours:], label='yhat')
    plt.plot(original[-hours:], label='y test')
    plt.legend()
    plt.title('Y test vs. y hat, lag ' + str(x))
    plt.xlabel('Hour')
    plt.ylabel('DNI')
    
    plt.show()
    
    #To calculate RMSE
    rmse = sqrt(mean_squared_error(yhat, y_test))
    r2 = r2_score(yhat, y_test)
    
    # To print results
    print('R2 score: %.3f' % r2)
    print('RMSE: %.3f' % rmse)
    print('RMSE (in DNI (W/m2)): %.3f' % sqrt(mean_squared_error(inv_y_test, inv_yhat)))
    
    print()
    
    # Save all the results
    
    model_results_dict = {}
    
    
    model_results_dict['lag'] = x
    model_results_dict['dropout'] = dropout_
    model_results_dict['epochs'] = epochs_
    model_results_dict['batch_size'] = batch_size_
    model_results_dict['params'] = history.params
    model_results_dict['loss'] = history.history
    model_results_dict['rmse'] = rmse
    model_results_dict['dni_rmse'] = sqrt(mean_squared_error(inv_y_test, inv_yhat))
    model_results_dict['r2'] = r2
    model_results_dict['time_ran'] = int(time.time())
    
    model_results.append(model_results_dict)



