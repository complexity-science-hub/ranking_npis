import os
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.inspection import permutation_importance
import multiprocessing as mp

"Author: N Haug, 2020"

def feature_ranking(permutation_importance,sd=np.inf):

    permutation_importance = permutation_importance.drop("Rank",axis=1).reset_index()[["Measure","mean_Delta","std_Delta"]]
    timeless_ranking       = permutation_importance.sort_values(["Measure","mean_Delta"],ascending=False).groupby("Measure").head(1).sort_values('mean_Delta',ascending=False)
    timeless_ranking['Rank'] = np.arange(1,timeless_ranking.shape[0]+1)
    timeless_ranking = timeless_ranking.set_index("Rank")
    timeless_ranking['CI'] = norm.ppf(0.975)*timeless_ranking['std_Delta']

    if sd<np.inf:

        timeless_ranking[['mean_Delta',"CI"]] = timeless_ranking[['mean_Delta',"CI"]].applymap(lambda x: RFA.round2msd(x,sd)).sort_values("Measure")
    
    timeless_ranking = timeless_ranking.sort_values("Measure")

    return timeless_ranking

def compute_performance_metrics(s,m,d,n,n_estimators,maxsamples,nsplits,X,y,enddate,countries):

    X,y = shift_outcome(X,y,s,shift_date="y")
    X,y = cut_dates(X,y,enddate)

    kf = KFold(n_splits=nsplits)

    results = pd.DataFrame()
                
    for train_index, test_index in kf.split(countries):
        
        X_train = X.loc[(countries[train_index]),:]
        y_train = y.loc[(countries[train_index]),:]
        
        X_test = X.loc[(countries[test_index]),:]
        y_test = y.loc[(countries[test_index]),:]
        
        if len(y_test)==0:
            continue

        forest = RandomForestRegressor(
            min_samples_leaf=m,
            max_depth=d,
            max_features=n,
            n_estimators=n_estimators,
            max_samples=maxsamples,
            bootstrap=True).fit(X_train.values,y_train.values.ravel())

        # compute several metrics for the goodness of the fit
        R2_train  = forest.score(X_train.values,y_train.values.ravel())
        MSE_train = mean_squared_error(forest.predict(X_train.values),y_train.values.ravel())
        RSS_train = np.sum((forest.predict(X_train.values)-y_train.values.ravel())**2)

        R2_test  = forest.score(X_test.values,y_test.values.ravel())
        MSE_test = mean_squared_error(forest.predict(X_test.values),y_test.values.ravel())
        RSS_test = np.sum((forest.predict(X_test.values)-y_test.values.ravel())**2)

        results = results.append(pd.DataFrame(
                {'timeshift':           [s],
                'min_samples_leaf':     [m],
                'max_tree_depth':       [d],
                'max_features':         [n],
                'R2_test':              [R2_test],
                'MSE_test':             [MSE_test],
                'RSS_test':             [RSS_test],
                'n_datapoints_test':    [len(y_test)],
                'R2_train':             [R2_train],
                'MSE_train':            [MSE_train],
                'RSS_train':            [RSS_train],
                'n_datapoints_train':   [len(y_train)]
            }),ignore_index=False)

    return results

def round2msd(x,k):
    if x==0:
        return int(0)
    elif abs(x)>=1:
        return int(np.round(x))
    else:
        nks = int(-np.floor(np.log10(abs(x))))
        return np.around(x,nks+k-1)
    
def shift_outcome(X,y,s,shift_date="y"):

    # If shift_date="y", then the date of y is shifted s days into the past. If shift_date="X", then the date of X is shifted s days into the future.

        if shift_date=="y":
    
            y_shifted = y.reset_index().copy()
            y_shifted["Date"] = y_shifted[["Date"]].applymap(lambda x: x.date()-timedelta(s))
            y_shifted = y_shifted.set_index(["Country","Date"])
            y_shifted.drop(y_shifted.loc[np.isnan(y_shifted.values)].index,inplace=True)
            data = X.join(y_shifted,how="inner")
            X = data[X.columns]
            y_shifted = data[y.columns]
            
            return X,y_shifted
        
        if shift_date=="X":
        
            X_shifted = X.reset_index().copy()
            X_shifted["Date"] = X_shifted[["Date"]].applymap(lambda x: x.date()+timedelta(s))
            X_shifted = X_shifted.set_index(["Country","Date"])
            y.drop(y.loc[np.isnan(y.values)].index,inplace=True)
            data = X_shifted.join(y,how="inner")
            X_shifted = data[X.columns]
            y = data[y.columns]
            
            return X_shifted,y
        
def cut_dates(X,y,enddate):
        
        X = X.reset_index()
        y = y.reset_index()

        X["Date"] = X["Date"].apply(lambda x: x.date())
        y["Date"] = y["Date"].apply(lambda x: x.date())

        if type(enddate) is date:
        
            X = X.loc[X["Date"].apply(lambda x: x<=enddate)].set_index(["Country","Date"])
            y = y.loc[y["Date"].apply(lambda x: x<=enddate)].set_index(["Country","Date"])

        elif type(enddate) is pd.DataFrame:

            X = X.join(enddate,on='Country').sort_values(["Country","Date"])
            y = y.join(enddate,on='Country').sort_values(["Country","Date"])

            X = X.loc[X.Date<X.enddate].set_index(["Country","Date"])
            y = y.loc[y.Date<y.enddate].set_index(["Country","Date"])

            X.drop('enddate',axis=1,inplace=True)
            y.drop('enddate',axis=1,inplace=True)

        return X,y        

class RandomForestAnalysis(object):
    """Class for random forest analysis of COVID-19 NPIs"""
    def __init__(self,**kwargs):
        
        self._nsplits            = kwargs.get('n_splits',             10                 )
        self._min_countries      = kwargs.get('min_n_countries',      5                  )
        self._excludecountries   = kwargs.get('exclude_countries',    []                 )
        self._excludemeasures    = kwargs.get('exclude_measures',     []                 )
        self._nminconfirmed      = kwargs.get('n_min_confirmed',      0                  )
        self._startdate          = kwargs.get('startdate',            date(2019,12,1)    )  
        self._enddate            = kwargs.get('enddate',              date.today()       )
        self._timeshift          = kwargs.get('timeshift',            [0]                )
        self._minsamplesleaf     = kwargs.get('minsamples_leaf',      [1]                )
        self._maxtreedepth       = kwargs.get('max_tree_depth',       [None]             )
        self._n_estimators       = kwargs.get('n_estimators',         100                )
        self._maxfeatures        = kwargs.get('max_features',         [1/3]              )
        self._maxsamples         = kwargs.get('max_samples',          3/4                )
        self._outcomename        = kwargs.get('outcome_name',         "Outcome"          )
        self._data_path          = kwargs.get('data_path',            '../../merged_data/COVID19_data_cumulative_PAPER_VERSION.csv')
        self.data = pd.read_csv(self._data_path,sep=';',parse_dates=['Date'],index_col=["Country","Date"])
                 
    def reduce_data(self):
    
        data = self.data.copy().reset_index()
        measurenames = self._measurenames
        data = data.loc[np.logical_not(data.Country.isin(self._excludecountries))]
        data = data.iloc[:,np.logical_not(np.in1d(np.array(data.columns),np.array(self._excludemeasures)))]
        data["Date"] = data["Date"].apply(lambda x: x.date())

        if type(self._enddate) is date:
            N_countries = pd.DataFrame(data.drop(data.loc[data.Date>=self._enddate].index).sort_values('Date').drop_duplicates('Country',keep='last')[        measurenames].aggregate('sum')).rename(columns={0:'N_countries'})
            exclude = list(N_countries.loc[lambda x: x.N_countries<self._min_countries].index)
            data.drop(exclude,axis=1,inplace=True)
            del exclude

        else: 
            exclude = []
            print('no features removed')

        data.loc[data["Confirmed"]<self._nminconfirmed,self._outcomename] = np.nan
        self._measurenames = list((set(measurenames).difference(set(exclude))).difference(self._excludemeasures))
        self._measurenames.sort()
        self.data = data.set_index(["Country","Date"])
        
    def get_predictors(self):
    
        self.predictors = self.data[self._measurenames]
    
    def get_outcome(self):
    
        self.outcome = pd.DataFrame(self.data[self._outcomename],columns=[self._outcomename])
        
    def crossvalidate(self,**kwargs):

        n_processes  = kwargs.get('n_processes',1)
        enddate      = self._enddate
        X            = self.predictors
        y            = self.outcome
        nsplits      = self._nsplits
        maxsamples   = self._maxsamples
        n_estimators = self._n_estimators

        countries = np.array(self.predictors.index.unique("Country"))
        np.random.shuffle(countries)
    
        results = pd.DataFrame()

        t = self._timeshift

        for p in itertools.product(self._minsamplesleaf,self._maxtreedepth,self._maxfeatures):
        
            print(p)
        
            m = p[0]
            d = p[1]
            n = p[2]        
            
            args = list(map(lambda s: (s,m,d,n,n_estimators,maxsamples,nsplits,X,y,enddate,countries),t))

            pool = mp.Pool(processes=n_processes)
            newres = pool.starmap(compute_performance_metrics,args)
            pool.close()
            pool.join()

            results = results.append(pd.concat(newres))

        self.cv_results = results
        
    def get_performance(self):
    
        performance = self.cv_results[["timeshift","min_samples_leaf","max_tree_depth","max_features","n_datapoints_test","R2_test","RSS_test","R2_train","RSS_train","n_datapoints_train","MSE_test","MSE_train"]].groupby(['timeshift','min_samples_leaf','max_tree_depth',"max_features"],as_index=False).agg(['mean','sum'])
        performance["RSS_test/n_datapoints_test"] = performance[("RSS_test",'sum')]/performance[("n_datapoints_test",'sum')]
        performance["RSS_train/n_datapoints_train"] = performance[("RSS_train",'sum')]/performance[("n_datapoints_train",'sum')]
        self.performance = performance
        
        
    def get_performance_table(self,parameter='min_samples_leaf',smoothed=True,window_size=50,stddev=10):
    
        performance = self.performance
        metric = performance.columns[0]
        performance = performance.reset_index()[["timeshift",parameter,metric]]
        performancetable = performance.set_index(['timeshift',parameter]).unstack(level="timeshift")
        performancetable.columns = performancetable.columns.droplevel(0)
        
        if smoothed == False:
            self.performancetable = performancetable
            
        else:
            self.performancetable = performancetable.rolling(window=window_size,win_type="gaussian",center=True,min_periods=1).mean(std=stddev)
       
    
    def get_optimal_parameters(self,metric,timeshift):

        if metric == 'R2_test':
            return self.performance.loc[(timeshift)][(metric,'mean')].idxmax()        

        elif metric == 'MSE_test':
            return self.performance.loc[(timeshift)][(metric,'mean')].idxmin()               

        elif metric == 'RSS_test':
            return self.performance.loc[(timeshift)][(metric,'sum')].idxmin()

        else:
            print("metric must be R2_test, MSE_test or RSS_test")
            return None
            
        
    def get_gini_importance(self,**kwargs):
        
        X = self.predictors
        y = self.outcome
        
        s                  = kwargs.get('time_shift',0)
        max_depth          = kwargs.get('max_depth',None)
        min_samples_leaf   = kwargs.get('min_samples_leaf',1)
        max_features       = kwargs.get('max_features',1)
        samplesize         = kwargs.get('n_samples',100)
        max_features       = kwargs.get('max_features',int(np.round(X.shape[1]/3)))
        
        X,y = shift_outcome(X,y,s,shift_date="y")
        X,y = cut_dates(X,y,self._enddate)
        
        measurenames = X.columns        

        forests = []

        for n in range(samplesize):

            forest = RandomForestRegressor(min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                           max_features=max_features).fit(X.values,y.values.ravel())
            forests = forests+[forest]
        
        std  = np.std(np.array([list(forest.feature_importances_) for forest in forests]),axis=0)
        mean = np.mean(np.array([list(forest.feature_importances_) for forest in forests]),axis=0)

        feature_importance = pd.DataFrame({'Measure':pd.Series(measurenames),
                                           'Importance':pd.Series(forest.feature_importances_)})
        feature_importance['Importance'] = mean
        feature_importance['$\sigma$'] = std
        feature_importance['Importance'] = feature_importance['Importance'].apply(lambda x: round2msd(x,2))
        feature_importance['95$\%$CI'] = feature_importance['$\sigma$'].apply(lambda x: round2msd(norm.ppf(0.975)*x,1))
        feature_importance = feature_importance.sort_values('Importance',ascending=False)
        feature_importance['Rank'] = np.arange(1,feature_importance.shape[0]+1)
        feature_importance = feature_importance.set_index('Rank')
        
        return feature_importance
        

    def plot_country_fits(self,**kwargs):

        s                  = kwargs.get('time_shift',0)
        d                  = kwargs.get('max_depth',None)
        m                  = kwargs.get('min_samples_leaf',1)
        n                  = kwargs.get('max_features',1/3)

        countries          = kwargs.get('countries',"all")
        
        X = self.predictors
        y = self.outcome
        
        X,y = shift_outcome(X,y,s,shift_date="y")
        X,y = cut_dates(X,y,self._enddate)

        forest = RandomForestRegressor(min_samples_leaf=m,max_depth=d,max_features=n).fit(X.values,y.values.ravel())
        
        y_predict = pd.DataFrame(forest.predict(X.values)).set_index(y.index)
        
        if countries == "all":

            countries = list(set(X.index.get_level_values(0)))
            
        countries.sort()       

        
        fig, axs = plt.subplots(int(np.ceil(len(countries)/2)), 2,figsize=(10,1.5*len(countries)))
        axs=axs.ravel()
        fig.tight_layout(pad=3.0)

        i = 0

        for country in countries:
            
            axs[i].plot(y.loc[(country)],label='observed',linewidth=3)
            axs[i].plot(y_predict.loc[(country)],'--',label='model s='+str(s))
                
            axs[i].set_title(country)
            axs[i].grid()

            axs[i].set_ylim(0.25,4.5)
            axs[i].set_xlim(date(2020,2,1),date(2020,4,30))
            i+=1

    def get_permutation_importance(self,**kwargs):
        
        s                  = kwargs.get('time_shift',0)
        d                  = kwargs.get('max_depth',None)
        m                  = kwargs.get('min_samples_leaf',1)
        n                  = kwargs.get('max_features',1/3)
        r                  = kwargs.get('max_samples',3/4)
        n_splits           = kwargs.get('n_splits',10)
        n_repeats          = kwargs.get('n_repeats',1)
        drop_countries     = kwargs.get('drop_countries',[])
               
        X = self.predictors.drop(drop_countries)
        y = self.outcome.drop(drop_countries)
        X,y = shift_outcome(X,y,s,shift_date="y")
        X,y = cut_dates(X,y,self._enddate)

        countries = np.array(X.index.unique("Country"))
        np.random.shuffle(countries)

        features = list(X.columns)

        forest = RandomForestRegressor(max_depth=d,max_features=n,min_samples_leaf=m,max_samples=r)

        permutation_scores = pd.DataFrame()

        for feature in features:

            Deltas = []

            kf = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=0)

            for train_index, test_index in kf.split(countries):

                X1 = X.copy()
                v = X1[feature].values
                np.random.shuffle(v)
                X1[feature] = v

                X_train = X.loc[(countries[train_index]),:]                               
                X_test = X.loc[(countries[test_index]),:]
                                
                X1_train = X1.loc[(countries[train_index]),:]
                y_train = y.loc[(countries[train_index]),:]
                
                X1_test = X1.loc[(countries[test_index]),:]
                y_test = y.loc[(countries[test_index]),:]      

                score = forest.fit(X1_train.values,y_train.values.ravel()).score(X1_test.values,y_test.values.ravel())
                baseline = forest.fit(X_train.values,y_train.values.ravel()).score(X_test.values,y_test.values.ravel())

                Deltas = Deltas + [baseline-score]

            mean = np.array(Deltas).mean()
            std = np.array(Deltas).std()/np.sqrt(n_repeats*n_splits)

            permutation_scores = permutation_scores.append(
                pd.DataFrame({
                    'Measure':[feature],
                    'Deltas':[np.array(Deltas)],
                    'mean_Delta': [mean],
                    'std_Delta':[std]
                }))

        permutation_scores = permutation_scores.sort_values('mean_Delta',ascending=False)        
        permutation_scores["Rank"] = np.arange(1,permutation_scores.shape[0]+1)
        permutation_scores = permutation_scores.set_index("Rank",drop=False)
        permutation_scores['timeshift'] = s
                              
        return permutation_scores

    def get_epidemic_age(self,**kwargs):

        mincount = kwargs.get('min_casecount',30)
        data = self.data.copy()
        epi_startdates = data.query("Confirmed>=@mincount").reset_index().drop_duplicates("Country",keep="first")[["Country","Date"]].set_index("Country")
        data = data.reset_index("Date").join(epi_startdates,how='left',rsuffix='_start')
        data['Epidemic age'] = data[['Date','Date_start']].apply(lambda x: (x.Date-x.Date_start).days,axis=1)
        
        return data[['Date','Epidemic age']].set_index("Date",append=True)

            
    def get_prediction(self,countries,**kwargs):

        s                  = kwargs.get('time_shift',0)
        d                  = kwargs.get('max_depth',None)
        m                  = kwargs.get('min_samples_leaf',1)
        n                  = kwargs.get('max_features',1/3)
        mincount           = kwargs.get('min_casecount',30)
        n_estimators       = kwargs.get('n_estimators',500)
        r                  = kwargs.get('max_samples',3/4)
        n_repeats          = kwargs.get('n_repeats',10)

        epi_age = self.get_epidemic_age(min_casecount=mincount).loc[(countries)]
        
        X = self.predictors.loc[(countries)].copy()
        y = self.outcome.loc[(countries)].copy()
        l
        X,y = shift_outcome(X,y,s,shift_date="X")
        X,y = cut_dates(X,y,self._enddate+timedelta(days=s))

        forest = RandomForestRegressor(min_samples_leaf=m,max_depth=d,max_features=n,n_estimators=n_estimators,max_samples=r)

        forests = [forest.fit(X.values,y.values.ravel()) for k in range(n_repeats)]
        
        predictions = np.concatenate(tuple([forest.predict(X.values).reshape(1,-1) for forest in forests]),axis=0)

        y_predict = predictions.mean(axis=0)
        std_y_predict = predictions.std(axis=0)
        
        results = y
        results['R_predict'] = y_predict
        results['Std(R)_predict'] = std_y_predict
        results = results.join(epi_age,how='right')
        results.index = results.index.droplevel("Date")
        results = results.set_index('Epidemic age',append=True)

        return results

def get_L1_ranking(ranking,scorename,variant="naive"):

    if variant == "naive":

        L1_ranking = ranking[[scorename,"L1"]].groupby("L1").sum().sort_values(scorename,ascending=True)
        L1_ranking["Rank"] = np.arange(1,L1_ranking.shape[0]+1)

        return L1_ranking

    elif variant == "top_5":

        L1_ranking = ranking[[scorename,"L1"]].sort_values(scorename,ascending=True).groupby("L1").head(5).groupby("L1").sum()
        L1_ranking = L1_ranking.sort_values(scorename)
        L1_ranking["Rank"] = np.arange(1,L1_ranking.shape[0]+1)

        return L1_ranking       

    elif variant == "rank_weighted":

        ranking = ranking.sort_values(scorename,ascending=True)
        ranking["Rank"] = np.arange(1,ranking.shape[0]+1)
        ranking[scorename+'/Rank'] = ranking[scorename]/ranking['Rank']
        L1_ranking = ranking[[scorename+'/Rank',"L1"]].groupby("L1").sum().sort_values(scorename+'/Rank',ascending=True)
        L1_ranking["Rank"] = np.arange(1,L1_ranking.shape[0]+1)
        
        return L1_ranking
