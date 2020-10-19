#basics
import numpy as np
import pandas as pd

import os
import itertools
import datetime
import time
import textwrap
import random

# plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf

# datawrappers
import coronadataclass as cdc
import measureclass as mc




class CrossValidation(object):
    def __init__(self, **kwargs):
        self.__MinCaseCount             = kwargs.get('MinCaseCount', 30) # start trajectories with at least 30 confirmed cases
        self.__MinMeasureCount          = kwargs.get('MinMeasureCount',5) # at least 5 countries have implemented measure
        self.__verbose                  = kwargs.get('verbose', True)
        self.__cvres_filename           = kwargs.get('CVResultsFilename', None)
        self.__external_observable_file = kwargs.get('ExternalObservableFile', None)
        self.__external_observable_info = {'Country':'Country','Date':'Date','Observable':'R','Cases':'Confirmed','dateformat':'%Y-%m-%d','date_offset':0,'readcsv':{'sep':';'},'dropna_subset': ['R']}
        self.__external_observable_info.update(kwargs.get('ExternalObservableInfo', {}))
        self.__external_indicators_file = kwargs.get('ExternalIndicatorsFile', None)
        self.__observable_name          = kwargs.get('ObservableName', 'Confirmed')
        self.__maxlen                   = kwargs.get('MaxObservableLength', None)
        self.__finaldate                = kwargs.get('FinalDate', None)
        self.__finaldatefile            = kwargs.get('FinalDateFile', None)
        self.__finaldatefile_dateformat = kwargs.get('FinalDateFileDateFormat','%d/%m/%Y')
        self.__finaldatefrommeasureDB   = kwargs.get('FinalDateFromDB', False)
        self.__extendfinaldateshiftdays = kwargs.get('FinalDateExtendWithShiftdays', False)
        self.__date_randomize           = kwargs.get('DateRandomize', None) # possible options: distribution (keep same distribution of implementation dates), random (flat distribution over full range)
        self.__remove_continent         = kwargs.get('RemoveContinent', None) # possible options: europe, asia, americas
        self.colornames                 = kwargs.get('ColorNames', None)
        
        
        # load data from DB files
        self.jhu_data          = cdc.CoronaData(**kwargs)
        self.measure_data      = mc.COVID19_measures(**kwargs)        
        self.measure_data.RemoveCountry('Diamond Princess')
        self.measure_data.RenameCountry('South Korea', 'Korea, South')
        self.measure_data.RenameCountry('Czech Republic', 'Czechia')
        self.measure_data.RenameCountry('Republic of Ireland', 'Ireland')
        self.measure_data.RenameCountry('Taiwan', 'Taiwan*')
        
        
        self.__UseExternalObs            = False
        if not self.__external_observable_file is None:
            self.__UseExternalObs        = True
            self.__ExternalObservables   = pd.read_csv(self.__external_observable_file,**self.__external_observable_info['readcsv'])
            if len(self.__external_observable_info['dropna_subset']) > 0:
                self.__ExternalObservables.dropna(subset = self.__external_observable_info['dropna_subset'], axis = 'index', inplace = True)

        
        self.__UseExternalIndicators     = False
        if not self.__external_indicators_file is None:
            self.__UseExternalIndicators = True
            self.__ExternalIndicators    = pd.read_csv(self.__external_indicators_file, index_col = 'Country')
            self.ExternalIndicatorsNames = pd.DataFrame({'Indicator':list(self.__ExternalIndicators.columns)})
            self.ExternalIndicatorsNames.index = [self.measure_data.CleanUpMeasureName(indicator) for indicator in self.ExternalIndicatorsNames['Indicator']]
        
        # set up internal storage
        self.CVresults                   = None
        self.__regrDF                    = {}
        self.__prevalence                = {}

        
        if not self.__cvres_filename is None:
            self.LoadCVResults(filename  = self.__cvres_filename)
    
        #self.colornames = ['#f563e2','#609cff','#00bec4','#00b938','#b79f00','#f8766c', '#75507b'] # Amelie's color scheme
        if self.colornames is None:
            self.colornames              = [cn.upper() for cn in matplotlib.colors.TABLEAU_COLORS.keys() if (cn.upper() != 'TAB:WHITE' and cn.upper() != 'TAB:GRAY')]
        self.L1colors                    = {L1name:self.colornames[i % len(self.colornames)] for i,L1name in enumerate(self.measure_data.MeasureList(mincount = 1).sort_values(by = 'Measure_L1')['Measure_L1'].unique())}
        self.L1colors['Country Effects'] = '#babdb6'
        
        
        self.continent_country_list = {'EUROPE': ['Albania', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Liechtenstein', 'Lithuania', 'Mauritius', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Portugal', 'Romania', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom'], 'ASIA': ['China', 'India', 'Indonesia', 'Japan', 'Kazakhstan', 'Korea, South', 'Kuwait', 'Malaysia', 'New Zealand', 'Singapore', 'Syria', 'Taiwan*', 'Thailand'], 'AMERICAS': ['Brazil', 'Canada', 'Ecuador', 'El Salvador', 'Honduras', 'Mexico', 'US - Alabama', 'US - Alaska', 'US - Arizona', 'US - California', 'US - Colorado', 'US - Connecticut', 'US - Delaware', 'US - Florida', 'US - Georgia', 'US - Hawaii', 'US - Idaho', 'US - Illinois', 'US - Indiana', 'US - Iowa', 'US - Kansas', 'US - Kentucky', 'US - Louisiana', 'US - Maine', 'US - Maryland', 'US - Massachusetts', 'US - Michigan', 'US - New York', 'US - Wisconsin', 'US - Wyoming']}
        if not self.__remove_continent is None:
            if not self.__remove_continent.upper() in self.continent_country_list:
                raise NotImplementedError('"{}" not valid continent'.format(self.__remove_continent))

        self.__FinalDateCountries = None
        if not self.__finaldatefile is None:
            self.__FinalDateCountries = pd.read_csv(self.__finaldatefile)
        
        self.finalModels                 = []
        self.finalResults                = []
        self.finalCV                     = None
        self.finalParameters             = []
        
        
        self.__kwargs_for_pickle         = kwargs
        
        
    
    def addDF(self, df = None, new = None):
        if not new is None:
            if df is None:
                return new
            else:
                return pd.concat([df,new], ignore_index = True, sort = False)
        else:
            return df



    def HaveCountryData(self, country = None):
        if self.__UseExternalObs:
            if country in self.__ExternalObservables[self.__external_observable_info['Country']].unique():
                return True
        else:
            if country in self.jhu_data.countrylist:
                return True
        return False
    
    
    
    def RestrictedCountry(self, country = None):
        if not self.__remove_continent is None:
            if self.__remove_continent.upper() in self.continent_country_list.keys():
                if country in self.continent_country_list[self.__remove_continent.upper()]:
                    return True
        return False


    
    def DateVector(self, start = '1/3/2020', end = None, dateformat = '%d/%m/%Y'):
        if end is None:
            end = datetime.datetime.strftime(datetime.datetime.today(), dateformat)
        if datetime.datetime.strptime(end, dateformat) < datetime.datetime.strptime(start, dateformat):
            start, end = end, start
            
        dvec = []
        curdate = datetime.datetime.strptime(start, dateformat)
        while curdate <= datetime.datetime.strptime(end, dateformat):
            dvec.append(curdate)
            curdate += datetime.timedelta(days = 1)
        
        return dvec
            
    
    
    def GetObservable(self, country, shiftdays = None):
        if not self.__UseExternalObs:
            observable = self.jhu_data.CountryGrowthRates(country = country)[self.__observable_name].values 
            
            startdate                 = self.jhu_data.DateStart(country = country)
            if not self.__MinCaseCount is None:
                startdate, startindex = self.jhu_data.DateAtCases(country = country, cases = self.__MinCaseCount, outputformat = '%d/%m/%Y', return_index = True)
                observable            = observable[startindex:]

        else:
            # import from Nils' files
            
            c_index         = np.array(self.__ExternalObservables[self.__external_observable_info['Country']] == country)
            observable      = self.__ExternalObservables[c_index][self.__external_observable_info['Observable']].values
            if not self.__MinCaseCount is None:
                startindex  = next((i for i,t in enumerate(self.__ExternalObservables[c_index][self.__external_observable_info['Cases']].fillna(0).astype(np.int64) >= 30) if t), None)
            else:
                startindex  = 0
            observable      = observable[startindex:]
            startdate       = (datetime.datetime.strptime(self.__ExternalObservables[c_index][self.__external_observable_info['Date']].values[startindex],self.__external_observable_info['dateformat']) + datetime.timedelta(days = self.__external_observable_info['date_offset'])).strftime('%d/%m/%Y')
                        
        
        startdate_dt = datetime.datetime.strptime(startdate,'%d/%m/%Y')
        possible_end_dates = [startdate_dt + datetime.timedelta(days = len(observable) - 1)]
        
        shiftdays_dt = datetime.timedelta(days = 0)
        if not shiftdays is None:
            shiftdays_dt = datetime.timedelta(days = int(shiftdays))
        
        if not self.__maxlen is None:
            possible_end_dates.append(startdate_dt + datetime.timedelta(days = self.__maxlen))

        if not self.__finaldate is None:
            possible_end_dates.append(datetime.datetime.strptime(self.__finaldate,'%d/%m/%Y') + shiftdays_dt)
            
        if not self.__finaldatefrommeasureDB is None:
            possible_end_dates.append(datetime.datetime.strptime(self.measure_data.FinalDates(countrylist = [country])['Date'].values[0],'%d/%m/%Y') + shiftdays_dt)
        
        if not self.__FinalDateCountries is None:
            finaldates_external = self.__FinalDateCountries[self.__FinalDateCountries['Country'] == country]['Date'].values
            if len(finaldates_external) > 0:
                possible_end_dates.append(datetime.datetime.strptime(finaldates_external[0], self.__finaldatefile_dateformat) + shiftdays_dt)
        
        enddate = np.min(possible_end_dates).strftime('%d/%m/%Y')
        
        obslen = (datetime.datetime.strptime(enddate,'%d/%m/%Y') - datetime.datetime.strptime(startdate,'%d/%m/%Y')).days + 1

        if obslen > 0:
            observable = observable[:obslen]
            return observable, self.DateVector(startdate, enddate)
        else:
            return None,None
    
    
    
    def GenerateRDF(self, shiftdays = 0, countrylist = None):
        if countrylist is None:
            countrylist = self.measure_data.countrylist
        
        regressionDF    = None
        measurelist     = self.measure_data.MeasureList(mincount = self.__MinMeasureCount, measure_level = 2, enddate = self.__finaldate)

        for country in countrylist:
            if (country in self.measure_data.countrylist) and self.HaveCountryData(country) and not self.RestrictedCountry(country):
                
                extend_shiftdays = None
                if self.__extendfinaldateshiftdays: extend_shiftdays = shiftdays
                
                observable, datevector = self.GetObservable(country, shiftdays = extend_shiftdays)
                
                if not observable is None:
                    DF_country = self.measure_data.ImplementationTable( country           = country,
                                                                        measure_level     = 2,
                                                                        startdate         = datetime.datetime.strftime(datevector[0],'%d/%m/%Y'),
                                                                        enddate           = datetime.datetime.strftime(datevector[-1],'%d/%m/%Y'),
                                                                        shiftdays         = shiftdays,
                                                                        mincount          = self.__MinMeasureCount)
                    
                    # additional statistical tests to shuffle data
                    if str(self.__date_randomize).upper() == 'DISTRIBUTION':
                        colnames = list(DF_country.columns)
                        random.shuffle(colnames)
                        DF_country.columns = colnames
                    elif str(self.__date_randomize).upper() == 'RANDOM':
                        for measurename in DF_country.columns:
                            new_implementation_column            = np.zeros(len(DF_country[measurename]))
                            newstart                             = random.randint(0, len(new_implementation_column)-1)
                            new_implementation_column[newstart:] = 1
                            DF_country[measurename]              = new_implementation_column
                    
                    
                    DF_country['Country']     = str(country)
                    DF_country['Observable']  = observable
                    DF_country['Date']        = datevector
                    
                    if self.__UseExternalIndicators:
                        for indicator in self.ExternalIndicatorsNames.iterrows():
                            if country in self.__ExternalIndicators.index:
                                DF_country[indicator[0]] = self.__ExternalIndicators.loc[country,indicator[1][0]]
                            else:
                                DF_country[indicator[0]] = 0

                    regressionDF = self.addDF(regressionDF,DF_country)

        # drop where NaN in Observable is found
        regressionDF.dropna(subset = ['Observable'], axis = 'index', inplace = True)
        # not implemented measures should be NaN values due to merge, set them to 0
        regressionDF.fillna(0, inplace = True)
        
        return regressionDF
            
    
    
    def RegressionDF(self,shiftdays = 0):
        if not shiftdays in self.__regrDF.keys():
            self.__regrDF[shiftdays] = self.GenerateRDF(shiftdays = shiftdays)
        return self.__regrDF[shiftdays]
    
    
    
    def SingleParamCV(self, shiftdays = None, alpha = None, crossvalcount = None, outputheader = {}, L1_wt = 1):
        
        curResDF          = None
        measurelist       = list(self.RegressionDF(shiftdays).columns)
        measurelist.remove('Observable')
        measurelist.remove('Country')
        measurelist.remove('Date')
        
        formula           = 'Observable ~ C(Country) + ' + ' + '.join(measurelist)
        
        countrylist       = list(self.RegressionDF(shiftdays)['Country'].unique())
        
        if crossvalcount is None:            crossvalcount = len(countrylist)
        if crossvalcount > len(countrylist): crossvalcount = len(countrylist)
            
        countrylen        = len(countrylist)
        chunklen          = np.ones(crossvalcount, dtype = np.int) * (countrylen // crossvalcount)
        chunklen[:countrylen % crossvalcount] += 1
        # sample_countries should be a list (of the same length as countrylist) with values corresponding to which test group the country is assigned
        sample_countries  = np.random.permutation(np.concatenate([i * np.ones(chunklen[i],dtype = np.int) for i in range(crossvalcount)]))
        # extend this list to the whole dataset
        samples           = np.concatenate([s * np.ones(len(self.RegressionDF(shiftdays)[self.RegressionDF(shiftdays)['Country'] == countrylist[i]]), dtype = np.int) for i,s in enumerate(sample_countries)])

        for xv_index in range(crossvalcount):
            testcountries = [countrylist[i] for i,s in enumerate(sample_countries) if s == xv_index]
            trainidx      = (samples != xv_index)
            testidx       = (samples == xv_index)
            trainmodel    = smf.ols(formula = formula, data = self.RegressionDF(shiftdays)[trainidx], drop_cols = 'Date')
            testmodel     = smf.ols(formula = formula, data = self.RegressionDF(shiftdays)[testidx], drop_cols = 'Date')
        
            trainresults  = trainmodel.fit_regularized(alpha = alpha, L1_wt = L1_wt)

            test_params   = []
            for paramname in testmodel.exog_names:
                if paramname in trainresults.params.keys():
                    test_params.append(trainresults.params[paramname])
                else:
                    test_params.append(0)

            obs_train     = np.array(trainmodel.endog)
            obs_test      = np.array(testmodel.endog)
            pred_train    = trainmodel.predict(trainresults.params)
            pred_test     = testmodel.predict(test_params)
                
            # store results in dict
            result_dict                          = {'shiftdays': shiftdays, 'alpha': alpha}
            result_dict['Test Countries']        = '; '.join([str(c) for c in testcountries])
            result_dict['Test Sample Size']      = np.sum([len(self.RegressionDF(shiftdays)[self.RegressionDF(shiftdays)['Country'] == country]) for country in testcountries])
            result_dict['Training Sample Size']  = len(self.RegressionDF(shiftdays)) - result_dict['Test Sample Size']
            result_dict['Loglike Training']      = trainmodel.loglike(trainresults.params)
            result_dict['Loglike Test']          = testmodel.loglike(np.array(test_params))
            result_dict['RSS Training']          = np.sum((obs_train - pred_train)**2)
            result_dict['RSS Test']              = np.sum((obs_test - pred_test)**2)
            result_dict['NVar Training']         = np.sum((obs_train - np.mean(obs_train))**2)
            result_dict['NVar Test']             = np.sum((obs_test - np.mean(obs_test))**2)
            result_dict['R2 Training']           = 1 - result_dict['RSS Training']/result_dict['NVar Training']
            result_dict['R2 Test']               = 1 - result_dict['RSS Test']/result_dict['NVar Test']

            result_dict.update({k:v for k,v in trainresults.params.items()})
            
            curResDF = self.addDF(curResDF,pd.DataFrame({k:np.array([v]) for k,v in result_dict.items()}))
        
        return curResDF
    
    
        
    def RunCV(self, shiftdaylist = [0], alphalist = [1e-5], verbose = None, crossvalcount = None, outputheader = {}, L1_wt = 1):
        if verbose is None: verbose = self.__verbose
        if L1_wt != 1:
            outputheader.update({'L1_wt':L1_wt})
            
        for shiftdays, alpha in itertools.product(shiftdaylist,alphalist):
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, 'computing'), end = '\r', flush = True)
            
            curResDF         = self.SingleParamCV(shiftdays = shiftdays, alpha = alpha, crossvalcount = crossvalcount, outputheader = outputheader, L1_wt = L1_wt)
            self.CVresults = self.addDF(self.CVresults,curResDF)
            
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, datetime.datetime.now().strftime('%H:%M:%S')))
        
        # automatically store latest CrossValidation run
        self.CVresults.to_csv('latest_CVrun.csv')



    def SaveCVResults(self, filename = None, reset = False):
        if (not filename is None) and (not self.CVresults is None):
            self.CVresults.to_csv(filename)
            if self.__verbose: print('# saving CV results as "{}"'.format(filename))
        if reset:   self.CVresults = None



    def LoadCVResults(self, filename, reset = True):
        if os.path.exists(filename):
            if self.__verbose:print('# loading CV results from "{}"'.format(filename))
            if reset:   self.CVresults = pd.read_csv(filename,index_col = 0)
            else:       self.CVresults = self.addDF(self.CVresults,pd.read_csv(filename,index_col = 0))
        else:
            raise IOError



    def ProcessCVresults(self, CVresults = None):
        if CVresults is None: CVresults = self.CVresults.copy(deep = True)
        CVresults['alpha'] = CVresults['alpha'].map('{:.6e}'.format) # convert alpha to a format that can be grouped properly
        CVresults =  CVresults.groupby(['shiftdays','alpha'], as_index = False).agg(
            { 'Loglike Test':['mean','std'],
              'Loglike Training':['mean','std'],
              'R2 Test': ['mean','std'],
              'R2 Training': ['mean','std'],
              'RSS Training' : ['sum'],
              'RSS Test': ['sum'],
              'NVar Training':['sum'],
              'NVar Test':['sum'],
              'Test Sample Size':['sum'],
              'Training Sample Size':['sum']
            })
        CVresults.columns = [ 'shiftdays','alpha',
                              'Loglike Test Avgd','Loglike Test Avgd Std',
                              'Loglike Training Avgd','Loglike Training Avgd Std',
                              'R2 Test Avgd','R2 Test Avgd Std',
                              'R2 Training Avgd','R2 Training Avgd Std',
                              'RSS Training Sum',
                              'RSS Test Sum',
                              'NVar Training Sum',
                              'NVar Test Sum',
                              'Test Sample Size',
                              'Training Sample Size'
                            ]
        CVresults['RSS per datapoint Training'] = CVresults['RSS Training Sum']/CVresults['Training Sample Size']
        CVresults['RSS per datapoint Test']     = CVresults['RSS Test Sum']/CVresults['Test Sample Size']
        
        CVresults['R2 Training Weighted']       = 1 - CVresults['RSS Training Sum']/CVresults['NVar Training Sum']
        CVresults['R2 Test Weighted']           = 1 - CVresults['RSS Test Sum']/CVresults['NVar Test Sum']
        
        CVresults['alpha']                      = CVresults['alpha'].astype(np.float64) # return to numbers
        
        CVresults.sort_values(by = ['shiftdays','alpha'], inplace = True)
        return CVresults



    def ComputeFinalModels(self, modelparameters = [(6,1e-3)], L1_wt = 1, crossvalcount = None):
        self.finalModels     = []
        self.finalResults    = []
        self.finalCV         = None
        self.finalParameters = []
        
        for i, (shiftdays, alpha) in enumerate(modelparameters):
            self.finalParameters.append((shiftdays,alpha))
            
            finalCV      = self.SingleParamCV(shiftdays = shiftdays, alpha = alpha, outputheader = {'modelindex':i}, crossvalcount = crossvalcount)
            self.finalCV = self.addDF(self.finalCV, finalCV)
            
            measurelist  = list(self.RegressionDF(shiftdays).columns)
            measurelist.remove('Observable')
            measurelist.remove('Country')
            measurelist.remove('Date')
            
            formula      = 'Observable ~ C(Country) + ' + ' + '.join(measurelist)
            
            self.finalModels.append(smf.ols(data = self.RegressionDF(shiftdays), formula = formula, drop_cols = 'Date'))
            self.finalResults.append(self.finalModels[i].fit_regularized(alpha = alpha, L1_wt = L1_wt))
    

    
    def FinalMeasureEffects(self, drop_zeros = False, rescale = False, include_countries = False, additional_columns = []):
        if not self.finalCV is None:
            finalCVrelative                = self.finalCV.copy(deep = True).drop(columns = 'Test Countries', axis = 0).fillna(0)
            if rescale: finalCVrelative    = finalCVrelative.divide(self.finalCV['Intercept'],axis = 0)
            finalCVrelative                = finalCVrelative.quantile([.5,.025,.975]).T
            finalCVrelative.columns        = ['median', 'low', 'high']
            
            if len(additional_columns) > 0:
                try:
                    finalCVadditional             = self.finalCV.copy(deep = True).drop(columns = 'Test Countries', axis = 0).fillna(0)
                    if rescale: finalCVadditional = finalCVadditional.divide(self.finalCV['Intercept'],axis = 0)
                    finalCVadditional             = finalCVadditional.apply(additional_columns).T
                    finalCVadditional.columns     = additional_columns
                    
                    finalCVrelative        = finalCVrelative.merge(finalCVadditional, left_index = True, right_index = True, how = 'left')
                except:
                    pass
            
            fCV_withNames                  = self.measure_data.MeasureList(mincount = self.__MinMeasureCount, enddate = self.__finaldate, measure_level = 2).merge(finalCVrelative, how = 'left', left_index = True, right_index = True).drop(columns = 'Countries with Implementation', axis = 0).fillna(0)
            
            if include_countries:
                countryDF                  = pd.DataFrame({'Measure_L2':[country[13:].split(']')[0] for country in finalCVrelative.index if country[:3] == 'C(C']})
                countryDF['Measure_L1']    = 'Country Effects'
                countryDF.index            = [country for country in finalCVrelative.index if country[:3] == 'C(C']
                fCV_withCountries          = countryDF.merge(finalCVrelative, how = 'inner', left_index = True, right_index = True)
                fCV_withNames              = self.addDF(fCV_withNames,fCV_withCountries)
            
            
            if drop_zeros: fCV_withNames   = fCV_withNames[(fCV_withNames['median'] != 0) | (fCV_withNames['low'] != 0) | (fCV_withNames['high'] != 0)]
            fCV_withNames.sort_values(by   = ['median','high','low'], inplace = True)
            return fCV_withNames
        else:
            return None



    def EstimateMeasurePrevalence(self, shiftdays = None):
        if shiftdays is None: shiftdays = 0
        
        if not shiftdays in self.__prevalence.keys():
            regrDF = self.RegressionDF(shiftdays)
            
            dropcolumns = ['Observable', 'Date']
            if self.__UseExternalIndicators:
                for indicator in self.ExternalIndicatorsNames.iterrows():
                    dropcolumns.append(indicator)
                
            measures = self.measure_data.MeasureList(mincount = self.__MinMeasureCount, measure_level = 2)

            prevalence_allcountries = regrDF.drop(['Country'] + dropcolumns, axis = 'columns').sum()/len(regrDF)
            prevalence_allcountries.name = 'Prevalence All Countries'
            
            country_implementation_time  = regrDF.drop(dropcolumns, axis = 'columns').groupby(by = 'Country').sum()
            country_observable_length    = regrDF.drop(dropcolumns, axis = 'columns').groupby(by = 'Country').count()

            prevalence_implementingcountries = country_implementation_time.sum()/country_observable_length[country_implementation_time > 0].sum()
            prevalence_implementingcountries.name = 'Prevalence Implementing Countries'
            
            measures = measures.merge(prevalence_allcountries, how = 'right', right_index = True, left_index = True)
            measures = measures.merge(prevalence_implementingcountries, how = 'right', right_index = True, left_index = True)
            
            measures['Fraction of Implementating Countries'] = regrDF.drop(dropcolumns, axis = 'columns').groupby('Country').apply(lambda grp:(grp.sum()>0).astype(int)).sum()/len(regrDF['Country'].unique())
        
            self.__prevalence[shiftdays] = measures.copy(deep = True)
        
        return self.__prevalence[shiftdays]
    
    
    
    def FinalTrajectories(self, countrylist = None):
        model_countrylist = list(set([countryname[13:].strip(']') for model in self.finalModels for countryname in model.exog_names if countryname[:3] == 'C(C']))
        model_countrylist.sort()
        
        if countrylist is None:
            countrylist = model_countrylist
        
        DF_finaltrajectories = None

        for country in [c for c in countrylist if c in model_countrylist]:
            curtraj = None
            for i,model in enumerate(self.finalModels):
                countrymask = np.array(self.RegressionDF(self.finalParameters[i][0])['Country'] == country, dtype = np.bool)
                if curtraj is None:
                    curtraj = self.RegressionDF(self.finalParameters[i][0])[countrymask][['Country','Date','Observable']].copy(deep = True)
                curtraj['Model ({:d},{:.2f})'.format(self.finalParameters[i][0],np.log10(self.finalParameters[i][1]))] = self.finalResults[i].predict()[countrymask]

            DF_finaltrajectories = self.addDF(DF_finaltrajectories, curtraj)
        
        return DF_finaltrajectories



    # ************************************************************************************
    # ** plotting output 
    # ************************************************************************************


    def CheckExternalAxes(self, external_axes = None, figsize = (15,6), panelsize = (1,1), additional_subplots_params = {}):
        if isinstance(panelsize,int):
            required_panels = panelsize
            figpanels = (panelsize,1)
        elif isinstance(panelsize,(tuple,list,np.ndarray)):
            required_panels = panelsize[0] * panelsize[1]
            figpanels = (panelsize[0],panelsize[1])
        else:
            raise ValueError
        
        if not external_axes is None:
            if len(external_axes) >= required_panels:
                return None, external_axes, False
            else:
                raise Exception('Not enough entries in "external_axes" for plot')
        
        fig, axes = plt.subplots(figpanels[0],figpanels[1],figsize = figsize, **additional_subplots_params)
        if required_panels > 1:
            ax = axes.flatten()
        else:
            ax = [axes]
        return fig, ax, True
    
            

    def PlotPrevalenceEffects(self, external_axes = None, filename = 'prevalence_effects.pdf', ylim = (-.3,.1), figsize = (20,6),drop_zeros = False, rescale = False, textlen = 40, title = ''):
        prevalence = self.EstimateMeasurePrevalence().drop(columns = ['Measure_L1','Measure_L2'])
        effects    = self.FinalMeasureEffects(drop_zeros = drop_zeros, rescale = rescale)

        combined   = effects.merge(prevalence, how = 'left', left_index = True, right_index = True)
        
        fig, ax, savefig = self.CheckExternalAxes(external_axes,figsize,(1,3))                

        for l1name, l1group in combined.groupby(by = 'Measure_L1'):
            errors = (l1group[['median','high']].values - l1group[['low','median']].values).T

            ax[0].errorbar(x = l1group['Prevalence All Countries'],             y = l1group['median'], yerr = errors, marker = 'o', ls = 'none', label = textwrap.shorten(l1name,textlen), c = self.L1colors[l1name])
            ax[1].errorbar(x = l1group['Prevalence Implementing Countries'],    y = l1group['median'], yerr = errors, marker = 'o', ls = 'none', label = textwrap.shorten(l1name,textlen), c = self.L1colors[l1name])
            ax[2].errorbar(x = l1group['Fraction of Implementating Countries'], y = l1group['median'], yerr = errors, marker = 'o', ls = 'none', label = textwrap.shorten(l1name,textlen), c = self.L1colors[l1name])
            
        for i in range(len(ax)):
            ax[i].set_xlim([0,1])
            ax[i].set_ylim(ylim)
            ax[i].legend()
            ax[i].set_ylabel('Measure Effect')

        ax[0].set_xlabel('Prevalence All Countries')
        ax[1].set_xlabel('Prevalence Implementing Countries')
        ax[2].set_xlabel('Fraction of Implementing Countries')
        
        if title != '':
            ax[0].set_title(title, weight = 'bold')
        
        if savefig:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches = 'tight')



    def PlotTrajectories(self, external_axes = None, filename = 'trajectories.pdf', columns = 2, ylim = (0,5)):
        ft          = self.FinalTrajectories()
        countrylist = list(ft['Country'].unique())
        models      = [modelname for modelname in ft.columns if modelname[:5] == 'Model']
        modelcount  = len(models)
        
        if modelcount > 0:
            ycount = len(countrylist) // columns
            if len(countrylist) % columns != 0:
                ycount += 1
            
            fig, ax, savefig = self.CheckExternalAxes(external_axes,(15,6.*ycount/columns),(ycount,columns))

            for j,country in enumerate(countrylist):
                countrymask = np.array(ft['Country'] == country)
                ax[j].plot(ft[countrymask]['Observable'].values, lw = 5, label = 'data')
                for modelname in models:
                    ax[j].plot(ft[countrymask][modelname].values, lw = 2, linestyle = '--', label = '{}'.format(modelname[6:]))
                ax[j].annotate(country,[5,0.42], ha = 'left', fontsize = 15)
                ax[j].set_ylim(ylim)
                ax[j].set_xlim([0,60])
                ax[j].legend()
            
            if savefig:
                fig.tight_layout()
                fig.savefig(filename, bbox_inches = 'tight')
       
    
    
    def PlotCVresults(self, external_axes = None, filename = 'CVresults.pdf', shiftdayrestriction = None, ylim = (0,1), xlim = None, figsize = (15,6), averaging_type = 'Weighted', title = '', highlight_shiftdays = None, highlight_alpha = None, ytics = None, mytics = None, grid = True):
        if not averaging_type in ['Weighted', 'Avgd']: averaging_type = 'Weighted'
        processedCV = self.ProcessCVresults().sort_values(by = 'alpha')
        
        fig, ax, savefig = self.CheckExternalAxes(external_axes, figsize, (1,2))
        
        shiftdaylist = np.array(processedCV['shiftdays'].unique(), dtype = np.int)
        shiftdaylist.sort()
        
        if shiftdayrestriction is None:
            shiftdayrestriction = shiftdaylist
        
        for shiftdays in shiftdaylist:
            if shiftdays in shiftdayrestriction:
                plot_parameters = {'lw':2,'alpha':.8}
                if not highlight_shiftdays is None:
                    plot_parameters.update({'linestyle':'--', 'alpha':.5})
                    if highlight_shiftdays == shiftdays:
                        plot_parameters.update({'lw':4,'c':'black','linestyle':'solid','alpha':.8})
                        
                s_index = (processedCV['shiftdays'] == shiftdays).values
                alphalist = processedCV[s_index]['alpha']
                ax[0].plot(alphalist, processedCV[s_index]['R2 Test {}'.format(averaging_type)],     label = r'$\tau = {}$'.format(shiftdays), **plot_parameters)
                ax[1].plot(alphalist, processedCV[s_index]['R2 Training {}'.format(averaging_type)], label = r'$\tau = {}$'.format(shiftdays), **plot_parameters)
        
        for i in range(2):
            ax[i].legend()
            ax[i].set_xlabel(r'Penalty parameter $\alpha$')
            ax[i].set_xscale('log')
            if grid:
                ax[i].grid()
            ax[i].set_ylim(ylim)
            if len(title) > 0:
                ax[i].set_title(title)
            if not xlim is None:
                ax[i].set_xlim(xlim)
            if not ytics is None:
                ax[i].yaxis.set_major_locator(MultipleLocator(ytics))
            if not mytics is None:
                ax[i].yaxis.set_minor_locator(MultipleLocator(mytics))
        
        ax[0].set_ylabel(r'$R^2$ Test')
        ax[1].set_ylabel(r'$R^2$ Training')
        
        
        if not highlight_alpha is None:
            ax[0].vlines(highlight_alpha,ylim[0],ylim[1],color = 'black', lw = 1)
            ax[1].vlines(highlight_alpha,ylim[0],ylim[1],color = 'black', lw = 1)
            xlim = ax[0].get_xlim()
            ax[0].annotate('Optimal paramters',(highlight_alpha*np.power(xlim[0]/xlim[1],.025),.96*ylim[1]+.04*ylim[0]),va='top',rotation=90)
        
        if savefig:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches = 'tight')
    
    
    
    def PlotCVAlphaSweep(self, external_axes = None, shiftdays = None, filename = 'crossval_evaluation.pdf', country_effects = False, measure_effects = True, ylim = (-1,1), figsize = (15,10), rescale = False, xlim = None, legend_parameters = {}, textbreak = 40, parampos = None, highlight_alpha = None):
        if isinstance(shiftdays,int):
            shiftdaylist = [shiftdays]
        elif isinstance(shiftdays,(tuple,list,np.ndarray)):
            shiftdaylist = shiftdays
        elif shiftdays is None:
            shiftdaylist = list(self.CVresults['shiftdays'].unique())
        
        fig, ax, savefig = self.CheckExternalAxes(external_axes, figsize, len(shiftdaylist))

        ax_index = 0
        
        if rescale: cvres_processed = self.CVresults.drop(columns = ['Test Countries']).divide(self.CVresults['Intercept'],axis = 0).T
        else:       cvres_processed = self.CVresults.drop(columns = ['Test Countries']).T
        grouped_parameters = self.measure_data.MeasureList(mincount = self.__MinMeasureCount, enddate = self.__finaldate).merge(cvres_processed,left_index=True,right_index=True,how='inner').T.merge(self.CVresults[['shiftdays','alpha']],left_index=True,right_index=True,how='inner').fillna(0)
        grouped_parameters['alpha'] = grouped_parameters['alpha'].map('{:.6e}'.format)
        grouped_parameters = grouped_parameters.groupby(by = ['shiftdays','alpha'],as_index=False)
    
        median_measures = grouped_parameters.quantile(.5)
        low_measures    = grouped_parameters.quantile(.025)
        high_measures   = grouped_parameters.quantile(.975)

        median_measures['alpha'] = median_measures['alpha'].astype(np.float64)
        low_measures['alpha']    = low_measures['alpha'].astype(np.float64)
        high_measures['alpha']   = high_measures['alpha'].astype(np.float64)

        median_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)
        low_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)
        high_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)

        measuredict = {index:l1name for index,l1name in self.measure_data.MeasureList(mincount = self.__MinMeasureCount, enddate = self.__finaldate)['Measure_L1'].items()}
        
        if country_effects:
            countrycolor    = '#777777'
            
            if rescale: countrydata = self.CVresults.drop(columns = ['Test Countries']).divide(self.CVresults['Intercept'],axis = 0).drop(columns = [colname for colname in self.CVresults.columns if colname[:3] != 'C(C' and not colname in ['shiftdays','alpha']])
            else:       countrydata = self.CVresults.drop(columns = [colname for colname in self.CVresults.columns if colname[:3] != 'C(C' and not colname in ['shiftdays','alpha']])
            countrydata.columns     = [(lambda x: x if x in ['shiftdays','alpha'] else x[13:].strip(']'))(cn) for cn in countrydata.columns]
            countrydata['alpha']    = countrydata['alpha'].map('{:.6e}'.format)
            grouped_country         = countrydata.groupby(by = ['shiftdays','alpha'],as_index=False)

            median_country          = grouped_country.quantile(.5)
            low_country             = grouped_country.quantile(.025)
            high_country            = grouped_country.quantile(.975)

            median_country['alpha'] = median_country['alpha'].astype(np.float64)
            low_country['alpha']    = low_country['alpha'].astype(np.float64)
            high_country['alpha']   = high_country['alpha'].astype(np.float64)

            median_country.sort_values(by = ['shiftdays','alpha'],inplace = True)
            low_country.sort_values(by = ['shiftdays','alpha'],inplace = True)
            high_country.sort_values(by = ['shiftdays','alpha'],inplace = True)

        
        for shiftdays in shiftdaylist:
            if shiftdays in self.CVresults['shiftdays']:
                legendhandles = [matplotlib.lines.Line2D([0],[0],c = value,label = textwrap.shorten(key,textbreak), lw=2) for key,value in self.L1colors.items() if key != 'Country Effects']

                if country_effects:
                    legendhandles.append(matplotlib.lines.Line2D([0],[0],c = self.L1colors['Country Effects'],label = 'Country Effects', lw=2))
                    s_index = (median_country['shiftdays'] == shiftdays)
                    alphalist = median_country[s_index]['alpha'].values
                    for country in [c for c in median_country.columns if c != 'shiftdays' and c != 'alpha']:
                        ax[ax_index].plot(alphalist, median_country[s_index][country].values, c = countrycolor, lw = 2)
                        ylow  = np.array(low_country[s_index][country].values,  dtype = np.float)
                        yhigh = np.array(high_country[s_index][country].values, dtype = np.float)
                        ax[ax_index].fill_between(alphalist, y1 = ylow, y2 = yhigh,color = countrycolor, alpha = .05)

                s_index = (median_measures['shiftdays'] == shiftdays)
                alphalist = median_measures[s_index]['alpha'].values
                for measure in [m for m in median_measures.columns if m != 'shiftdays' and m != 'alpha']:
                    color = self.L1colors[measuredict[measure]]
                    ax[ax_index].plot(alphalist,median_measures[s_index][measure].values, c = color, lw = 2)
                    ylow  = np.array(low_measures[s_index][measure].values,dtype=np.float)
                    yhigh = np.array(high_measures[s_index][measure].values,dtype=np.float)
                    ax[ax_index].fill_between(alphalist,y1 = ylow,y2=yhigh,color = color,alpha = .05)

                ax[ax_index].legend(handles = legendhandles, **legend_parameters)
                ax[ax_index].set_xlabel(r'Penalty Parameter $\alpha$')
                if rescale:
                    ax[ax_index].set_ylabel(r'Relative $\Delta R_t$')
                else:
                    ax[ax_index].set_ylabel(r'$\Delta R_t$')
                
                parampos_rescaled = [np.power(np.min(alphalist),.95)*np.power(np.max(alphalist),0.05), ylim[0] * 0.95 + ylim[1] * 0.05]
                if not parampos is None:
                    parampos_rescaled = [np.power(np.min(alphalist),1 - parampos[0])*np.power(np.max(alphalist),parampos[0]), ylim[0] * (1-parampos[1]) + ylim[1] * parampos[1]]
                ax[ax_index].annotate(r'$\tau = {:d}$'.format(shiftdays),parampos_rescaled)
                ax[ax_index].set_ylim(ylim)
                ax[ax_index].set_xscale('log')
                
                if not xlim is None:
                    ax[ax_index].set_xlim(xlim)
                
                
                if not highlight_alpha is None:
                    if isinstance(highlight_alpha,float):
                        highlight_alpha = [highlight_alpha]
                    for alpha in highlight_alpha:
                        ax[ax_index].vlines(alpha,ylim[0],ylim[1],color = 'black', lw = 1)
                
                ax_index += 1
        
        if savefig:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches = 'tight')
    
    

    def PlotCVShiftdaySweep(self, external_axes = None, alphalist = None, filename = 'crossval_evaluation.pdf', country_effects = False, measure_effects = True, ylim = (-1,1), figsize = (15,10), verticallines = [], rescale = False):
        if isinstance(alphalist,int):
            alphalist = [alphalist]
        elif alphalist is None:
            alphalist = list(self.CVresults['alpha'].unique())
        
        fig, axes, savefig = self.CheckExternalAxes(external_axes, figsize, len(alphalist))

        ax_index = 0
        
        if rescale: cvres_processed = self.CVresults.drop(columns = ['Test Countries']).divide(self.CVresults['Intercept'],axis = 0).T
        else:       cvres_processed = self.CVresults.drop(columns = ['Test Countries']).T
        grouped_parameters = self.measure_data.MeasureList(mincount = self.__MinMeasureCount, enddate = self.__finaldate).merge(cvres_processed,left_index=True,right_index=True,how='inner').T.merge(self.CVresults[['shiftdays','alpha']],left_index=True,right_index=True,how='inner').fillna(0)
        grouped_parameters['alpha'] = grouped_parameters['alpha'].map('{:.6e}'.format)
        grouped_parameters = grouped_parameters.groupby(by = ['shiftdays','alpha'],as_index=False)
    
        median_measures = grouped_parameters.quantile(.5)
        low_measures    = grouped_parameters.quantile(.025)
        high_measures   = grouped_parameters.quantile(.975)

        median_measures['alpha'] = median_measures['alpha'].astype(np.float64)
        low_measures['alpha']    = low_measures['alpha'].astype(np.float64)
        high_measures['alpha']   = high_measures['alpha'].astype(np.float64)

        median_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)
        low_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)
        high_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)

        measuredict = {index:l1name for index,l1name in self.measure_data.MeasureList(mincount = self.__MinMeasureCount, enddate = self.__finaldate)['Measure_L1'].items()}
        
        if country_effects:
            countrycolor    = '#777777'
            countrylist     = pd.DataFrame({'Country':[country for country in restrictedCV.columns if country[:3] == 'C(C']})

            grouped_country = countrylist.merge(self.CVresults.drop(columns = ['Test Countries']).divide(self.CVresults['Intercept'],axis = 0).T,left_index=True,right_index=True,how='inner').T.merge(self.CVresults[['shiftdays','alpha']],left_index=True,right_index=True,how='inner').fillna(0)
            grouped_country['alpha'] = grouped_country['alpha'].map('{:.6e}'.format)
            grouped_country = grouped_country.groupby(by = ['shiftdays','alpha'],as_index=False)

            median_country  = grouped_country.quantile(.5)
            low_country     = grouped_country.quantile(.025)
            high_country    = grouped_country.quantile(.975)

            median_country['alpha'] = median_country['alpha'].astype(np.float64)
            low_country['alpha']    = low_country['alpha'].astype(np.float64)
            high_country['alpha']   = high_country['alpha'].astype(np.float64)

            median_country.sort_values(by = ['shiftdays','alpha'],inplace = True)
            low_country.sort_values(by = ['shiftdays','alpha'],inplace = True)
            high_country.sort_values(by = ['shiftdays','alpha'],inplace = True)


        for alpha in alphalist:
            if alpha in self.CVresults['alpha'].astype(float):
                
                if len(alphalist) == 1:
                    ax = axes
                else:
                    ax = axes[ax_index]
                    ax_index += 1

                if country_effects:
                    a_index = (median_country['alpha'] == alpha)
                    shiftdaylist = median_country[a_index]['shiftdays'].values
                    for country in [c for c in median_country.columns if c != 'shiftdays' and c != 'alpha']:
                        ax.plot(shiftdaylist,median_country[country].values,c = countrycolor,lw = .5)
                        ylow  = np.array(low_country[country].values,dtype=np.float)
                        yhigh = np.array(high_country[country].values,dtype=np.float)
                        ax.fill_between(shiftdaylist,y1 = ylow,y2=yhigh,color = countrycolor,alpha = .05)

                a_index = (median_measures['alpha'] == alpha)
                shiftdaylist = median_measures[a_index]['shiftdays'].values

                for measure in [m for m in median_measures.columns if m != 'shiftdays' and m != 'alpha']:
                    color = self.L1colors[measuredict[measure]]
                    ax.plot(shiftdaylist,median_measures[s_index][measure].values, c = color, lw = 2)
                    ylow  = np.array(low_measures[s_index][measure].values,dtype=np.float)
                    yhigh = np.array(high_measures[s_index][measure].values,dtype=np.float)
                    ax.fill_between(shiftdaylist,y1 = ylow,y2=yhigh,color = color,alpha = .05)

                legendhandles = [matplotlib.lines.Line2D([0],[0],c = value,label = key,lw=2) for key,value in self.L1colors.items() if key != 'Country Effects']
                if country_effects:
                    legendhandles += [matplotlib.lines.Line2D([0],[0],c = countrycolor,label = 'Country Effects',lw=.5)]
                
                for alpha in verticallines:
                    ax.vline(x,zorder = 0, lw = 2, alpha = .5, c = '#000000')
                
                ax.legend(handles = legendhandles )
                ax.set_xlabel(r'Penalty Parameter $\alpha$')
                ax.set_ylabel(r'Relative Effect Size')
                ax.annotate(r'$\tau = {:d}$'.format(shiftdays),[np.power(np.min(alphalist),.97)*np.power(np.max(alphalist),0.03),np.max(ylim)*.9])
                ax.set_ylim(ylim)
                ax.set_xscale('log')            
        
        if savefig:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches = 'tight')
    
    
        


    def PlotMeasureListSorted(self, external_axes = None, filename = 'measurelist_sorted.pdf', drop_zeros = False, figsize = (15,30), labelsize = 40, blacklines = [0], graylines = [-30,-20,-10,10], border = 2, title = '', textbreak = 40, include_countries = False, rescale = False, entryheight = None):
        # get plotting area
        minplot      = np.min(blacklines + graylines)
        maxplot      = np.max(blacklines + graylines)

        # function to plot one row in DF
        def PlotRow(ax, ypos = 1, values = None, color = '#ffffff', boxalpha = .2, textbreak = 40):
            count_labels = len([label for label in values.index if label.startswith('Measure_L')])
            ax.plot(values['median'],[ypos], c = self.L1colors[values[0]], marker = 'D')
            ax.plot([values['low'],values['high']],[ypos,ypos], c = self.L1colors[values[0]], lw = 2)
            background = plt.Rectangle([1e-2 * (minplot - border - count_labels * labelsize), ypos - .4], 1e-2*(count_labels*labelsize + maxplot + border - minplot), .9, fill = True, fc = color, alpha = boxalpha, zorder = 10)
            ax.add_patch(background)
            for i in range(count_labels):
                ax.annotate(textwrap.shorten(str(values[i]), width = textbreak), [1e-2*(minplot - (count_labels - i) * labelsize), ypos - .1])

        # setup
        measure_effects = self.FinalMeasureEffects(drop_zeros = drop_zeros, include_countries = include_countries, rescale = rescale)
        
        # actual plotting including vertical lines
        if not entryheight is None:
            figsize = (figsize[0],(len(measure_effects) + 4.5) * entryheight)
        fig, ax, savefig = self.CheckExternalAxes(external_axes, figsize, 1)

        for j,(index,values) in enumerate(measure_effects.iterrows()):
            PlotRow(ax[0], ypos = -j,values = values, color = self.L1colors[values[0]], textbreak = textbreak)
        for x in blacklines:
            ax[0].plot([1e-2 * x,1e-2 * x],[0.7,-j-0.5], lw = 2, c = 'black',zorder = -2)
            if rescale: label = '{:.0f}%'.format(x)
            else:       label = '{:.2f}'.format(x * 1e-2)
            ax[0].annotate(label,[1e-2*x,0.9],fontsize = 12, c = 'gray', ha = 'center')
        for x in graylines:
            ax[0].plot([1e-2 * x,1e-2 * x],[0.6,-j-0.4], lw = 1, c = 'gray',zorder = -2)
            if rescale: label = '{:.0f}%'.format(x)
            else:       label = '{:.2f}'.format(x * 1e-2)
            ax[0].annotate(label,[1e-2*x,0.9],fontsize = 12, c = 'gray', ha = 'center')
        
        # format output
        if title != '':
            ax[0].annotate(title,[1e-2 * (-2*labelsize + minplot),2], fontsize = 12, weight = 'bold')
        ax[0].annotate('L1 Theme',[1e-2 * (-2*labelsize + minplot), 1], weight = 'bold', fontsize = 12)
        ax[0].annotate('L2 Category', [1e-2 * (-labelsize + minplot), 1], weight = 'bold', fontsize = 12)
        ax[0].annotate(r'$\Delta R_t$',[0,2], fontsize = 12)
        ax[0].set_xlim([1e-2 * (-(len(measure_effects.columns) -3 ) * labelsize - 2*border + minplot), 1e-2 * (maxplot+border)])
        ax[0].set_ylim([-j-2,2.5])
        ax[0].axis('off')
        
        if savefig:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches = 'tight')



    # ************************************************************************************
    # ** python stuff
    # ************************************************************************************

    # make CrossValidation object pickleable ...
    # (getstate, setstate) interacts with (pickle.dump, pickle.load)
    def __getstate__(self):
        return {'kwargs':          self.__kwargs_for_pickle,
                'CVresults':       self.CVresults,
                'finalModels':     self.finalModels,
                'finalResults':    self.finalResults,
                'finalCV':         self.finalCV,
                'finalParameters': self.finalParameters,
                'regressionDF':    self.__regrDF,
                'prevalence':      self.__prevalence}
    
    
    def __setstate__(self,state):
        kwargs = state['kwargs']
        if kwargs is None:      kwargs = {'download_data':False}
        else:                   kwargs.update({'download_data':False})
        self.__init__(**kwargs)
        self.CVresults        = state['CVresults']
        self.finalModels      = state['finalModels']
        self.finalResults     = state['finalResults']
        self.finalCV          = state['finalCV']
        self.finalParameters  = state['finalParameters']
        try:
            self.__regrDF     = state['regressionDF']
        except:
            pass
        try:
            self.__prevalence = state['prevalence']
        except:
            pass
                
