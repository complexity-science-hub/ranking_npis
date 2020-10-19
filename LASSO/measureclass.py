import numpy as np
import pandas as pd
import os
import datetime
import re
import urllib.request
import zipfile
import warnings


class COVID19_measures(object):
    '''
    ***************************************************
    **  wrapper for COVID19 measures                 **
    **  github.com/lukasgeyrhofer/corona/            **
    ***************************************************

    access multiple databases for measures,
    choose with option "datasource = DATABASENAME",
    read table of measures from file,
    and download data directly from source if not present or forced (option "download_data = True")
    

    * CCCSL
       https://covid19-interventions.com/
       https://github.com/amel-github/covid19-interventionmeasures
       Desvars-Larrive et al (2020) Scientific Data 7, 285 [https://doi.org/10.1038/s41597-020-00609-9]
    * OXFORD
       https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker
       Hale, Webster, Petherick, Phillips, Kira (2020), CC-BY-SA 4.0
    * ACAPS
       https://www.acaps.org/covid19-government-measures-dataset
       info@acaps.org
       README: https://www.acaps.org/sites/acaps/files/key-documents/files/acaps_covid-19_government_measures_dataset_readme.pdf
    * WHOPHSM
       https://www.who.int/emergencies/diseases/novel-coronavirus-2019/phsm
       PHSMCOVID19@who.int
    * CORONANET
       https://www.coronanet-project.org/
       Cheng, Barceló, Hartnett, Kubinec, Messerschmidt (2020), CC-BY 4.0
       https://doi.org/10.1038/s41562-020-0909-7
    * HIT-COVID
       https://akuko.io/post/covid-intervention-tracking
       https://github.com/HopkinsIDD/hit-covid
       Zheng et al (2020), Scientific Data 7, 286 [https://doi.org/10.1038/s41597-020-00610-2]
       
    main usage:
    ( with default options )
    ***************************************************
        # initialize dataset
        measure_data = COVID19_measures( datasource           = 'CCCSL',
                                         download_data        = False,
                                         measure_level        = 2,
                                         only_first_dates     = False,
                                         unique_dates         = True,
                                         extend_measure_names = False,
                                         resolve_US_states    = False )

        # dataset is iterable
        for countryname, measuredata in measure_data:
            # do stuff with measuredata
            # measuredata is dictionary:
            #   keys:   name of measures 
            #   values: list of dates when implemented


        # obtain DF with columns of implemented measures (0/1) over timespan as index
        # only 'country' is required option, defaults as below
        
        imptable = measure_data.ImplementationTable( country       = 'Austria',
                                                     measure_level = 2,
                                                     startdata     = '22/1/2020',
                                                     enddate       = None,  # today
                                                     shiftdays     = 0 )
    
    ***************************************************
    
    options at initialization:
     * only return measures that correspond to 'measure_level' = [1 .. 4]
     * if 'only_first_dates == True' only return date of first occurence of measure for this level, otherwise whole list
     * if 'extend_measure_dates == True' keys are changed to include all names of all levels of measures
     * if 'unique_dates == True' remove duplicate days in list of values

    ***************************************************        
    '''
    
    def __init__(self,**kwargs):
        
        # set default values of options
        self.__downloaddata       = kwargs.get('download_data',        False )
        self.__measurelevel       = kwargs.get('measure_level',        2     )
        self.__onlyfirstdates     = kwargs.get('only_first_dates',     False )
        self.__uniquedates        = kwargs.get('unique_dates',         True  )
        self.__extendmeasurenames = kwargs.get('extend_measure_names', False )
        self.__countrycodes       = kwargs.get('country_codes',        False )
        self.__dateformat         = kwargs.get('dateformat',           '%d/%m/%Y')
        self.__resolve_US_states  = kwargs.get('resolve_US_states',    False)
        self.__store_raw_data     = kwargs.get('store_raw_data',       False)
        self.__index_name_level   = kwargs.get('index_name_level',     self.__measurelevel)

        self.__max_date_check     = 40 # how many days back from today, used so far only in HIT-COVID, which has upload date in their filenames
        
        self.__datasource         = kwargs.get('datasource','CCCSL').upper()
        self.__datasourceinfo     = {   'CCCSL':  {'dateformat':          '%Y-%m-%d',
                                                   'Country':             'Country',
                                                   'CountryCodes':        'iso3c',
                                                   'MaxMeasureLevel':     4,
                                                   'DownloadURL':         'https://raw.githubusercontent.com/amel-github/covid19-interventionmeasures/master/COVID19_non-pharmaceutical-interventions_version2_utf8.csv',
                                                   'DatafileName':        'COVID19_non-pharmaceutical-interventions.csv',
                                                   'USName':              'United States of America',
                                                   'DatafileReadOptions': {'sep': ',', 'quotechar': '"', 'encoding': 'latin-1'}},
                                        'OXFORD': {'dateformat':          '%Y%m%d',
                                                   'Country':             'CountryName',
                                                   'CountryCodes':        'CountryCode',
                                                   'MaxMeasureLevel':     1,
                                                   'DownloadURL':         'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv',
                                                   'DatafileName':        'OxCGRT_latest.csv',
                                                   'USDownloadURL':       'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_US_states_temp.csv',
                                                   'USDatafileName':      'OxCGRT_US_latest.csv',
                                                   'USName':              'United States',
                                                   'DatafileReadOptions': {}},
                                        'ACAPS':  {'dateformat':          '%d/%m/%Y',
                                                   'Country':             'COUNTRY',
                                                   'CountryCodes':        'ISO',
                                                   'MaxMeasureLevel':     2,
                                                   'DownloadURL':         'https://www.acaps.org/sites/acaps/files/resources/files/acaps_covid19_government_measures_dataset_0.xlsx',
                                                   'DatafileName':        'ACAPS_covid19_measures.xlsx',
                                                   'USName':              'United States of America',
                                                   'DatafileReadOptions': {'sheet_name':'Database'}},
                                        'WHOPHSM':{'dateformat':          '%d/%m/%Y',
                                                   'Country':             'country_territory_area',
                                                   'CountryCodes':        'iso',
                                                   'DownloadURL':         'https://www.who.int/docs/default-source/documents/phsm/{DATE}-phsm-who-int.zip',
                                                   'DownloadURL_dateformat': '%Y%m%d',
                                                   'DownloadFilename':    'who_phsm.zip',
                                                   'DatafileName':        'who_phsm.xlsx',
                                                   'MaxMeasureLevel':      2,
                                                   'USName':              'United States Of America',
                                                   'DatafileReadOptions': {'encoding':'latin-1'}},
                                        'CORONANET':{'dateformat':        '%Y-%m-%d',
                                                   'DownloadURL':         'http://coronanet-project.org/data/coronanet_release.csv',
                                                   'DatafileName':        'coronanet_release.csv',
                                                   'MaxMeasureLevel':     3,
                                                   'Country':             'country',
                                                   'USName':              'United States of America',
                                                   'DatafileReadOptions': {}},
                                        'HITCOVID':{'dateformat':         '%Y-%m-%d',
                                                   'DownloadURL':         'https://github.com/HopkinsIDD/hit-covid/raw/master/data/hit-covid-longdata.csv',
                                                   'DatafileName':        'hit-covid-longdata.csv',
                                                   'MaxMeasureLevel':     2,
                                                   'Country':             'country_name',
                                                   'USName':              'United States of America',
                                                   'DatafileReadOptions': {}}
                                    }

        if not self.__datasource in self.__datasourceinfo.keys():
            raise NotImplementedError('Implemented databases: [' + ', '.join('{}'.format(dbname) for dbname in self.__datasourceinfo.keys()) + ']')
        
        self.__update_dsinfo = kwargs.get('datasourceinfo',None)
        if not self.__update_dsinfo is None:
            self.__datasourceinfo[self.__datasource].update(self.__update_dsinfo)
        
        # can switch internal declaration of countries completely to the ISO3C countrycodes
        # no full names of countries can be used then
        if self.__countrycodes:
            self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['CountryCodes']
            self.__USname         = 'USA'
        else:
            self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['Country']
            self.__USname         = self.__datasourceinfo[self.__datasource]['USName']
        
        # hard coded list since some databases are a mess when trying to resolve US states. need to check against this
        self.__USstateList = ['Alabama', 'Alaska', 'American Samoa', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Guam', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Northern Mariana Islands', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virgin Islands', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

        
        # after setting all options and parameters: load data
        self.ReadData()


    
    def addDF(self, df1 = None, df2 = None):
        if not df2 is None:
            if df1 is None:
                return df2
            else:
                return pd.concat([df1,df2], ignore_index = True, sort = False)
        else:
            return df1



    def URLexists(self, url):
        request = urllib.request.Request(url)
        request.get_method = lambda: 'HEAD'
        try:
            urllib.request.urlopen(request)
            return True
        except urllib.request.HTTPError:
            return False


    
    def GetDownloadURL(self, url):
        if '{DATE}' in url:
            d = 0
            while not self.URLexists(url.format(DATE = (datetime.datetime.today() - datetime.timedelta(days = d)).strftime(self.__datasourceinfo[self.__datasource]['DownloadURL_dateformat']))):
                d += 1
                if d >= self.__max_date_check:
                    break
            return url.format(DATE = (datetime.datetime.today() - datetime.timedelta(days = d)).strftime(self.__datasourceinfo[self.__datasource]['DownloadURL_dateformat']))
        else:
            return url
        


    def filetype(self, datasource = None, filename = None):
        if filename is None:
            filename = self.__datasourceinfo[datasource]['DatafileName']
        return os.path.splitext(filename)[1].strip('.').upper()
    
    
    
    def DownloadData(self):
        # if cannot directly download CSV files, but need to download something else first
        if 'DownloadFilename' in self.__datasourceinfo[self.__datasource].keys():
            download_savefile = self.__datasourceinfo[self.__datasource]['DownloadFilename']
        else:
            download_savefile = self.__datasourceinfo[self.__datasource]['DatafileName']
        
        # if download filename contains a date, try different dates, starting from today, with max (self.__max_date_check) days back
        download_url = self.GetDownloadURL(self.__datasourceinfo[self.__datasource]['DownloadURL'])
        
        # download actual data
        urllib.request.urlretrieve(download_url, download_savefile)
        
        # for some databases, resolution on US states needs additional files
        if self.__resolve_US_states and 'USDownloadURL' in self.__datasourceinfo[self.__datasource].keys():
            download_url = self.GetDownloadURL(self.__datasourceinfo[self.__datasource]['USDownloadURL'])
            download_savefile = self.__datasourceinfo[self.__datasource]['USDatafileName']
            urllib.request.urlretrieve(download_url, download_savefile)

        # download for WHO PHSM comes as zipfile. need to extract file first
        if self.__datasource == 'WHOPHSM':
            who_archive = zipfile.ZipFile(download_savefile)
            who_filename = None
            for fileinfo in who_archive.infolist():
                if self.filetype(filename = fileinfo.filename) in ['CSV','XLSX']:
                    who_archive.extract(fileinfo)
                    who_filename = fileinfo.filename
            if not who_filename is None:
                os.rename(who_filename, self.__datasourceinfo['WHOPHSM']['DatafileName'])
            else:
                raise IOError('did not find appropriate files in ZIP archive')
        
        
        
    def convertDate(self, datestr, inputformat = None, outputformat = None):
        if inputformat is None: inputformat = self.__datasourceinfo[self.__datasource]['dateformat']
        if outputformat is None: outputformat = self.__dateformat
        if isinstance(datestr, (list, tuple, np.ndarray, pd.Series)):
                          return [self.convertDate(x, inputformat = inputformat, outputformat = outputformat) for x in datestr]
                      
        if isinstance(datestr, pd._libs.tslibs.timestamps.Timestamp):
            return datestr.dt.strftime(outputformat)
        else:
            return datetime.datetime.strptime(str(datestr),inputformat).strftime(outputformat)



    def AddNumberToDuplicates(self, namelist, seperator=''):
        if isinstance(namelist, np.ndarray):
            namelist = list(namelist)
        for name in namelist:
            name_count         = namelist.count(name)
            if name_count > 1:
                postfix_str    = ''.join([str(i+1) for i in range(name_count)])
                postfix_iter   = iter(postfix_str)
                for i in range(name_count):
                    name_index = namelist.index(name)
                    namelist[name_index] += seperator + next(postfix_iter)
        return np.array(namelist)



    def CleanUpMeasureName(self, measurename = ''):
        return ''.join([w.capitalize() for w in re.sub('[^a-zA-Z0-9]',' ',measurename).split()]).strip().lstrip('1234567890')



    def ReadData(self):
        def CleanWHOName(name):
            return name.replace('nan -- ','').replace(' -- nan','')
        
        if not os.path.exists(self.__datasourceinfo[self.__datasource]['DatafileName']) or self.__downloaddata:
            self.DownloadData()
        
        if self.filetype(datasource = self.__datasource) == 'CSV':
            readdata       = pd.read_csv(self.__datasourceinfo[self.__datasource]['DatafileName'],**self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        elif self.filetype(datasource = self.__datasource) == 'XLSX':
            readdata       = pd.read_excel(self.__datasourceinfo[self.__datasource]['DatafileName'], **self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        else:
            raise NotImplementedError
        
        # some DBs have additional files for US state resolution
        if self.__resolve_US_states and 'USDatafileName' in self.__datasourceinfo[self.__datasource].keys():
            if self.filetype(filename = self.__datasourceinfo[self.__datasource]['USDatafileName']) == 'CSV':
                readdata_us = pd.read_csv(self.__datasourceinfo[self.__datasource]['USDatafileName'], **self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
            elif self.filetype(filename = self.__datasourceinfo[self.__datasource]['USDatafileName']) == 'XLSX':
                readdata_us = pd.read_excel(self.__datasourceinfo[self.__datasource]['USDatafileName'], **self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
            else:
                raise NotImplementedError
        
        # set a preliminary countrylist, is updated again at the end
        self.__countrylist = list(readdata[self.__countrycolumn].unique())
        
        # individual loading code for the different databases
        # internal structure of the data matched to CCCSL structure
        # create pd.DataFrame with columns: [self.__countrycolumn, 'Date', 'Measure_L1', 'Measure_L2', ... ]
        
        if self.__datasource == 'CCCSL':
            # store CSV directly as data
            self.__data    = readdata.copy(deep = True)
            self.__data['Date'] = self.__data['Date'].apply(self.convertDate)
            for i in range(self.__datasourceinfo['CCCSL']['MaxMeasureLevel']):
                self.__data['Measure_L{:d}'.format(i+1)] = self.__data['Measure_L{:d}'.format(i+1)].str.strip()
            
            # treat each US state as individual country
            if self.__resolve_US_states:
                # first, rename states to 'US - STATENAME'
                self.__data[self.__countrycolumn] = np.where(self.__data[self.__countrycolumn] == self.__USname, 'US - ' + self.__data['State'], self.__data[self.__countrycolumn])
                
                nationwide_id = 'US - United States of America'
                us_states = list(self.__data[self.__data[self.__countrycolumn].str.startswith('US - ')][self.__countrycolumn].unique())
                us_states.remove(nationwide_id)
                
                # copy all nationwide measures for all states
                for us_state in us_states:
                    for index, measure_item in self.__data[self.__data[self.__countrycolumn] == nationwide_id].iterrows():
                        self.__data = self.__data.append(measure_item.replace({self.__countrycolumn:nationwide_id}, value = us_state), ignore_index = True)
                
                # remove nationwide measures
                self.__data.drop(self.__data[self.__data[self.__countrycolumn] == nationwide_id].index, inplace = True)
                
    
        elif self.__datasource == 'OXFORD':
            # construct list of measures from DB column names
            # naming scheme is '[CEH][NUMBER]_NAME'
            # in addition to columns '[CEH][NUMBER]_IsGeneral', '[CEH][NUMBER]_Notes' and '[CEH][NUMBER]_Flag' for more info
            measurecolumns = []
            for mc in readdata.columns:
                if not re.search('^[CEH]\d+\_',mc) is None:
                    if mc[-7:].lower() != 'general' and mc[-5:].lower() != 'notes' and mc[-4:].lower() != 'flag':
                        measurecolumns.append(mc)
            
            # reconstruct same structure of CCCSL DB bottom up
            self.__data    = None
            for country in self.__countrylist:
                countrydata = readdata[readdata[self.__countrycolumn] == country]
                for mc in measurecolumns:
                    for date in countrydata[countrydata[mc].diff() > 0]['Date']:
                        db_entry = pd.DataFrame({self.__countrycolumn: country, 'Date': self.convertDate(date), 'Measure_L1': mc}, index = [0])
                        self.__data = self.addDF(self.__data, db_entry)
            
            if self.__resolve_US_states:
                self.__data.drop(self.__data[self.__data[self.__countrycolumn] == self.__USname].index, inplace = True)
                self.__us_states = readdata_us['RegionName'].unique()
                for usstate in self.__us_states:
                    statedata = readdata_us[readdata_us['RegionName'] == usstate]
                    for mc in measurecolumns:
                        for date in statedata[statedata[mc].diff() > 0]['Date']:
                            db_entry = pd.DataFrame({self.__countrycolumn: 'US - {}'.format(usstate), 'Date': self.convertDate(date), 'Measure_L1': mc}, index = [0])
                            self.__data = self.addDF(self.__data, db_entry)
        
        
        elif self.__datasource == 'ACAPS':
            self.__data = readdata[[self.__countrycolumn,'DATE_IMPLEMENTED','CATEGORY','MEASURE']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn,'Date', 'Measure_L1', 'Measure_L2']
            self.__data.dropna(inplace = True)
            self.__data['Date'] = self.__data['Date'].dt.strftime(self.__dateformat)
            if self.__resolve_US_states:
                warnings.warn('Database "ACAPS" does not support US state resolution')
        
        
        elif self.__datasource == 'WHOPHSM':
            self.__data = readdata[[self.__countrycolumn,'date_start','who_category']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn,'Date','Measure_L1']
            self.__data['Measure_L2'] = (readdata['who_subcategory'].astype(str) + ' -- ' + readdata['who_measure'].astype(str)).apply(CleanWHOName)
            
            # some cleanup
            self.__data.dropna(subset = ['Date'], inplace = True)
            self.__data['Date'] = self.__data['Date'].dt.strftime(self.__dateformat)
            
            # resolve US states
            if self.__resolve_US_states:
                self.__data.drop(self.__data[self.__data[self.__countrycolumn] == self.__USname].index, inplace = True)
                
                nationwide_data = None
                for index, datarow in readdata[readdata[self.__countrycolumn] == self.__USname].dropna(subset = ['date_start']).iterrows():
                    if not datarow['area_covered'] is np.nan:
                        states = [state.strip() for state in str(datarow['area_covered']).split(',')]
                        if datarow['admin_level'] == 'state':
                            for state in states:
                                if state in self.__USstateList:
                                    db_entry = pd.DataFrame({self.__countrycolumn: 'US - {}'.format(state),
                                                            'Date': datarow['date_start'].strftime(self.__dateformat),
                                                            'Measure_L1': str(datarow['who_category']),
                                                            'Measure_L2': CleanWHOName(str(datarow['who_subcategory']) + ' -- ' + str(datarow['who_measure']))
                                                            }, index = [0])
                                    self.__data = self.addDF(self.__data, db_entry)
                    elif datarow['admin_level'] == 'national':
                        nationwide_data = self.addDF(nationwide_data, pd.DataFrame(datarow.to_dict(), index = [0]))
                
                us_states = self.__data[self.__data[self.__countrycolumn].str.startswith('US - ')][self.__countrycolumn].unique()
                for state in us_states:
                    for index, datarow in nationwide_data.iterrows():
                        db_entry = pd.DataFrame({self.__countrycolumn: state,
                                                    'Date': datarow['date_start'].strftime(self.__dateformat),
                                                    'Measure_L1': str(datarow['who_category']),
                                                    'Measure_L2': CleanWHOName(str(datarow['who_subcategory']) + ' -- ' + str(datarow['who_measure']))
                                                    }, index = [0])
                        self.__data = self.addDF(self.__data, db_entry)
                        
            # some cleanup, might not be enough
            self.__data.drop(self.__data[self.__data['Measure_L2'] == 'nan'].index, inplace = True)
            self.__data.drop(self.__data[self.__data['Measure_L2'] == 'unkown -- unknown'].index, inplace = True)
        
        
        elif self.__datasource == 'CORONANET':
            self.__data = readdata[[self.__countrycolumn, 'date_start', 'type', 'type_sub_cat', 'type_text']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn,'Date', 'Measure_L1', 'Measure_L2', 'Measure_L3']
            self.__data.dropna(subset = ['Date'], inplace = True)
            self.__data['Date'] = self.__data['Date'].apply(self.convertDate)
            
            if self.__resolve_US_states:
                self.__data.drop(self.__data[self.__data[self.__countrycolumn] == self.__USname].index, inplace = True)
                readdata_us = readdata[readdata[self.__countrycolumn] == self.__USname].copy(deep = True)
                us_states = readdata_us['province'].dropna().unique()
                for i,datarow in readdata_us.iterrows():
                    db_entry = pd.DataFrame({self.__countrycolumn:'US - {}'.format(datarow['province']),
                                             'Date': self.convertDate(datarow['date_start']),
                                             'Measure_L1': datarow['type'],
                                             'Measure_L2': datarow['type_sub_cat'],
                                             'Measure_L3': datarow['type_text']}, index = [0])
                    if not datarow['province'] is np.nan:
                        # specific measures implemented in a state
                        self.__data = self.addDF(self.__data, db_entry)
                    else:
                        # this seems to code for nationwide measures. add entry to all states
                        for state in us_states:
                            if state in self.__USstateList:
                                db_entry[self.__countrycolumn] = 'US - {}'.format(state)
                                self.__data = self.addDF(self.__data, db_entry)

            # general measures seem to have no L2 description, thus if empty, copy L1
            self.__data['Measure_L2'].fillna(self.__data['Measure_L1'], inplace = True)
            
                
        elif self.__datasource == 'HITCOVID':
            # general structure seems similar to CORONANET, columns have different names, though ...
            self.__data = readdata[[self.__countrycolumn, 'date_of_update', 'intervention_group', 'intervention_name']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn, 'Date', 'Measure_L1', 'Measure_L2']
            self.__data.dropna(subset = ['Date'], inplace = True)
            self.__data['Date'] = self.__data['Date'].apply(self.convertDate)
            
            if self.__resolve_US_states:
                self.__data.drop(self.__data[self.__data[self.__countrycolumn] == self.__USname].index, inplace = True)
                readdata_us = readdata[readdata[self.__countrycolumn] == self.__USname].dropna(subset = ['date_of_update']).copy(deep = True)
                us_states = readdata_us['admin1_name'].dropna().unique()
                for i, datarow in readdata_us.iterrows():
                    db_entry = pd.DataFrame({self.__countrycolumn: 'US - {}'.format(datarow['admin1_name']),
                                             'Date': self.convertDate(datarow['date_of_update']),
                                             'Measure_L1': datarow['intervention_group'],
                                             'Measure_L2': datarow['intervention_name']
                                             }, index = [0])
                    if not datarow['admin1_name'] is np.nan:
                        self.__data = self.addDF(self.__data, db_entry)
                    elif str(datarow['national_entry']).upper() == 'YES':
                        for state in us_states:
                            if state in self.__USstateList:
                                db_entry[self.__countrycolumn] = 'US - {}'.format(state)
                                self.__data = self.addDF(self.__data, db_entry)
            
            self.__data.dropna(inplace = True)
            
        
        else:
            NotImplementedError

        # update countrylist with potential changes during load
        self.__countrylist = list(self.__data[self.__countrycolumn].unique())
        self.__countrylist.sort()
        
        # keep raw data for debug purposes
        if self.__store_raw_data:   self.rawdata = readdata.copy(deep = True)
        else:                       self.rawdata = None

        # generate indexname
        self.__index_name_level = np.min([self.__index_name_level, self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']])
        mheader = ['Measure_L{}'.format(i+1) for i in range(self.__index_name_level)]
        allmeasures = self.__data[mheader].drop_duplicates()
        allmeasures = allmeasures.replace(r'^\s*$', np.nan, regex = True)
        allmeasures['tmp_L1'] = allmeasures['Measure_L1']
        for i in range(self.__index_name_level - 1):
           allmeasures['tmp_L{}'.format(i+2)] = allmeasures['Measure_L{}'.format(i+2)].fillna(allmeasures['tmp_L{}'.format(i+1)])
        allmeasures['indexname'] = self.AddNumberToDuplicates(allmeasures['tmp_L{}'.format(self.__index_name_level)].apply(self.CleanUpMeasureName).values)
        allmeasures.drop(columns = ['tmp_L{}'.format(i+1) for i in range(self.__index_name_level)], inplace = True)
        self.__data = self.__data.merge(allmeasures, left_on = mheader, right_on = mheader, how = 'left')
        
    
    
    def RemoveCountry(self, country = None):
        if country in self.__countrylist:
            self.__countrylist.remove(country)
            self.__data = self.__data[self.__data[self.__countrycolumn] != country]
    
    
    
    def RemoveMeasures(self, measure = '', measure_level = None):
        if len(measure) > 0:
            if not measure_level is None:
                self.__data = self.__data[~self.__data['Measure_L{}'.format(measure_level)].fillna('').str.contains(measure)]
            else:
                for measure_level in range(self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']):
                    self.__data = self.__data[~self.__data['Measure_L{}'.format(measure_level+1)].fillna('').str.contains(measure)]
    
    
    
    def RenameCountry(self, country = None, newname = None):
        if country in self.__countrylist:
            self.__countrylist.remove(country)
            self.__countrylist.append(newname)
            self.__countrylist.sort()
            self.__data.replace(to_replace = {self.__countrycolumn: country}, value = newname, inplace = True)
    
    
    
    def SortDates(self,datelist):
        tmp_datelist = list(datelist[:])
        tmp_datelist.sort(key = lambda x:datetime.datetime.strptime(x,self.__dateformat))
        return tmp_datelist
    
    
    
    def CountryData(self, country = None, measure_level = None, only_first_dates = None, unique_dates = None, extend_measure_names = None, group_on_indexname = False):
        if country in self.__countrylist:
            
            # if options are not provided, use defaults set at initialization of class
            if measure_level is None:        measure_level        = self.__measurelevel
            if only_first_dates is None:     only_first_dates     = self.__onlyfirstdates
            if unique_dates is None:         unique_dates         = self.__uniquedates
            if extend_measure_names is None: extend_measure_names = self.__extendmeasurenames
            
            if measure_level > self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']: measure_level = self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']

            retdict = {}
            for indexname, datagroup in self.__data[self.__data[self.__countrycolumn] == country].groupby(by = 'indexname'):
                if group_on_indexname:
                    key = indexname
                else:
                    if extend_measure_names:
                        key = ' -- '.join([datagroup['Measure_L{}'.format(i+1)].values[0] for i in range(measure_level)])
                    else:
                        key = datagroup['Measure_L{}'.format(measure_level)].values[0]
                
                dates                      = self.SortDates(datagroup['Date'].values)
                if unique_dates:     dates = list(set(dates))
                if only_first_dates: dates = [dates[0]]
                retdict[key] = dates
                                                           
            return retdict
        else:
            warnings.warn('No data found for country "{}"'.format(country))
            return None
    
    

    def dates2vector(self, implementdate, start = '22/1/2020', end = None, shiftdays = 0, maxlen = None, datefmt = '%d/%m/%Y', only_pulse = False, binary_output = False):
        # generate vector of 0s and 1s when measure is implemented or not
        # or, when 'only_pulse == True', then output 1 only at dates of implementation
        starttime     = datetime.datetime.strptime(start,         datefmt)
        if end is None:
            endtime   = datetime.datetime.today()
        else:
            endtime   = datetime.datetime.strptime(end,           datefmt)
        implementlist = [datetime.datetime.strptime(date, datefmt) for date in self.SortDates(implementdate)]

        totaldays   = (endtime - starttime).days + 1
        vec         = np.zeros(totaldays)

        if only_pulse:
            for implementtime in implementlist:
                measuredays = (implementtime - starttime).days
                if 0 <= measuredays+shiftdays < len(vec):
                    vec[measuredays+shiftdays] = 1
        else:
            measuredays = (implementlist[0] - starttime).days
            if 0 <= measuredays + shiftdays < len(vec):
                vec[measuredays+shiftdays:] = 1
            
        if not maxlen is None:
            vec     = vec[:min(maxlen,len(vec))]
        
        if binary_output:
            vec = np.array(vec,dtype=np.bool)
        
        return vec



    def ImplementationTable(self, country, measure_level = None, startdate = '22/1/2020', enddate = None, shiftdays = 0, maxlen = None, only_pulse = False, binary_output = False, mincount = None):
        if country in self.__countrylist:
            countrydata  = self.CountryData(country = country, measure_level = measure_level, only_first_dates = False, group_on_indexname = True)
            ret_imptable = pd.DataFrame( { measurename: self.dates2vector(implemented, start = startdate, end = enddate, shiftdays = shiftdays, maxlen = maxlen, only_pulse = only_pulse, binary_output = binary_output)
                                           for measurename, implemented in countrydata.items() } )
            ret_imptable.index = [(datetime.datetime.strptime(startdate,'%d/%m/%Y') + datetime.timedelta(days = i)).strftime(self.__dateformat) for i in range(len(ret_imptable))]
            # check to only return Measures that are in measurelist (which funnels mincount)
            measurelist = self.MeasureList(measure_level = measure_level, mincount = mincount)
            ret_imptable = ret_imptable[ret_imptable.columns[ret_imptable.columns.isin(measurelist.index)]]
            return ret_imptable

        else:
            return None

    
    
    def FindMeasure(self, country, measure_name, measure_level):
        cd = self.CountryData(country, measure_level = measure_level)
        if measure_name in cd.keys():
            return cd[measure_name][0]
        else:
            return None
    
    
    
    def MeasureList(self, countrylist = None, measure_level = None, mincount = None, enddate = None):
        enforce_measure_level_diff     = 0
        if measure_level is None:
            measure_level              = self.__measurelevel
        if measure_level > self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']:
            enforce_measure_level_diff = measure_level - self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']
            measure_level              = self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']
        
        # get list ['Measure_L1', 'Measure_L2', ...]
        mheaders = ['indexname'] + ['Measure_L{:d}'.format(ml+1) for ml in range(measure_level)]
        
        # copy from rawdata
        measurenameDF = pd.DataFrame(self.__data[mheaders + [self.__countrycolumn,'Date']]).replace(np.nan, '', regex = True)
        
        # only keep entries before enddata (set enddate to today if not specified)
        if enddate is None: enddate = datetime.datetime.today().strftime(self.__dateformat)
        measurenameDF.drop(measurenameDF[measurenameDF['Date'].apply(lambda x:datetime.datetime.strptime(x,self.__dateformat) > datetime.datetime.strptime(enddate,self.__dateformat))].index, inplace = True)
        
        # drop 'Date' column, then multiple entries for same measure in same country are indistinguishable. then drop all duplicates
        measurenameDF.drop('Date', axis = 'columns', inplace = True)
        measurenameDF.drop_duplicates(inplace = True)
        
        # if countrylist is explicitely specified, drop all entries not in these countries
        if not countrylist is None: measurenameDF = measurenameDF[measurenameDF[self.__countrycolumn].isin(countrylist)]
        
        # count countries with implementations by grouping
        measurenameDF = measurenameDF.groupby(by = mheaders, as_index=False).count()
        measurenameDF.columns = mheaders + ['Countries with Implementation']
        
        # set index with cleaned measurename
        measurenameDF.set_index('indexname', drop = True, inplace = True)

        # restrict to measures that are implemented in at least 'mincount' countries
        if not mincount is None: measurenameDF = measurenameDF[measurenameDF['Countries with Implementation'] >= mincount]

        # repeatedly copy last available column, if more columns are requested with 'measure_level' parameter
        for ml in range(enforce_measure_level_diff):
            measurenameDF['Measure_L{}'.format(measure_level + ml + 1)] = measurenameDF['Measure_L{}'.format(measure_level)]

        return measurenameDF.sort_values(by = ['Measure_L{}'.format(i+1) for i in range(measure_level)])
    
    
    
    def FinalDates(self, countrylist = None):
        def LastDate(datelist):
            return self.SortDates(datelist)[-1]
        
        if countrylist is None: countrylist = self.__countrylist
        finaldatesDF = self.__data[[self.__countrycolumn,'Date']].groupby(by = self.__countrycolumn, as_index = False).agg({'Date':LastDate})
        return finaldatesDF[finaldatesDF[self.__countrycolumn].isin(countrylist)].set_index(self.__countrycolumn, drop = True)
    
    
    
    def __getattr__(self,key):
        if key == 'data':
            return self.__data
        elif key == 'countrylist':
            return self.__countrylist
        elif key in self.__countrylist:
            return self.CountryData(country = key)
    
    
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country,self.CountryData(country = country)
        
    
