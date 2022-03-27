import nltk
nltk.download('punkt')
import torch
torch.autograd.set_detect_anomaly(True)
import datetime
import pandas as pd
from functools import total_ordering



class Location(object):

    country_frame = pd.read_csv('locations.csv')
    country_frame['Continent'] = country_frame['Continent'].str.lower()
    country_frame['Country'] = country_frame['Country'].str.lower()

    def __init__(self, geo_string):



        self.full_str = geo_string
        self._set_country_continent_and_rest()


    def _set_country_continent_and_rest(self):

        tokenized_sentence = nltk.word_tokenize(self.full_str)
        string_list = [word.lower() for word in tokenized_sentence if word.isalnum()]
        if not string_list:
            string_list = tokenized_sentence[0]
        string_final = ' ' + ' '.join(string_list) + ' '

        self.country = None
        self.continent = None

        for index, row in self.country_frame.iterrows():

            if ' ' + row['Country'] + ' ' in string_final:
                self.country = row['Country']
                self.continent = row['Continent']



        if self.country is None:
            for index, row in self.country_frame.iterrows():

                if row['Continent'] in string_final:
                    self.continent = row['Continent']

        if self.country is not None:
            string_final = string_final.replace(self.country, '')

        if self.continent is not None:
            string_final = string_final.replace(self.continent, '')


        self.rest = string_final.strip()




@total_ordering
class TimePoint(object):

    def __init__(self, data_point, idx, root_file):


        img_url = data_point['image_url']
        time_p = data_point['date']
        geo = data_point['location']
        self.img_url = img_url

        info_base_count = time_p.count('-')

        if info_base_count == 2:
            final_time_list = [int(time) for time in time_p.split('-')]
            self.info_base_level = 0

        elif info_base_count == 1:
            final_time_list = [int(time) for time in time_p.split('-')] + [1]
            self.info_base_level = 1


        elif 's' in time_p:
            final_time_list = [int(time_p.strip('s')), 1, 1]
            self.info_base_level = 3

        else:
            final_time_list = [int(time) for time in time_p.split('-')] + [1, 1]
            self.info_base_level = 2


        self.time = datetime.datetime(*final_time_list)
        self.decade = int(self.time.year)%10 - int(self.time.year)
        self.geo = Location(geo)
        self.idx = idx
        self.root_file = root_file
        self.timeline_place = None

    @classmethod
    def process_time_string(cls, time_p):

        try:
            fin_len = len([int(time) for time in time_p.split('-')])
        except:
            fin_len = 0

        return fin_len == 3

    def set_timeline_place(self, idx):

        self.timeline_place = idx

    def __lt__(self, obj):

        return ((self.time < obj.time))

    def __gt__(self, obj):

        return ((self.time > obj.time))

    def __le__(self, obj):

        return ((self.time <= obj.time))

    def __ge__(self, obj):

        return ((self.time >= obj.time))

    def __eq__(self, obj):

        return ((self.time == obj.time))

    def __repr__(self):

        return self.img_url

