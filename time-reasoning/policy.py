import nltk
nltk.download('punkt')
import torch
torch.autograd.set_detect_anomaly(True)
from typing import Callable
from datapoints import Location, TimePoint


class SamplingPolicy():

    def __init__(self, type, policy):
        '''

        :param type: 1, time sampling policy; 2, geo sampling policy
        :param policy: list of tuples of the form (percent_sample, lambda for subsetting all possible negatives)
        '''

        self.type = type
        self.policy = policy

        self.percents = None
        self.lambdas = [each[1] for each in self.policy]
        self.assert_correctness_of_policy()



    def assert_correctness_of_policy(self):

        percents = [float(each[0]) for each in self.policy]
        try:
            assert sum(percents) == 1.0
            new_percents = percents

        except:
            new_percents = [float(each[0])/sum(percents) for each in self.policy]


        self.percents = new_percents

    @classmethod
    def convert_policy_to_numbers(cls, percents, total_requested_sample):

        values = [int(float(each)*total_requested_sample) for each in percents]


        while (sum(values) != total_requested_sample):

            if sum(values < total_requested_sample):
                values[0] += 1

            else:
                values[0] -= 1

        return values




def get_example_time_policy():

    exp_type = 1

    not_same_year: Callable[[TimePoint, TimePoint], bool] \
        = lambda a, b: a.time.date().year != b.time.date().year

    same_year_diff_month: Callable[[TimePoint, TimePoint], bool] = \
        lambda a, b: a.time.date().year == b.time.date().year \
                     and a.time.date().month != b.time.date().month

    same_year_same_month_diff_day: Callable[[TimePoint, TimePoint], bool] = \
        lambda a, b: a.time.date().year == b.time.date().year \
                     and a.time.date().month == b.time.date().month \
                     and a.time.date().day != b.time.date().day \

    sample_rule = [
        (1.0, not_same_year),
        (0.0, same_year_diff_month),
        (0.0, same_year_same_month_diff_day),
    ]

    policy = SamplingPolicy(exp_type, sample_rule)

    return policy

def get_example_geo_policy():

    exp_type = 2

    all_valid: Callable[[TimePoint, TimePoint], bool] \
        = lambda a, b: True

    not_same_continent: Callable[[TimePoint, TimePoint], bool] \
        = lambda a, b: (a.geo.continent != b.geo.continent) or a.geo.continent is None

    not_same_country: Callable[[TimePoint, TimePoint], bool] \
        = lambda  a, b: (a.geo.continent == b.geo.continent and \
                        a.geo.country != b.geo.country) or a.geo.continent is None

    same_country_diff_rest: Callable[[TimePoint, TimePoint], bool] \
        = lambda  a, b: (a.geo.country == b.geo.country and \
                        a.geo.rest != b.geo.rest) or a.geo.continent is None

    sample_rule = [
        (1.0, all_valid),
        (0.0, not_same_continent),
        (0.0, not_same_country),
        (0.0, same_country_diff_rest),
    ]

    policy = SamplingPolicy(exp_type, sample_rule)

    return policy
