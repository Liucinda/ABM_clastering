import pandas as pd
import random
from ABM.model import Market
from scipy import stats
import os
import multiprocessing as mp
import pyprind
import time
import functools
import numpy as np
import sys


def get_pattern(values):
    """ Patterns:
        1) price decreasing, volatility increasing
        2) price decreasing, stable volatility
        3) price decreasing, volatility decreasing
        4) stable price, volatility increasing
        5) stable price, stable volatility
        6) stable price, volatility decreasing
        7) price increasing, volatility increasing
        8) price increasing, stable volatility
        9) price increasing, volatility decreasing

        Volatility discription
        1 - volatility decreasing
        2 - stable volatility
        3 - volatility increasing

        1 - stable price, stable volatility
        2 - price decreasing, volatility decreasing
        3 - stable price, volatility decreasing
        4 - price decreasing, stable volatility
        5 - price increasing, stable volatility
        6 - price increasing, volatility decreasing
        7 - stable price, volatility increasing
        8 - price increasing, volatility increasing
        9 - price decreasing, volatility increasing
        """
    not_stable = 0
    # print(len(values))
    for p in range(len(values)):
        if len(values[p]) < 200:
            not_stable += 1
    if not_stable > 0:
        return 0
    pattern = []
    volatility = []
    for h in range(200):
        trend_price = np.mean([values[j][h][0] for j in range(len(values))])
        volat = np.mean([values[j][h][2] for j in range(len(values))])
        volatility_changing = stats.mode([values[j][h][1] for j in range(len(values))]).mode[0]
        volatility.append(round(volat, 2))
        if trend_price > 0.05:
            if volatility_changing == 3:
                pattern.append(8)
            elif volatility_changing == 1:
                pattern.append(6)
            else:
                pattern.append(5)
        elif trend_price < -0.05:
            if volatility_changing == 3:
                pattern.append(9)
            elif volatility_changing == 1:
                pattern.append(2)
            else:
                pattern.append(4)
        else:
            if volatility_changing == 3:
                pattern.append(7)
            elif volatility_changing == 1:
                pattern.append(3)
            else:
                pattern.append(1)
    return volatility


def g(model):
    res = model.run_model(step_count=1000)
    return res


class DataCollector:
    rows = 10

    def __init__(self, rows=rows):
        # parameters_list
        self.rows = rows
        noise_traders_options = [200]  # [60, 70, 80]
        noise_traders_cash_options = [3000]  # [900, 1000, 1100]
        noise_traders_wealth_options = [10]  # [110, 100, 90]

        market_maker_cash_options = [200000]  # [80000, 100000, 120000]
        market_maker_wealth_options = [1000]  # [2000, 3000, 4000]
        mm_type1_options = [2]  # [1, 2, 3]
        mm_type2_options = [3]  # [1, 2, 3]
        mm_type3_options = [1]  # [1, 2, 3]
        mm_spread_width_type1_options = [0.005]  # [0.005, 0.007, 0.003]
        mm_spread_width_type2_options = [0.01]  # [0.01, 0.02, 0.03]
        mm_spread_width_type3_options = [0.01]  # [0.1, 0.15, 0.05]
        mm_max_wealth_options = [10000]  # [8000, 10000, 12000]
        mm_hard_line_sell_options = [400]  # [200, 230, 250]
        mm_hard_line_buy_options = [400]  # [180, 200, 220]

        long_term_type1_options = [50]  # [40, 50, 60]
        long_term_type2_options = [20]  # [15, 20, 25]
        long_term_type3_options = [100]  # [40, 50, 60]
        long_term_cash_options = [3000]  # [2000, 3000, 4000]
        long_term_wealth_options = [50]  # [20, 30, 40]

        high_freq_options = [70]  # [80, 100, 120]
        high_freq_cash_options = [1000]  # [800, 1000, 1200]
        high_freq_wealth_options = [100]  # [40, 50, 60]
        high_freq_lag_options = [20]  # [4, 5, 6]
        high_freq_threshold_options = [0.01]  # [0.01, 0.02, 0.03]

        shock_step_options = [300]
        shock_price_change_options = [0.9]

        noise_traders = random.choices(noise_traders_options, k=self.rows)
        noise_traders_cash = random.choices(noise_traders_cash_options, k=self.rows)
        noise_traders_wealth = random.choices(noise_traders_wealth_options, k=self.rows)

        market_maker_cash = random.choices(market_maker_cash_options, k=self.rows)
        market_maker_wealth = random.choices(market_maker_wealth_options, k=self.rows)
        mm_type1 = random.choices(mm_type1_options, k=self.rows)
        mm_type2 = random.choices(mm_type2_options, k=self.rows)
        mm_type3 = random.choices(mm_type3_options, k=self.rows)
        mm_spread_width_type1 = random.choices(mm_spread_width_type1_options, k=self.rows)
        mm_spread_width_type2 = random.choices(mm_spread_width_type2_options, k=self.rows)
        mm_spread_width_type3 = random.choices(mm_spread_width_type3_options, k=self.rows)
        mm_max_wealth = random.choices(mm_max_wealth_options, k=self.rows)
        mm_hard_line_sell = random.choices(mm_hard_line_sell_options, k=self.rows)
        mm_hard_line_buy = random.choices(mm_hard_line_buy_options, k=self.rows)

        high_freq = random.choices(high_freq_options, k=self.rows)
        high_freq_cash = random.choices(high_freq_cash_options, k=self.rows)
        high_freq_wealth = random.choices(high_freq_wealth_options, k=self.rows)
        high_freq_lag = random.choices(high_freq_lag_options, k=self.rows)
        high_freq_threshold = random.choices(high_freq_threshold_options, k=self.rows)

        long_term_type1 = random.choices(long_term_type1_options, k=self.rows)
        long_term_type2 = random.choices(long_term_type2_options, k=self.rows)
        long_term_type3 = random.choices(long_term_type3_options, k=self.rows)
        long_term_cash = random.choices(long_term_cash_options, k=self.rows)
        long_term_wealth = random.choices(long_term_wealth_options, k=self.rows)

        shock_step = random.choices(shock_step_options, k=self.rows)
        shock_price_change = random.choices(shock_price_change_options, k=self.rows)

        d = {
            'noise_traders': noise_traders,
            'noise_traders_cash': noise_traders_cash,
            'noise_traders_wealth': noise_traders_wealth,
            'market_maker_cash': market_maker_cash,
            'market_maker_wealth': market_maker_wealth,
            'mm_type1': mm_type1,
            'mm_type2': mm_type2,
            'mm_type3': mm_type3,
            'mm_spread_width_type1': mm_spread_width_type1,
            'mm_spread_width_type2': mm_spread_width_type2,
            'mm_spread_width_type3': mm_spread_width_type3,
            'mm_max_wealth': mm_max_wealth,
            'mm_hard_line_sell': mm_hard_line_sell,
            'mm_hard_line_buy': mm_hard_line_buy,
            'long_term_type1': long_term_type1,
            'long_term_type2': long_term_type2,
            'long_term_type3': long_term_type3,
            'long_term_cash': long_term_cash,
            'long_term_wealth': long_term_wealth,
            'high_freq': high_freq,
            'high_freq_cash': high_freq_cash,
            'high_freq_wealth': high_freq_wealth,
            'high_freq_lag': high_freq_lag,
            'high_freq_threshold': high_freq_threshold,
            'shock_step': shock_step,
            'shock_price_change': shock_price_change
        }

        self.data = pd.DataFrame(data=d)
        # self.data = self.data.drop_duplicates()
        self.results = pd.Series([0 for x in range(self.data.shape[0])])
        self.series = pd.DataFrame()

    def fill_results(self, iterations):

        bar = pyprind.ProgBar(self.rows, stream=sys.stdout)
        for p in range(self.rows):
            noise_traders, noise_trader_cash, noise_trader_wealth, market_maker_cash, market_maker_wealth, \
                mm_type1, mm_type2, mm_type3, mm_spread_width_type1, mm_spread_width_type2, mm_spread_width_type3, \
                mm_max_wealth, mm_hard_line_sell, mm_hard_line_buy, long_term_type1, long_term_type2, long_term_type3, \
                long_term_cash, long_term_wealth, high_freq, high_freq_cash, high_freq_wealth, high_freq_lag, \
                high_freq_threshold, shock_step, shock_price_change = self.data.iloc[p].values
            # print(high_frequency)
            mod = Market(100.0,
                         int(noise_traders),
                         noise_trader_cash,
                         int(noise_trader_wealth),
                         market_maker_cash,
                         int(market_maker_wealth),
                         int(mm_type1),
                         int(mm_type2),
                         int(mm_type3),
                         mm_spread_width_type1,
                         mm_spread_width_type2,
                         mm_spread_width_type3,
                         int(mm_max_wealth),
                         int(mm_hard_line_sell),
                         int(mm_hard_line_buy),
                         int(long_term_type1),
                         int(long_term_type2),
                         int(long_term_type3),
                         long_term_cash,
                         int(long_term_wealth),
                         int(high_freq),
                         high_freq_cash,
                         int(high_freq_wealth),
                         int(high_freq_lag),
                         high_freq_threshold,
                         int(shock_step),
                         shock_price_change)

            #   Code was used to find parameters, which produce special behavior of the system
            # print(mp.cpu_count())
            # pool = mp.Pool(mp.cpu_count())
            # multiple_results = [pool.apply_async(g, args=(mod,)) for i in range(iterations)]
            #
            # for i in multiple_results:
            #     i.wait()
            # print([i.get() for i in multiple_results])
            # results = [i.get() for i in multiple_results]
            # res = stats.mode(results).mode[0]
            # # results = pool.apply_async(f, args=(mod, iterations))
            # pool.close()
            #
            # # for k in range(iterations):
            # #     results_list.append(model.run_model())
            # # res = stats.mode(results.get()).mode[0]
            # self.results[p] = res
            # # print('Row num', p)
            # bar.update()

            pool = mp.Pool(mp.cpu_count())
            rrr = pool.apply_async(g, args=(mod,))
            rrr.wait()
            print(rrr.get())
            results = rrr.get()
            pool.close()
            self.results[p] = ', '.join(map(str, results))
            bar.update()

        self.data['res'] = self.results

        return self.data


if __name__ == '__main__':
    d = DataCollector(500)
    df = d.fill_results(iterations=1)
    os.makedirs('C:/Users/l-usoltseva/PycharmProjects/pythonProject/ABM', exist_ok=True)
    df.to_csv('C:/Users/l-usoltseva/PycharmProjects/pythonProject/ABM/shock_300_07_06_1.csv')
    print(df)
