import math

from mesa import Model
from ABM.scheduler import RandomActivationByType
from ABM.agents import MarketMaker, NoiseTrader, LongTerm, HighFrequency
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def define_pattern_1(price_history, shock_step):

    """ Patterns:
    1) price decreasing -- shock -- price increasing
    2) price increasing -- shock -- price increasing
    3) price decreasing -- shock -- price decreasing
    4) price increasing -- shock -- price decreasing
    5) stable price -- shock -- price decreasing
    6) stable price -- shock -- price increasing
    7) stable price -- shock -- stable price
    8) price decreasing -- shock -- stable price
    9) price increasing -- shock -- stable price

    Function was used to find special parameters of the system, where some shock happens
    """

    before = price_history[:shock_step]
    after = price_history[shock_step:]
    trend_before, intercept0, r0, p0, se0 = stats.linregress([i for i in range(len(before))], before)
    if not after:
        pattern = 3
        return pattern
    trend_after, intercept1, r1, p1, se1 = stats.linregress([i for i in range(len(after))], after)
    if trend_before >= 0.05:
        if trend_after >= 0.05:
            pattern = 2
        elif trend_after <= -0.05:
            pattern = 4
        else:
            pattern = 9
    elif trend_before <= -0.05:
        if trend_after >= 0.05:
            pattern = 1
        elif trend_after <= -0.05:
            pattern = 3
        else:
            pattern = 8
    else:
        if trend_after >= 0.05:
            pattern = 6
        elif trend_after <= -0.05:
            pattern = 5
        else:
            pattern = 7
    return pattern


def define_pattern(price_history):
    """
    Volatility discription
    1 - decreasing volatility
    2 - stable volatility
    3 - increasing volatility
    """
    values = []
    for i in range(20, len(price_history), 5):
        hist = price_history[i-20:i]
        volatility = np.var(hist)**0.5
        moving_avg = np.sum(hist) / 20
        bollinger_bands_width_val = 4 * volatility / moving_avg
        values.append(round(bollinger_bands_width_val, 5))
    return values


def sorting_orders(orders, cur_price):
    sorted_orders = []
    for i in orders:
        if i[0] is not None:
            sorted_orders.append(i)
    return sorted(sorted_orders, key=lambda x: ((x[0][0] - cur_price) * x[0][1]), reverse=True)


class Market(Model):
    noise_traders = 200
    noise_traders_cash = 3000
    noise_traders_wealth = 10

    market_maker_cash = 200000
    market_maker_wealth = 1000
    mm_type1 = 2
    mm_type2 = 3
    mm_type3 = 1
    mm_spread_width_type1 = 0.005
    mm_spread_width_type2 = 0.01
    mm_spread_width_type3 = 0.01
    mm_max_wealth = 10000
    mm_hard_line_sell = 400
    mm_hard_line_buy = 400

    long_term_type1 = 50
    long_term_type2 = 20
    long_term_type3 = 100
    long_term_cash = 3000
    long_term_wealth = 50

    high_freq = 70
    high_freq_cash = 1000
    high_freq_wealth = 100
    high_freq_lag = 20
    high_freq_threshold = 0.01

    start_price = 100.0
    shock_step = 300
    shock_price_change = 0.9
    verbose = False

    description = (
        "Model"
    )

    def __init__(self,
                 start_price=start_price,
                 noise_traders=noise_traders,
                 noise_traders_cash=noise_traders_cash,
                 noise_traders_wealth=noise_traders_wealth,

                 market_maker_cash=market_maker_cash,
                 market_maker_wealth=market_maker_wealth,
                 mm_type1=mm_type1,
                 mm_type2=mm_type2,
                 mm_type3=mm_type3,
                 mm_spread_width_type1=mm_spread_width_type1,
                 mm_spread_width_type2=mm_spread_width_type2,
                 mm_spread_width_type3=mm_spread_width_type3,
                 mm_max_wealth=mm_max_wealth,
                 mm_hard_line_sell=mm_hard_line_sell,
                 mm_hard_line_buy=mm_hard_line_buy,

                 long_term_type1=long_term_type1,
                 long_term_type2=long_term_type2,
                 long_term_type3=long_term_type3,
                 long_term_cash=long_term_cash,
                 long_term_wealth=long_term_wealth,

                 high_freq=high_freq,
                 high_freq_cash=high_freq_cash,
                 high_freq_wealth=high_freq_wealth,
                 high_freq_lag=high_freq_lag,
                 high_freq_threshold=high_freq_threshold,

                 shock_step=shock_step,
                 shock_price_change=shock_price_change,

                 verbose=verbose):

        super().__init__()
        self.verbose = verbose

        self.schedule = RandomActivationByType(self)
        self.orders = []
        # Create agent's parameters
        self.noise_traders = noise_traders
        self.noise_traders_cash = noise_traders_cash
        self.noise_traders_wealth = noise_traders_wealth

        self.mm_type1 = mm_type1
        self.mm_type2 = mm_type2
        self.mm_type3 = mm_type3
        self.market_maker_cash = market_maker_cash
        self.market_maker_wealth = market_maker_wealth
        self.mm_spread_width_type1 = mm_spread_width_type1
        self.mm_spread_width_type2 = mm_spread_width_type2
        self.mm_spread_width_type3 = mm_spread_width_type3
        self.mm_max_wealth = mm_max_wealth
        self.mm_hard_line_sell = mm_hard_line_sell
        self.mm_hard_line_buy = mm_hard_line_buy

        self.long_term_type1 = long_term_type1
        self.long_term_type2 = long_term_type2
        self.long_term_type3 = long_term_type3
        self.long_term_cash = long_term_cash
        self.long_term_wealth = long_term_wealth

        self.high_freq = high_freq
        self.high_freq_cash = high_freq_cash
        self.high_freq_wealth = high_freq_wealth
        self.high_freq_lag = high_freq_lag
        self.high_freq_threshold = high_freq_threshold

        self.price = start_price

        self.shock_step = shock_step
        self.shock_price_change = shock_price_change

        # Creating noise traders
        for i in range(self.noise_traders):
            a = NoiseTrader(i, self, self.noise_traders_cash, self.noise_traders_wealth, self.price)
            self.schedule.add(a)

        # Creating high frequency traders
        for i in range(self.high_freq):
            a = HighFrequency(i, self, self.high_freq_cash, self.high_freq_wealth, self.price, self.high_freq_lag,
                              self.high_freq_threshold)
            self.schedule.add(a)

        # Creating Market maker type 1
        for i in range(self.mm_type1):
            m = MarketMaker(i,
                            self,
                            self.market_maker_cash,
                            self.market_maker_wealth,
                            self.price,
                            self.mm_spread_width_type1,
                            self.mm_max_wealth)
            self.schedule.add(m)

        # Creating Market maker type 2
        for i in range(self.mm_type1, self.mm_type1 + self.mm_type2):
            m = MarketMaker(i,
                            self,
                            self.market_maker_cash,
                            self.market_maker_wealth,
                            self.price,
                            self.mm_spread_width_type2,
                            self.mm_max_wealth)
            self.schedule.add(m)

        # Creating Market maker type 3
        for i in range(self.mm_type1 + self.mm_type2, self.mm_type1 + self.mm_type2 + self.mm_type3):
            m = MarketMaker(i,
                            self,
                            self.market_maker_cash,
                            self.market_maker_wealth,
                            self.price,
                            self.mm_spread_width_type3,
                            self.mm_max_wealth)
            self.schedule.add(m)

        # Creating 3 types of long-term traders
        for i in range(self.long_term_type1):
            l = LongTerm(i, self, self.long_term_cash, self.long_term_wealth, self.price, 0)
            self.schedule.add(l)
        for i in range(self.long_term_type1, self.long_term_type1 + self.long_term_type2):
            l = LongTerm(i, self, self.long_term_cash, self.long_term_wealth, self.price, 1)
            self.schedule.add(l)
        for i in range(self.long_term_type1 + self.long_term_type2,
                       self.long_term_type1 + self.long_term_type2 + self.long_term_type3):
            l = LongTerm(i, self, self.long_term_cash, self.long_term_wealth, self.price, 2)
            self.schedule.add(l)

        self.running = True

    def step(self, step_num):
        if step_num == self.shock_step:
            self.price = self.price * self.shock_price_change

        nt = self.schedule.step_type(self.price, NoiseTrader, True)
        hf = self.schedule.step_type(self.price, HighFrequency, True)
        lt = self.schedule.step_type_lt(self.price, LongTerm, nt + hf, True)
        offers = sorting_orders(nt + hf + lt, self.price)
        while not offers:
            self.price = 0.95 * self.price
            nt = self.schedule.step_type(self.price, NoiseTrader, True)
            hf = self.schedule.step_type(self.price, HighFrequency, True)
            lt = self.schedule.step_type_lt(self.price, LongTerm, nt + hf, True)
            offers = sorting_orders(nt + hf + lt, self.price)

        prices = []
        list_mm = list(self.schedule.agents_by_type[MarketMaker].keys())
        dict_for_hard_lines = {}
        for i in list_mm:
            dict_for_hard_lines[i] = {'sell': 0, 'buy': 0}

        for h, t in offers:
            result = self.schedule.step_market_maker(MarketMaker, t, h, self.price,
                                                     dict_for_hard_lines,
                                                     self.mm_hard_line_sell,
                                                     self.mm_hard_line_buy, True)
            if result is not None:
                prices.append(result)
        if not prices:
            for h, t in offers[:math.ceil(len(offers)//3)]:
                result = self.schedule.step_market_maker(MarketMaker, t, h, self.price,
                                                         dict_for_hard_lines,
                                                         self.mm_hard_line_sell,
                                                         self.mm_hard_line_buy, False)
                prices.append(result)

        pr = [x[0] for x in prices if x is not None]
        weights = [abs(x[1]) for x in prices if x is not None]
        self.price = np.average(pr, weights=weights) if weights != [] else self.price
        prices.clear()
        if self.verbose:
            print('sum offers', sum(weights))
            print(self.price)
            print(dict_for_hard_lines)

        return self.price, sum(weights)

    def run_model(self, step_count=1000):

        if self.verbose:
            print("Initial number noise traders: ", self.schedule.get_type_count(NoiseTrader))
            print("Initial number high freq: ", self.schedule.get_type_count(HighFrequency))
            print("Initial number long term traders: ", self.schedule.get_type_count(LongTerm))
            print("Initial number market_makers: ", self.schedule.get_type_count(MarketMaker))
        x = []
        price_history = []
        volume = []
        noise_traders_tot_wealth = []
        high_freq_tot_wealth = []
        long_term_tot_wealth = []
        market_maker_tot_wealth = []

        noise_traders_tot_income = []
        high_freq_tot_income = []
        long_term_tot_income = []
        market_maker_tot_income = []

        for i in range(step_count):

            if self.verbose:
                print("STEP", i)

            x.append(i)
            last_price, vol = self.step(i)
            price_history.append(last_price)
            volume.append(vol)

            noise_traders_tot_wealth.append(self.schedule.get_total_wealth(NoiseTrader))
            high_freq_tot_wealth.append(self.schedule.get_total_wealth(HighFrequency))
            long_term_tot_wealth.append(self.schedule.get_total_wealth(LongTerm))
            market_maker_tot_wealth.append(self.schedule.get_total_wealth(MarketMaker))

            noise_traders_tot_income.append(self.schedule.get_total_wealth(NoiseTrader) * last_price +
                                            self.schedule.get_total_cash(NoiseTrader))
            high_freq_tot_income.append(self.schedule.get_total_wealth(HighFrequency) * last_price +
                                        self.schedule.get_total_cash(HighFrequency))
            long_term_tot_income.append(self.schedule.get_total_wealth(LongTerm) * last_price +
                                        self.schedule.get_total_cash(LongTerm))
            market_maker_tot_income.append(self.schedule.get_total_wealth(MarketMaker) * last_price +
                                           self.schedule.get_total_cash(MarketMaker))

            if self.verbose:
                # print("Bankrupts noise traders", self.schedule.get_num_bankrupt(NoiseTrader))
                # print("Bankrupts high freq", self.schedule.get_num_bankrupt(HighFrequency))
                # print("Bankrupts long term", self.schedule.get_num_bankrupt(LongTerm))
                # print("Bankrupts market maker", self.schedule.get_num_bankrupt(MarketMaker))

                print("Noise trader total wealth", self.schedule.get_total_wealth(NoiseTrader))
                print("Noise trader total cash", round(self.schedule.get_total_cash(NoiseTrader), 2))

                print("HF trader total wealth", self.schedule.get_total_wealth(HighFrequency))
                print("HF trader total cash", round(self.schedule.get_total_cash(HighFrequency), 2))

                print("Market maker total wealth", self.schedule.get_total_wealth(MarketMaker))
                print("Market maker total cash", round(self.schedule.get_total_cash(MarketMaker), 2))

                print("Long Term total wealth", self.schedule.get_total_wealth(LongTerm))
                print("Long Term total cash", round(self.schedule.get_total_cash(LongTerm), 2))

            if self.schedule.get_type_count(NoiseTrader) + self.schedule.get_type_count(LongTerm) == \
                    self.schedule.get_num_bankrupt(NoiseTrader) + self.schedule.get_num_bankrupt(LongTerm):
                # print("All agents became bankrupt")
                pattern = define_pattern(price_history)
                return pattern

            if self.schedule.get_type_count(MarketMaker) == self.schedule.get_num_bankrupt(MarketMaker):
                # print("All MoneyMakers became bankrupt")
                pattern = define_pattern(price_history)
                return pattern

            if last_price < 3.5:
                # print("GAME OVER! Price is too low")
                pattern = define_pattern(price_history)
                return pattern

        pattern = define_pattern(price_history)

        if self.verbose:
            print("Noise traders total income start",
                  (self.noise_traders_wealth * 100 + self.noise_traders_cash) * self.noise_traders)
            print("HF traders total income start",
                  (self.high_freq_wealth * 100 + self.high_freq_cash) * self.high_freq)
            print("Long Term total income start", (self.long_term_wealth * 100 + self.long_term_cash) *
                  (self.long_term_type1 + self.long_term_type2 + self.long_term_type3))
            print("Market makers total income start",
                  (self.market_maker_wealth * 100 + self.market_maker_cash) * (
                              self.mm_type1 + self.mm_type2 + self.mm_type3))

            print("Noise traders total income finish", self.schedule.get_total_wealth(NoiseTrader) * last_price +
                  self.schedule.get_total_cash(NoiseTrader))
            print("HF traders total income finish", self.schedule.get_total_wealth(HighFrequency) * last_price +
                  self.schedule.get_total_cash(HighFrequency))
            print("Long Term total income finish", self.schedule.get_total_wealth(LongTerm) * last_price +
                  self.schedule.get_total_cash(LongTerm))
            print("Market makers total income finish", self.schedule.get_total_wealth(MarketMaker) * last_price +
                  self.schedule.get_total_cash(MarketMaker))
            print('pattern', pattern)

            fig0, axs0 = plt.subplots(2, 1)
            axs0[0].plot(x, price_history)
            axs0[0].set_xlabel('step')
            axs0[0].set_ylabel('Price history')
            axs0[0].grid(True)

            axs0[1].plot(x, volume)
            axs0[1].set_xlabel('step')
            axs0[1].set_ylabel('Volume')
            axs0[1].grid(True)

            fig0.tight_layout()
            plt.show()

            fig, axs = plt.subplots(4, 1)

            axs[0].plot(x, noise_traders_tot_income)
            axs[0].set_xlabel('step')
            axs[0].set_ylabel('Total income NT')
            axs[0].grid(True)

            axs[1].plot(x, high_freq_tot_income)
            axs[1].set_xlabel('step')
            axs[1].set_ylabel('Total income HF')
            axs[1].grid(True)

            axs[2].plot(x, long_term_tot_income)
            axs[2].set_xlabel('step')
            axs[2].set_ylabel('Total income LT')
            axs[2].grid(True)

            axs[3].plot(x, market_maker_tot_income)
            axs[3].set_xlabel('step')
            axs[3].set_ylabel('Total income MM')
            axs[3].grid(True)

            fig.tight_layout()
            plt.show()

            fig1, axs1 = plt.subplots(4, 1)
            axs1[0].plot(x, noise_traders_tot_wealth)
            axs1[0].set_xlabel('step')
            axs1[0].set_ylabel('Total wealth NT')
            axs1[0].grid(True)

            axs1[1].plot(x, high_freq_tot_wealth)
            axs1[1].set_xlabel('step')
            axs1[1].set_ylabel('Total wealth HF')
            axs1[1].grid(True)

            axs1[2].plot(x, long_term_tot_wealth)
            axs1[2].set_xlabel('step')
            axs1[2].set_ylabel('Total wealth LT')
            axs1[2].grid(True)

            axs1[3].plot(x, market_maker_tot_wealth)
            axs1[3].set_xlabel('step')
            axs1[3].set_ylabel('Total wealth MM')
            axs1[3].grid(True)

            fig1.tight_layout()
            plt.show()

        return pattern


if __name__ == "__main__":
    print("Run model")

    model = Market(verbose=True)
    a = model.run_model(1000)
    print(a)
