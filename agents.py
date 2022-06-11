from mesa import Agent
import random
import numpy as np


def moving_avg_intersection(points_short, points_long):
    if points_short[-2] < points_long[-2] and points_short[-1] >= points_long[-1]:  # Golden cross
        return 'buy'
    if points_short[-2] > points_long[-2] and points_short[-1] <= points_long[-1]:  # Death cross
        return 'sell'
    return False


class NoiseTrader(Agent):

    def __init__(self, unique_id, model, cash, wealth, start_price):
        super().__init__(unique_id, model)
        self.cur_price = start_price
        self.bunkrupt = False
        self.wealth = wealth
        self.cash = cash

    def bankruptcy(self):
        if self.wealth < 1 and self.cash < 1:
            self.bunkrupt = True
        else:
            self.bunkrupt = False
        return self.bunkrupt

    def step(self, price_cur):
        self.cur_price = price_cur
        if self.bankruptcy() is False:
            offer_price = round(self.random.gauss(self.cur_price, self.cur_price / 64), 2)
            if self.wealth == 0:
                if self.cash // self.cur_price <= 1:
                    a = (self.cash, 1, self.unique_id)
                else:
                    amount_to_buy = self.random.randint(1, self.cash // offer_price)
                    a = (offer_price, amount_to_buy, self.unique_id)
            elif self.cash < offer_price:
                amount_to_buy = self.random.randint(-self.wealth, -1)
                a = (offer_price, amount_to_buy, self.unique_id)
            else:
                intention = random.randint(0, 1)
                if intention == 0:  # buy
                    if self.cash // offer_price == 1:
                        amount_to_buy = 1
                    else:
                        amount_to_buy = self.random.randint(1, self.cash // offer_price)
                else:  # sell
                    if self.wealth == 1:
                        amount_to_buy = -1
                    else:
                        amount_to_buy = self.random.randint(-self.wealth, -1)
                a = (offer_price, amount_to_buy, self.unique_id)
            return a


class LongTerm(Agent):

    def __init__(self, unique_id, model, cash, wealth, start_price, type):
        super().__init__(unique_id, model)
        self.cur_price = start_price
        self.bunkrupt = False
        self.wealth = wealth
        self.cash = cash
        self.type = type

        self.price_history = [self.cur_price] * 59
        self.moving_avg_5 = [self.cur_price]
        self.moving_avg_10 = [self.cur_price]
        self.moving_avg_20 = [self.cur_price]

        self.moving_avg_15 = [self.cur_price]
        self.moving_avg_30 = [self.cur_price]
        self.moving_avg_60 = [self.cur_price]

    def bankruptcy(self):
        if self.wealth < 1 and self.cash < 1:
            self.bunkrupt = True
        else:
            self.bunkrupt = False
        return self.bunkrupt

    def step(self, price_cur, orders):
        self.price_history.append(price_cur)

        self.moving_avg_5.append(np.sum(self.price_history[-5:]) / 5)
        self.moving_avg_10.append(np.sum(self.price_history[-10:]) / 10)
        self.moving_avg_20.append(np.sum(self.price_history[-20:]) / 20)
        self.moving_avg_15.append(np.sum(self.price_history[-15:]) / 15)
        self.moving_avg_30.append(np.sum(self.price_history[-30:]) / 30)
        self.moving_avg_60.append(np.sum(self.price_history[-60:]) / 60)

        if self.bankruptcy() is False:
            if self.type == 0:
                intention = moving_avg_intersection(self.moving_avg_5, self.moving_avg_15)
            if self.type == 1:
                intention = moving_avg_intersection(self.moving_avg_10, self.moving_avg_30)
            if self.type == 2:
                intention = moving_avg_intersection(self.moving_avg_20, self.moving_avg_60)

            if intention == 'buy':
                if not orders:
                    offer_price = price_cur
                else:
                    pr = [x[0][0] for x in orders if x[0][1] > 0]
                    weights = [x[0][1] for x in orders if x[0][1] > 0]
                    offer_price = np.average(pr, weights=weights) if weights else price_cur
                if self.cash < offer_price:
                    a = (self.cash, 1, self.unique_id)
                    return a
                else:
                    amount_to_buy = self.random.randint(self.cash // (offer_price * 2), self.cash // offer_price)
                    a = (offer_price, amount_to_buy, self.unique_id)
                    return a
            if intention == 'sell':
                if not orders:
                    offer_price = price_cur
                else:
                    pr = [x[0][0] for x in orders if x[0][1] < 0]
                    weights = [x[0][1] for x in orders if x[0][1] < 0]
                    offer_price = np.average(pr, weights=weights) if weights else price_cur
                if self.wealth != 0:
                    amount_to_sell = self.random.randint(self.wealth//2, self.wealth)
                    a = (offer_price, -amount_to_sell, self.unique_id)
                    return a


class HighFrequency(Agent):

    def __init__(self, unique_id, model, cash, wealth, start_price, lag, threshold):
        super().__init__(unique_id, model)
        self.cur_price = start_price
        self.bunkrupt = False
        self.wealth = wealth
        self.cash = cash
        self.lag = lag
        self.threshold = threshold
        self.price_hist = []

    def bankruptcy(self):
        if self.wealth < 1 and self.cash < 1:
            self.bunkrupt = True
        else:
            self.bunkrupt = False
        return self.bunkrupt

    def step(self, price_cur):
        self.price_hist.append(price_cur)
        # rate of change - roc
        roc = (price_cur - self.price_hist[-self.lag])/self.price_hist[-self.lag] \
            if len(self.price_hist) >= self.lag else 0
        if self.bankruptcy() is False:

            if roc >= self.threshold:
                amount_to_buy = int(10 * abs(roc) * (self.cash // price_cur))  # self.cash // (price_cur * 2)
                if amount_to_buy != 0:
                    a = (price_cur, amount_to_buy, self.unique_id)
                    return a
            elif roc <= -self.threshold:
                amount_to_sell = int(10 * abs(roc) * self.wealth)  # self.wealth // 2
                if amount_to_sell != 0:
                    a = (price_cur, -amount_to_sell, self.unique_id)
                    return a


class MarketMaker(Agent):

    def __init__(self, unique_id, model, cash,
                 wealth, start_price,
                 spread_width, max_wealth):
        super().__init__(unique_id, model)
        self.bunkrupt = False
        self.orders = None
        self.cash = cash
        self.wealth = wealth
        self.cur_price = start_price
        self.spread_width = spread_width
        self.max_wealth = max_wealth

    def bankruptcy(self):
        if self.wealth < 1 and self.cash < 1:
            self.bunkrupt = True
        else:
            self.bunkrupt = False
        return self.bunkrupt

    def not_enough_cash(self, order):
        if self.cash + order[0] * order[1] < 0:
            return True
        return False

    def not_enough_wealth(self, order):
        if self.wealth - order[1] < 0:
            return True
        return False

    def spread_check(self, order, price):
        # print('case1', order[1], 'price:', price, 'offer', order[0], 'spread:', price * (1 - self.spread_width) <=
        # order[0] <= price * (1 + self.spread_width))
        if order[1] > 0 and price * (1 - self.spread_width) <= order[0]:
            return False
        elif order[1] < 0 and order[0] <= price * (1 + self.spread_width):
            return False
        return True

    def max_wealth_check(self, order):
        if self.wealth - order[1] >= self.max_wealth:
            return True
        else:
            return False

    def step(self, agent, order, price):
        agent.wealth += order[1]
        agent.cash -= order[0] * order[1]
        self.wealth -= order[1]
        self.cash += order[0] * order[1]
        return order[0], order[1]
