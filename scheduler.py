from mesa.time import BaseScheduler
from collections import defaultdict
from typing import List, Type
from mesa.agent import Agent
from mesa.model import Model


class RandomActivationByType(BaseScheduler):
    """
    A scheduler which activates each type of agent once per step, in random
    order, with the order reshuffled every step.
    The `step_type` method is equivalent to the NetLogo 'ask [breed]...' and is
    generally the default behavior for an ABM. The `step` method performs
    `step_type` for each of the agent types.
    Assumes that all agents have a step() method.
    This implementation assumes that the type of an agent doesn't change
    throughout the simulation.
    If you want to do some computations / data collections specific to an agent
    type, you can either:
    - loop through all agents, and filter by their type
    - access via `your_model.scheduler.agents_by_type[your_type_class]`
    """

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.agents_by_type = defaultdict(dict)

    def add(self, agent: Agent) -> None:
        """
        Add an Agent object to the schedule
        Args:
            agent: An Agent to be added to the schedule.
        """

        self._agents[agent.unique_id] = agent
        agent_class: Type[Agent] = type(agent)
        self.agents_by_type[agent_class][agent.unique_id] = agent

    def remove(self, agent: Agent) -> None:
        """
        Remove all instances of a given agent from the schedule.
        """

        del self._agents[agent.unique_id]

        agent_class: Type[Agent] = type(agent)
        del self.agents_by_type[agent_class][agent.unique_id]

    def step(self, shuffle_types: bool = True, shuffle_agents: bool = True) -> None:
        """
        Executes the step of each agent type, one at a time, in random order.
        Args:
            shuffle_types: If True, the order of execution of each types is
                           shuffled.
            shuffle_agents: If True, the order of execution of each agents in a
                            type group is shuffled.
        """
        type_keys: List[Type[Agent]] = list(self.agents_by_type.keys())
        if shuffle_types:
            self.model.random.shuffle(type_keys)
        for agent_class in type_keys:
            self.step_type(agent_class, shuffle_agents=shuffle_agents)
        self.steps += 1
        self.time += 1

    def step_type(self, price, type_class: Type[Agent], shuffle_agents: bool = True) -> list:
        """
        Shuffle order and run all agents of a given type.
        This method is equivalent to the NetLogo 'ask [breed]...'.
        Args:
            type_class: Class object of the type to run.
            :param price:
            :param type_class:
            :param shuffle_agents:
        """
        orders = []
        agent_keys: List[int] = list(self.agents_by_type[type_class].keys())
        if shuffle_agents:
            self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            if self.agents_by_type[type_class][agent_key].bankruptcy is True:
                pass
            else:
                order = self.agents_by_type[type_class][agent_key].step(price)
                if order is not None:
                    orders.append((order, type_class))
        return orders

    def step_type_lt(self, price, type_class: Type[Agent], orders_all, shuffle_agents: bool = True) -> list:
        """
        Shuffle order and run all agents of a given type.
        This method is equivalent to the NetLogo 'ask [breed]...'.
        Args:
            type_class: Class object of the type to run.
            :param orders_all:
            :param price:
            :param type_class:
            :param shuffle_agents:
        """
        orders = []
        agent_keys: List[int] = list(self.agents_by_type[type_class].keys())
        if shuffle_agents:
            self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            if self.agents_by_type[type_class][agent_key].bankruptcy is True:
                pass
            else:
                order = self.agents_by_type[type_class][agent_key].step(price, orders_all)
                if order is not None:
                    orders.append((order, type_class))
        return orders

    def step_market_maker(self, type_class: Type[Agent], type_agent_class: Type[Agent], order, price,\
                          dict_for_hard_lines, hard_line_buy, hard_line_sell, flag) -> list:
        """
        Shuffle order and run all market makers.
        This method is equivalent to the NetLogo 'ask [breed]...'.
        Args:
            type_class: Class object of the type to run.
            :param hard_line_sell: hard line for selling
            :param hard_line_buy: hard line for buying
            :param dict_for_hard_lines: helper dict for checking hard lines
            :param price: current price
            :param order: order for market maker
            :param type_class: market maker class
            :param type_agent_class: who made order
        """
        agent_keys: List[int] = list(self.agents_by_type[type_class].keys())
        self.model.random.shuffle(agent_keys)
        # print(dict_for_hard_lines)
        for agent_key in agent_keys:
            if flag:
                if self.agents_by_type[type_class][agent_key].spread_check(order, price):
                    continue
            if self.agents_by_type[type_class][agent_key].bankruptcy() or \
                    self.agents_by_type[type_class][agent_key].not_enough_cash(order) or \
                    self.agents_by_type[type_class][agent_key].not_enough_wealth(order) or \
                    self.agents_by_type[type_class][agent_key].max_wealth_check(order):
                # print(1, self.agents_by_type[type_class][agent_key].bankruptcy(),
                #       2, self.agents_by_type[type_class][agent_key].not_enough_cash(order),
                #       3, self.agents_by_type[type_class][agent_key].not_enough_wealth(order),
                #       4, self.agents_by_type[type_class][agent_key].spread_check(order, price),
                #       5, self.agents_by_type[type_class][agent_key].max_wealth_check(order))
                continue
            elif order[1] > 0 and dict_for_hard_lines[agent_key]['sell'] + order[1] > hard_line_sell:
                # print(6, dict_for_hard_lines[agent_key]['sell'] + order[1])
                continue
            elif order[1] < 0 and dict_for_hard_lines[agent_key]['buy'] - order[1] > hard_line_buy:
                # print(7, dict_for_hard_lines[agent_key]['buy'] - order[1])
                continue
            else:
                price, count = self.agents_by_type[type_class][agent_key].\
                    step(self.agents_by_type[type_agent_class][order[2]], order, price)
                if count < 0:
                    dict_for_hard_lines[agent_key]['buy'] -= count
                else:
                    dict_for_hard_lines[agent_key]['sell'] += count
                res = (price, count)

                return res

    def get_type_count(self, type_class: Type[Agent]) -> int:
        """
        Returns the current number of agents of certain type in the queue.
        """
        return len(self.agents_by_type[type_class].values())

    def get_num_bankrupt(self, type_class: Type[Agent]):
        """return number of bankrupts"""
        k = 0
        for i in self.agents_by_type[type_class]:
            if self.agents_by_type[type_class][i].bankruptcy():
                k += 1
        return k

    def get_total_wealth(self, type_class: Type[Agent]):
        """sum of all agents' wealth"""
        k = 0
        for i in self.agents_by_type[type_class]:
            k += self.agents_by_type[type_class][i].wealth
        return k

    def get_total_cash(self, type_class: Type[Agent]):
        """sum of all agents' cash"""
        k = 0
        for i in self.agents_by_type[type_class]:
            k += self.agents_by_type[type_class][i].cash
        return k
