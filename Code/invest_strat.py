from datetime import datetime
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import DCNN


class TradingObject:
    stable = 0
    upward = 1
    downward = -1    
    TC = 1

    def __init__(self, bid_price, ask_price, label, label_con, spread_ratio, prob, prop_para):
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.label = label
        self.label_con = label_con
        self.event_point = 0
        self.position = 0
        self.target = 0
        self.target_cut_loss = 0
        self.open_time = 0
        self.shares = 0
        self.profit = 0
        self.signal_num = 0
        self.starttime = 0
        self.spread_ratio = spread_ratio
        self.prop_para = prop_para
        self.prob = prob
        self.cash = 1
        self.details_df = pd.DataFrame(columns=['Time_start', 'Price_start', 'Time_end', 'Price_end',
                                                'Action', 'Profit','Profit_opt', 'Prop', 'Time_duration'])
    
    def update_target(self):
        self.long_target_cut_loss = 1 - self.spread_ratio*2    
        self.short_target_cut_loss = 1 + self.spread_ratio*2   
        self.long_target = 1 + self.spread_ratio*4             
        self.short_target = 1 - self.spread_ratio*4          
        self.post_inter = 20

    def one_step_position(self):
        if self.position == TradingObject.stable:
            if self.label[self.event_point] == TradingObject.upward:
                self.take_long_position()
            elif self.label[self.event_point] == TradingObject.downward:
                self.take_short_position()
        elif self.position == TradingObject.upward:
            if self.label_con[self.event_point] == TradingObject.upward:
                self.continuous_long()
            elif self.label[self.event_point] == TradingObject.downward:
                self.close_position()
                self.take_short_position()
            else:
                self.open_time += 1
        else:  # downward
            if self.label_con[self.event_point] == TradingObject.downward:
                self.continuous_short()
            elif self.label[self.event_point] == TradingObject.upward:
                self.close_position()
                self.take_long_position()
            else:
                self.open_time += 1

    def take_long_position(self):
        self.open_time = 0
        self.position = TradingObject.upward
        self.target = self.ask_price[self.event_point] * self.long_target
        self.target_cut_loss = self.ask_price[self.event_point] * self.long_target_cut_loss
        self.shares = self.prop() / (self.ask_price[self.event_point] * TradingObject.TC)
        self.cash = 1 - self.prop()
        self.starttime = self.event_point

    def take_short_position(self):
        self.open_time = 0
        self.position = TradingObject.downward
        self.target = self.bid_price[self.event_point] * self.short_target
        self.target_cut_loss = self.bid_price[self.event_point] * self.short_target_cut_loss
        self.shares = self.prop() / (self.bid_price[self.event_point] * TradingObject.TC)
        self.cash = 1 - self.prop()
        self.starttime = self.event_point

    def continuous_long(self):
        self.open_time = 0
        self.target = self.ask_price[self.event_point] * self.long_target
        self.target_cut_loss = self.ask_price[self.event_point] * self.long_target_cut_loss
        self.shares += self.cash*self.prop()/(self.ask_price[self.event_point] * TradingObject.TC)
        self.cash = self.cash*(1 - self.prop())

    def continuous_short(self):
        self.open_time = 0
        self.target = self.bid_price[self.event_point] * self.short_target
        self.target_cut_loss = self.bid_price[self.event_point] * self.short_target_cut_loss
        self.shares += self.cash*self.prop()/(self.bid_price[self.event_point] * TradingObject.TC)
        self.cash = self.cash*(1 - self.prop())

    def close_position(self):
        if self.position == TradingObject.downward:
            self.profit += 1 - self.cash - self.shares * self.ask_price[self.event_point] * TradingObject.TC
            self.details_df = self.details_df.append(
                {'Time_start': self.starttime,
                 'Price_start': self.bid_price[self.starttime],
                 'Time_end': self.event_point, 'Price_end': self.ask_price[self.event_point],
                 'Action': 'short',
                 'Profit': 2 - self.ask_price[self.event_point] / self.bid_price[self.starttime] - TradingObject.TC,
                 'Profit_opt': 1 - self.cash - self.shares * self.ask_price[self.event_point] * TradingObject.TC,
                 'Prop': 1 - self.cash,
                 'Time_duration': self.event_point - self.starttime
                 }, ignore_index=True)
        elif self.position == TradingObject.upward:
            self.profit += self.shares * self.bid_price[self.event_point] - 1 + self.cash
            self.details_df = self.details_df.append(
                {'Time_start': self.starttime,
                 'Price_start': self.ask_price[self.starttime],
                 'Time_end': self.event_point, 'Price_end': self.bid_price[self.event_point],
                 'Action': 'long',
                 'Profit': self.bid_price[self.event_point] / self.ask_price[self.starttime] - TradingObject.TC,
                 'Profit_opt': self.shares * self.bid_price[self.event_point] * TradingObject.TC - 1 + self.cash,
                 'Prop': 1 - self.cash,
                 'Time_duration': self.event_point - self.starttime
                 }, ignore_index=True)
        self.shares = 0
        if self.cash != 1:
            self.signal_num += 1
        self.cash = 1
        self.position = TradingObject.stable
        self.open_time = -1

    def prop(self):
        if self.prop_para == [0, 0, 0, 0]:
            return 1
        else:
            p_up = self.prob[self.event_point, 1]
            p_down = self.prob[self.event_point, 0]
            if self.position == TradingObject.upward:
                return max(min(-(p_up*self.prop_para[0]+p_down*self.prop_para[1])/(self.prop_para[0]*self.prop_para[1])*0.04,1), 0)
            elif self.position == TradingObject.downward:
                return max(min(-(p_down*self.prop_para[2]+p_up*self.prop_para[3])/(self.prop_para[2]*self.prop_para[3])*0.04,1), 0)

    def spread_check(self):
        if self.ask_price[self.event_point] >= self.bid_price[self.event_point] * (1 + self.spread_ratio):
            return False
        return True


def signal_processing(signal_df, threshold):
    labels = []
    for i in range(signal_df.shape[0]):
        if max(signal_df[i, :]) > threshold:
            index = np.argmax(signal_df[i, :])
            if index == 0:    # downward
                labels.append(-1)  
            elif index == 1:  # upward
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(0)
    return labels


def return_calculate(bid_prices, ask_prices, labels, labels_continuous, spread_ratios, probs, prop_paras):
    tra_obj = TradingObject(bid_price=bid_prices, ask_price=ask_prices, label=labels, label_con=labels_continuous, 
                             spread_ratio=spread_ratios, prob=probs, prop_para=prop_paras)

    profit_process = [[0,0]]
    signal_nums = [0]
    tra_obj.update_target()

    for tra_obj.event_point in range(50, len(bid_prices)-tra_obj.post_inter):
        if tra_obj.spread_check():
            tra_obj.one_step_position()
        if tra_obj.open_time >= tra_obj.post_inter:
            tra_obj.close_position()
        elif tra_obj.position == TradingObject.upward and tra_obj.target <= tra_obj.bid_price[tra_obj.event_point]:
            tra_obj.close_position()
        elif tra_obj.position == TradingObject.upward and tra_obj.target_cut_loss >= tra_obj.bid_price[
            tra_obj.event_point]:
            tra_obj.close_position()
        elif tra_obj.position == TradingObject.downward and tra_obj.target >= tra_obj.ask_price[tra_obj.event_point]:
            tra_obj.close_position()
        elif tra_obj.position == TradingObject.downward and tra_obj.target_cut_loss <= tra_obj.ask_price[
            tra_obj.event_point]:
            tra_obj.close_position()
        profit_process.append([tra_obj.profit, tra_obj.signal_num])
        signal_nums.append(tra_obj.signal_num)
    if tra_obj.position != TradingObject.stable:
        tra_obj.close_position()
    profit_process.append([tra_obj.profit, tra_obj.signal_num])
    signal_nums.append(tra_obj.signal_num)
    return profit_process, signal_nums, tra_obj.details_df


def invest_strat(data):
    prob, df, spread_ratio = DCNN.DCNN_training(data)

    labels = signal_processing(prob, 0.35)
    labels_continuous = signal_processing(prob, 0)
    T = 50
    bid_prices = df[T-1:, 0]
    ask_prices = df[T-1:, 2]
    spread_ratios = spread_ratio
    prop_para = [0.007128, -0.007995, 0.006866, -0.007494]

    # optimize with FL 
    profit_process_op, signal_nums, details_df = return_calculate(bid_prices, ask_prices, labels, labels_continuous, spread_ratios,
                                                                   prob, prop_para)

    # no optimize with FL 
    profit_process_no, signal_nums, details_df = return_calculate(bid_prices, ask_prices, labels, labels_continuous, spread_ratios,
                                                                   None, [0, 0, 0, 0])

    # no optimize with CE loss function
    labels = signal_processing(prob, 0)
    labels_continuous = signal_processing(prob, 0)                                                           
    profit_process_no_lf, signal_nums, details_df = return_calculate(bid_prices, ask_prices, labels, labels_continuous, spread_ratios,
                                                                      None, [0, 0, 0, 0])
 
    df = np.concatenate((np.array(profit_process_no_lf), np.array(profit_process_no),np.array(profit_process_op)), axis=1)
    return df  
