import numpy as np

class Environment:




    def __init__(self,  initial_balance, chart_data=None, training_data = None):
        self.chart_data = chart_data
        self.training_data = training_data
        self.idx = 0

        self.observation_space_dim = 14     # leverage,average buy price, remaining balance + ohlc +ma5/10/20/60/120
        self.action_space_dim =1
        self.action_space_high = 20          #leverage : -10~10
        self.initial_balance = initial_balance
        self.fee = 0.0004  # 거래 수수료 0.04% (price taker fee of binance)


        #needs constant updates
        self.margin = initial_balance
        self.quantity = 0  # 보유 주식 수
        self.portfolio_value = initial_balance
        self.leverage = 0
        self.average_buy_price = 0
        self.past_portfolio_value = initial_balance


    def reset(self):

        self.idx = 0

        self.quantity = 0
        self.portfolio_value = self.initial_balance
        self.leverage = 0
        self.average_buy_price = 0
        self.margin = self.initial_balance

        return self.build_sample(0)

    def build_sample(self, location):
        if len(self.training_data) > location:
            data = self.training_data.iloc[location].tolist() #open price
            sample=[]
            for i in range(0,self.observation_space_dim-2):
                sample.append(data[i])

            sample.extend(self.get_states())
            return sample
        return None


    def get_states(self):
        return (
            self.leverage/self.action_space_high,
            self.average_buy_price/40000,
            #self.portfolio_value/self.initial_balance
        )

    def random_action(self):
        return [np.random.random_sample() * 2 * self.action_space_high - self.action_space_high]

    def step(self, action):
        done = False
        epsilon = 1
        if self.idx == len(self.chart_data)-1 or self.portfolio_value <epsilon:
            done = True
            next_obs = [0]*self.observation_space_dim
            reward = 0
            info = False

        if done==False:


            [num, open, high, low, close,volume] = self.chart_data.iloc[self.idx]

            past_portfolio_value = self.portfolio_value
            self.past_portfolio_value = past_portfolio_value
            past_leverage = self.leverage
            past_quantity = self.quantity
            past_average_buy_price = self.average_buy_price
            past_margin = self.margin

            self.leverage = action[0]
            ############################################calculate margin, quantity, average buy price #########################################

            if self.leverage>past_leverage and self.leverage<0 or self.leverage<past_leverage and self.leverage>0: #줄이기
                self.margin = past_margin
                self.quantity = self.margin * self.leverage / open

                self.average_buy_price = past_average_buy_price

            elif self.leverage>past_leverage>0 or self.leverage<past_leverage<0: #늘리기
                self.margin = past_margin
                self.quantity = self.margin*self.leverage/open

                self.average_buy_price = past_average_buy_price * abs(past_quantity/self.quantity) + open*abs((self.quantity-past_quantity)/self.quantity)

            else:
                self.margin = past_portfolio_value
                self.quantity = past_portfolio_value * self.leverage / open

                self.average_buy_price = open
            ##########################################calculate trading fee ####################################################################

            penalty = (self.quantity - past_quantity) * open * self.fee
            penalty = abs(penalty)

            self.margin -= penalty
            self.leverage = (self.quantity * self.average_buy_price)/self.margin

            ##########################################calculate liquidation price and determine liquidation######################################

            if self.leverage>0:
                if self.leverage<=1:
                    liquidation_price = -100
                else:
                    liquidation_price = self.average_buy_price*(1-1/self.leverage)
            else:
                    liquidation_price = self.average_buy_price*(1-1/self.leverage)


            liquidation = False

            if self.leverage>0 and low <liquidation_price or self.leverage<0 and high>liquidation_price:
                liquidation=True

            ############################################calculate portfolio value####################################################################

            if(liquidation==False):
                self.portfolio_value = max(0,  self.margin + self.quantity*(close - self.average_buy_price) )
            else:
                self.portfolio_value = 0

            ###########################################calculate reward##############################################################################


            self.idx += 1
            next_obs = self.build_sample(self.idx)


            reward = (self.portfolio_value - past_portfolio_value)/max(epsilon, self.portfolio_value)
            if self.portfolio_value<epsilon and liquidation==False:
                reward =0


            info = liquidation
            reward = np.clip(reward,-1000,1000)


        return next_obs, reward, done, info