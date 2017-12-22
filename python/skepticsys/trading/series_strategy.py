import pandas as pd
import backtrader as bt
from .basictradestats import BasicTradeStats
from .multiposition import MultiPositionStrategy

class SeriesSignals(bt.Indicator):
    lines = ('signal',)
    params = {
        'signalsrc': None
    }

    def __init__(self):
        if not isinstance(self.params.signalsrc, pd.Series) and not isinstance(self.params.signalsrc, pd.DataFrame):
            raise ValueError('Signalsrc (type %s) must be a Pandas Series.' % (type(self.params.signalsrc)))

        # if signalsrc values are [0,1], make them [-1,1]
        if -1 not in self.params.signalsrc.values and 0 in self.params.signalsrc.values:
            self.signalsrc = self.params.signalsrc.replace(0, -1)
        else:
            self.signalsrc = self.params.signalsrc

    def next(self):
        signalsrc = self.signalsrc
        dt = self.datas[0].datetime.datetime(0)
            # get current dt signal, as next() runs
            # on current bar close and order executes
            # on next bar open
        self.lines.signal[0] = signalsrc.loc[dt] if dt in signalsrc.index else 0

# Create a Stratey
class SeriesStrategy(MultiPositionStrategy):
    params = {
        'signals': None
        , 'enteronloss': True
        , 'closeonsame': True
        , 'executeonsame': False
        , 'tradeintervalbars': 0
        , 'tradeexpirebars': 60
        , 'stake': 1
    }

    def log(self, msg, dt=None):
        dt = self.datas[0].datetime.datetime(0)
        #print('%s | %s' % (dt.isoformat(), msg))

    def log_signal(self, signal, dt_signal):
        self.log('Signal from %s | %s' % (dt_signal.isoformat(), signal))

    def notify_order(self, order):
        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            self.log(
                '%s | %s EXECUTED | Price: %.2f | Cost: %.2f' %
                (order.data._name.upper().ljust(9),
                    'BUY ' if order.isbuy() else 'SELL',
                    order.executed.price,
                    order.executed.value
                )
            )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        super().notify_order(order)

    def notify_cashvalue(self, cash, value):
        self.log('Cash: %.2f | Value: %.2f' % (cash, value))
    
    def notify_trade(self, trade):
        if trade.status in [trade.Closed]:
            self.log('Trade %s Closed | Price: %.2f | PNL: %.2f' % (trade.ref, trade.price, trade.pnl))

        super().notify_trade(trade)

    def __init__(self):
        self.seriessignals = SeriesSignals(signalsrc=self.params.signals)
        self.entrylapse_ = 0
        self.tradeid_ = 100
        super().__init__()

    def next(self):
        ##### Setup #####
        enteronloss = self.params.enteronloss
        closeonsame = self.params.closeonsame
        executeonsame = self.params.executeonsame
        tradeinterval = self.params.tradeintervalbars
        tradeexpire = self.params.tradeexpirebars

        # long/short feeds
        long_feed = self.datas[0]
        short_feed = self.datas[1] if len(self.datas) > 1 else self.datas[0]
        long_position = self.getlongpositionsize(data=long_feed)
        short_position = self.getshortpositionsize(data=short_feed)

        # get signal
        signal = self.seriessignals[0]
        self.log_signal(signal, long_feed.datetime.datetime(0))
        
        ##### Execution #####

        # close
        current_bar = len(self)
        for tradeid in self.long_opentrades_:
            if current_bar - self.long_opentrades_[tradeid] >= tradeexpire:
                self.closetrade(data=long_feed, tradeid=tradeid)
        for tradeid in self.short_opentrades_:
            if current_bar - self.short_opentrades_[tradeid] >= tradeexpire:
                self.closetrade(data=short_feed, tradeid=tradeid)

        # entry
        if self.entrylapse_ < tradeinterval:
            self.entrylapse_ += 1
        else:
            if signal == 1:
                if True: #if enteronloss or long_balance >= 0:
                    self.buy(data=long_feed, tradeid=self.tradeid_)
            elif signal == -1:
                if True: #if enteronloss or short_balance >= 0:
                    self.sell(data=short_feed, tradeid=self.tradeid_)
            self.entrylapse_ = 0
            self.tradeid_ += 1

def process_df(prices, y_pred):
    """Prepare DataFrame indexes for backtrader. `prices` and `y_pred` are assumed
    to have all necessary data points.
    """
    prices_trade = prices.copy()
    y_trade = y_pred.copy()

    # convert index to string type (dtype object)
    if prices_trade.index.dtype != object:
        prices_index = prices_trade.index.map(str)
    else:
        prices_index = prices_trade.index

    if y_trade.index.dtype != object:
        trade_index = y_trade.index.map(str)
    else:
        trade_index = y_trade.index

    # convert index to datetime
    prices_trade.index = pd.to_datetime(prices_index)
    y_trade.index = pd.to_datetime(trade_index)

    return prices_trade, y_trade
