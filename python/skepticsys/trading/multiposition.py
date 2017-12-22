import backtrader as bt

class MultiPositionStrategy(bt.Strategy):
    def gettradesize(self, data=None, tradeid=0):
        if data in self._trades and tradeid in self._trades[data]:
            size = sum([trade.size for trade in self._trades[data][tradeid] if trade.status == trade.Open])
            return size
        else:
            return 0

    def closetrade(self, data=None, tradeid=0, **kwargs):
        size = self.gettradesize(data, tradeid)
        abssize = abs(size)
        
        if size > 0:
            return self.sell(data=data, size=abssize, tradeid=tradeid, **kwargs)
        elif size < 0:
            return self.buy(data=data, size=abssize, tradeid=tradeid, **kwargs)
        else:
            return None

    def getlongpositionsize(self, data=None):
        return sum([x for x in [self.gettradesize(data=data, tradeid=tradeid) for tradeid in self.long_opentrades_] if x > 0])

    def getshortpositionsize(self, data=None):
        return sum([x for x in [self.gettradesize(data=data, tradeid=tradeid) for tradeid in self.short_opentrades_] if x < 0])

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                if order.tradeid not in self.long_opentrades_:
                    self.long_opentrades_[order.tradeid] = len(self) # current bar
            elif order.issell():
                if order.tradeid not in self.short_opentrades_:
                    self.short_opentrades_[order.tradeid] = len(self) # current bar

    def notify_trade(self, trade):
        if trade.status in [trade.Closed]:
            if trade.tradeid in self.long_opentrades_:
                self.long_opentrades_.pop(trade.tradeid, 0)
            if trade.tradeid in self.short_opentrades_:
                self.short_opentrades_.pop(trade.tradeid, 0)

    def __init__(self):
        self.long_opentrades_ = {}
        self.short_opentrades_ = {}
