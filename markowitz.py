import numpy as np # For array operations
import pandas as pd # Dataframes
import matplotlib.pyplot as plt # Graph
import quandl # Download financial data

class Portfolio():
    def __init__(self, asset_list, key, start, end):
        self.asset_list = asset_list
        self.key = key
        self.start = start
        self.end = end

    def optimal_portfolio(self):
        quandl.ApiConfig.api_key = self.key
        assets = self.asset_list

        data = quandl.get_table('WIKI/PRICES', ticker = assets, \
        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },\
        date = { 'gte': self.start, 'lte': self.end }, paginate=True)


        wrangle = data.set_index('date')
        table = wrangle.pivot(columns = 'ticker')

        returns_daily = table.pct_change()
        returns_annual = returns_daily.mean() * 250

        # get daily and covariance of returns of the stock
        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 250

        port_ret = []
        port_vola = []
        sharp_ratio = []
        asset_weights = []

        num_assets = len(assets)
        num_port = 50000

        np.random.seed(101)

        for single_portfolio in range(num_port):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns = np.dot(weights, returns_annual)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            premium = returns - 0.05
            sharpe = premium / volatility
            port_ret.append(returns)
            port_vola.append(volatility)
            asset_weights.append(weights)
            sharp_ratio.append(sharpe)

        portfolio = {'Returns': port_ret, 'Volatility': port_vola, \
        'Sharpe Ratio': sharp_ratio}

        # extend original dictionary to accomodate each ticker and weight in the portfolio
        for counter,symbol in enumerate(assets):
            portfolio[symbol+' weight'] = [weight[counter] for weight in asset_weights]

        # make a nice dataframe of the extended dictionary
        df = pd.DataFrame(portfolio)

        # get better labels for desired arrangement of columns
        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] \
        + [stock+' weight' for stock in assets]

        # reorder dataframe columns
        df = df[column_order]



        # find min Volatility & max sharpe values in the dataframe (df)
        min_volatility = df['Volatility'].min()
        max_sharpe = df['Sharpe Ratio'].max()

        # use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_port = df.loc[df['Volatility'] == min_volatility]

        # print
        print(min_variance_port.T)
        print(sharpe_portfolio.T)

        # plot frontier, max sharpe & min Volatility values with a scatterplot
        plt.style.use('seaborn-dark')
        df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                        cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
        plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
        plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        plt.show()

if __name__ == "__main__":
    Portfolio()


"""
port = Portfolio(["AAPL", "IBM", "TSLA", "ADBE", "AMZN"], "v-Kcojh8qzLnaS6MC5dS", 2014-1-1, 2017-12-31)
"""
