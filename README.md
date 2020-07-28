# PyBATS
> PyBATS is a package for Bayesian time series modeling and forecasting. It is designed to be flexible, offering many options to customize the model form, prior, and forecast period. The focus of the package is the class Dynamic Generalized Linear Model ('dglm'). The supported DGLMs are Poisson, Bernoulli, Normal (a DLM), and Binomial. These models are based upon *Bayesian Forecasting and Dynamic Models*, by West and Harrison (1997).


## Install

PyBATS is hosted on _PyPI_ and can be installed with pip:

    $ pip install pybats

The most recent development version is hosted on [GitHub](https://github.com/lavinei/pybats). You can download and install from there:

```
$ git clone git@github.com:lavinei/pybats.git pybats
$ cd pybats
$ sudo python setup.py install
```

## Quick Start

This provides the most basic example of Bayesian time series analysis using PyBATS. We'll use a public dataset of the sales of a dietary weight control product, along with the advertising spend. These are integer valued counts, which we model with a Poisson Dynamic Generalized Linear Model (DGLM).

First we load in the data, and take a quick look at the first couples of entries:

```python
import numpy as np
import pandas as pd

from pybats_nbdev.shared import load_sales_example
from pybats_nbdev.analysis import *
from pybats_nbdev.point_forecast import *
from pybats_nbdev.plot import *

# Load example sales and advertising data. Source: Abraham & Ledolter (1983)
data = load_sales_example()             
data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
      <th>Advertising</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>21.0</td>
    </tr>
  </tbody>
</table>
</div>



Second, we extract the outcome (_Y_) and covariate (_X_) from this dataset. We'll set the forecast horizon _k=1_ for this example. We could look at multiple forecast horizons by setting k to a larger value. Then the 'analysis' function will automatically perform marginal forecasts across horizons _1:k_.

Finally, we set the start and end time for forecasting. In this case we specify the start and end date with integers, because there are no dates associated with this dataset.

```python
Y = data['Sales'].values
X = data['Advertising'].values

k = 1                                               # Forecast 1 step ahead
forecast_start = 15                                 # Start forecast at time step 15
forecast_end = 35                                   # End forecast at time step 35 (final time step)
```

We use the _analysis_ function as a helper to a) define the model b) Run sequential updating (forward filtering) and c) forecasting. By default, it will return samples from the forecast distribution as well as the model after the final observation.

```python
mod, samples = analysis(Y, X, family="poisson",
forecast_start=forecast_start,      # First time step to forecast on
forecast_end=forecast_end,          # Final time step to forecast on
k=k,                                # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
prior_length=6,                     # How many data point to use in defining prior
rho=.5,                             # Random effect extension, increases variance of Poisson DGLM (see Berry and West, 2019)
deltrend=0.95,                      # Discount factor on the trend component (intercept)
delregn=0.95                        # Discount factor on the regression component
)
```

    beginning forecasting


The model has the posterior mean and variance of the state vector stored as ```mod.a``` and ```mod.C``` respectively.  In this example, we are purely interested in the forecasts. We plot the sales, median forecast, and 95\% credible interval.

```python
import matplotlib.pyplot as plt

# Take the median as the point forecast
forecast = median(samples)                                  

# Plot the 1-step ahead point forecast plus the 95% credible interval
fig, ax = plt.subplots(1,1, figsize=(8, 6))   
ax = plot_data_forecast(fig, ax, Y[forecast_start:forecast_end + k], forecast, samples,
                        dates=np.arange(forecast_start, forecast_end+1, dtype='int'))
ax = ax_style(ax, ylabel='Sales', xlabel='Time', xlim=[forecast_start, forecast_end],
              legend=['Forecast', 'Sales', 'Credible Interval'])
```


![png](docs/images/output_15_0.png)


## Model Components

All models in PyBATS are based on DGLMs, which are well described by their name:

1. **Dynamic**: The coefficients are changing over *time*
2. **Generalized**: We can choose the distribution of the observations (Normal, Poisson, Bernoulli, or Binomial)
3. **Linear**: Forecasts are made by a standard linear combination of coefficients multiplied by predictors

The coefficients in a DGLM are all stored in the state vector, $\theta_t$. The state vector is defined by a set of different components, which are defined up front by the modeler. 

### Trend Component

PyBATS supports $0$, $1$, and $2$ trend coefficients in a DGLM. $1$ trend term is simply an intercept in the model. If there are $2$ trend terms, then the model contains and intercept *and* a local slope, which increments (is added to) the intercept at each time step. Because all coefficients are dynamic, both the intercept and local slope will change over time.

The default setting is to only have an intercept term, which we can see from the model defined in the example above:

```python
mod.ntrend
```




    1



We can access the mean $E\left[ \theta_t \right]$ and variance $V\left[ \theta_t \right]$ of the state vector by the attribute names **a** and **R**:

```python
mod.a[mod.itrend], mod.R.diagonal()[mod.itrend]
```




    (array([[0.63431615]]), array([0.128739]))



So the intercept term has a mean of $0.63$ and a variance of $0.13$.

To add in a local slope, we can re-run the analysis from above, while specifying that `ntrend=2`.

```python
mod, samples = analysis(Y, X, family="poisson",
ntrend=2,                           # Use an intercept and local slope
forecast_start=forecast_start,      # First time step to forecast on
forecast_end=forecast_end,          # Final time step to forecast on
k=k,                                # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
prior_length=6,                     # How many data point to use in defining prior
rho=.5,                             # Random effect extension, increases variance of Poisson DGLM (see Berry and West, 2019)
deltrend=0.95,                      # Discount factor on the trend component (intercept)
delregn=0.95                        # Discount factor on the regression component
)
```

    beginning forecasting


```python
mod.ntrend
```




    2



We can plot the forecasts with this new model, and see that the results are quite similar!

```python
# Take the median as the point forecast
forecast = median(samples)                                  

# Plot the 1-step ahead point forecast plus the 95% credible interval
fig, ax = plt.subplots(1,1, figsize=(8, 6))   
ax = plot_data_forecast(fig, ax, Y[forecast_start:forecast_end + k], forecast, samples,
                        dates=np.arange(forecast_start, forecast_end+1, dtype='int'))
ax = ax_style(ax, ylabel='Sales', xlabel='Time', xlim=[forecast_start, forecast_end],
              legend=['Forecast', 'Sales', 'Credible Interval'])
```


![png](docs/images/output_27_0.png)


### Regression Component

The regression component is the most commonly recognized part of a linear model, where any known predictors are used. In this example, the advertising budget is our known predictor, which is stored in the $X$ array. When there are multiple predictors, each one is stored in a column of $X$.

```python
X[:5]
```




    array([12. , 20.5, 21. , 15.5, 15.3])



The `analysis` function automatically detected that $X$ only had $1$ column, and so it defined the model to have the correct number of regression components.

```python
mod.nregn
```




    1



To understand the impact of advertising on sales, we can look at the regression coefficients:

```python
mod.a[mod.iregn], mod.R.diagonal()[mod.iregn]
```




    (array([[0.09542123]]), array([0.00048791]))



The coefficient mean is $0.095$, with a very small variance. Because the mean of the coefficient is positive, we can interpret this coefficient as saying that an increase in advertising will increase our forecast of sales. Good, that makes sense! To precisely interpret the size of the effect, you need to know the link function for a Poisson DGLM, provided **HERE XXX INSERT LINK XXX**.

To quantify the uncertainty of the parameter, many people like to use the standard deviation (or standard error) of the coefficient, which is simply the square root of the variance. A good rule of thumb to get a pseudo-confidence interval is to add $\pm$ 2*sd(coefficient).

```python
np.sqrt(mod.R.diagonal()[mod.iregn])
```




    array([0.0220886])



### Seasonal Component

Seasonal components represent cyclical or periodic behavior in the time series  - e.g. daily, weekly, or annual patterns. In PyBATS, seasonal components are defined by their period (e.g. $p = 7$ for a weekly trend on daily observation), and their harmonic components. The seasonal trends are defined through sine and cosine functions. Each harmonic component involves $2$ parameters, so there should never be more than $p/2$ harmonic components.

For example, if the period is $p=7$ (defined by setting `seasPeriods=[7]`) then a fully parameterized seasonal component has harmonic components `seasHarmComponents = [1,2,3]`.

If there is an annual trend on daily data, then the period is $p=365$. However, using all possible harmonic components, `seasHarmComponents = [1,2,...,182]`, is far too many parameters to learn. It is much more common to use the first several harmonic components, such as `seasHarmComponents = [1,2]`. The $r^{th}$ harmonic component has a cycle legth of $p/r$. So in the example of an annual pattern, the first harmonic component will have a cycle length of $365/2 = 182.5$, representing a semi-annual cycle, and the second harmonic component will have a length of $365/4$, for a quarterly cycle.

For more details, refer to Chapter 8.6 in Bayesian Forecasting and Dynamic Models by West and Harrison **XXX PROVIDE BETTER REFERENCE/LINK HERE**.

To see this in code, we'll load in some simulated daily sales data:

```python
from pybats_nbdev.shared import load_sales_example2
data = load_sales_example2()
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
      <th>Price</th>
      <th>Promotion</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-06-01</th>
      <td>15.0</td>
      <td>1.11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-06-02</th>
      <td>13.0</td>
      <td>2.19</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-06-03</th>
      <td>6.0</td>
      <td>0.23</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2014-06-04</th>
      <td>2.0</td>
      <td>-0.05</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2014-06-05</th>
      <td>6.0</td>
      <td>-0.14</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The simulated dataset contains daily sales of an item from June 1, 2014 to June 1, 2018. 

- The Price column represents percent change in price from the moving average, so it's centered at 0.
- Promotion is a 0-1 indicator for a specific sale or promotion on the item.

Before we run an analysis, we again need to specify a few arguments

```python
prior_length = 21   # Number of days of data used to set prior
k = 1               # Forecast horizon
rho = 0.5           # Random effect discount factor to increase variance of forecast distribution
forecast_samps = 1000  # Number of forecast samples to draw
forecast_start = pd.to_datetime('2018-01-01') # Date to start forecasting
forecast_end = pd.to_datetime('2018-06-01')   # Date to stop forecasting
Y = data['Sales'].values
X = data[['Price', 'Promotion']].values
```

And most importantly, we need to specify that the retail sales will have a seasonality component with a $7-$day period:

```python
seasPeriods=[7]
seasHarmComponents = [[1,2,3]]
```

```python
mod, samples = analysis(Y, X,
                        k, forecast_start, forecast_end, nsamps=forecast_samps,
                        family='poisson',
                        seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                        prior_length=prior_length, dates=data.index,
                        rho=rho,
                        ret = ['model', 'forecast'])
```

    beginning forecasting


We can visualize the forecasts, and instantly see how the weekly seasonality has effected them:

```python
plot_length = 30
data_1step = data.loc[forecast_end-pd.DateOffset(30):forecast_end]
samples_1step = samples[:,-31:,0]
fig, ax = plt.subplots(1,1)
ax = plot_data_forecast(fig, ax,
                        data_1step.Sales,
                        median(samples_1step),
                        samples_1step,
                        data_1step.index,
                        credible_interval=75)
```


![png](docs/images/output_46_0.png)


### Holidays and Special Events

PyBATS provides a method to specify known holidays, special events, and other outliers in the data. This will automatically add in a special indicator (dummy) variable to the regression component in order to account for this special behavior.

This can be useful for two purposes. Primarily, adding in a known, repeated, special event can improve forecasts. A secondary effect is that the indicator variable will help 'protect' the other coefficients in the model from overreacting to outliers in the data.

To demonstrate, let's repeat the analysis of simulated retail sales from above, while including a holiday effect:

```python
holidays = USFederalHolidayCalendar.rules

mod, samples = analysis(Y, X,
                        k, forecast_start, forecast_end, nsamps=forecast_samps,
                        family='poisson',
                        seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                        prior_length=prior_length, dates=data.index,
                        holidays=holidays,
                        rho=rho,
                        ret = ['model', 'forecast'])
```

    beginning forecasting


We've just used the standard holidays in the US Federal Calendar, but you can specify your own using this class in *Pandas*: **XXX ADD IN A LINK TO THE PANDAS DATE-TIME HOLIDAY CLASS**.

```python
holidays
```




    [Holiday: New Years Day (month=1, day=1, observance=<function nearest_workday at 0x7fc3616cf320>),
     Holiday: Martin Luther King Jr. Day (month=1, day=1, offset=<DateOffset: weekday=MO(+3)>),
     Holiday: Presidents Day (month=2, day=1, offset=<DateOffset: weekday=MO(+3)>),
     Holiday: Memorial Day (month=5, day=31, offset=<DateOffset: weekday=MO(-1)>),
     Holiday: July 4th (month=7, day=4, observance=<function nearest_workday at 0x7fc3616cf320>),
     Holiday: Labor Day (month=9, day=1, offset=<DateOffset: weekday=MO(+1)>),
     Holiday: Columbus Day (month=10, day=1, offset=<DateOffset: weekday=MO(+2)>),
     Holiday: Veterans Day (month=11, day=11, observance=<function nearest_workday at 0x7fc3616cf320>),
     Holiday: Thanksgiving (month=11, day=1, offset=<DateOffset: weekday=TH(+4)>),
     Holiday: Christmas (month=12, day=25, observance=<function nearest_workday at 0x7fc3616cf320>)]



Look again at the plot of forecasts above. A number of the observations fall outside of the $95\%$ credible intervals. One of those is on May 28, 2018 - Memorial Day. How does it look now?

```python
plot_length = 30
data_1step = data.loc[forecast_end-pd.DateOffset(30):forecast_end]
samples_1step = samples[:,-31:,0]
fig, ax = plt.subplots(1,1)
ax = plot_data_forecast(fig, ax,
                        data_1step.Sales,
                        median(samples_1step),
                        samples_1step,
                        data_1step.index,
                        credible_interval=75)
```


![png](docs/images/output_54_0.png)


By adding in these holidays, the indicator variable changed both the point forecast and the forecast variance for observations on these specific days. The forecast for Memorial day is now spot on, along with added uncertainty, knowing that the sales could have spiked even higher.

## Discount Factors

Discount factors are model hyperparameters that have to be specified before an analysis begins. They control the rate that coefficients are allowed to change, and have a very simple interpretation. If the discount factor $\delta=0.98$, then we *discount* the old information about a coefficient by adding $100\% - 98\% = 2\%$ more uncertainty (variance). This enables the coefficient to learn more quickly from *new* observations in the time series. If all discount factors are set to $\delta=1$, then there is no discounting, and the models become standard generalized linear models.

There is a trade-off to lowering the discount factors. While it allows the coefficients to change more over time, it will increase the uncertainty in the forecasts, and can make the coefficients too sensitive to noise in the data.

PyBATS has built-in some safety measures that prevents the variance from growing too large. This makes the models significantly more robust to poorly specified, and too low, discount factors. Even so, most discount factors should be between $0.9$ and $1$. For rarely observed predictors - like holidays - it's typically best to leave the discount factor at $1$.

PyBATS allows discount factors to be set separately for each component, and even for each individual coefficient if desired.

```python
deltrend = 0.98 # Discount factor on the trend component
delregn = 0.98 # Discount factor on the regression component
delseas = 0.98 # Discount factor on the seasonal component
delhol = 1 # Discount factor on the holiday component
rho = .3 # Random effect discount factor to increase variance of forecast distribution

mod, samples = analysis(Y, X,
                        k, forecast_start, forecast_end, nsamps=forecast_samps,
                        family='poisson',
                        seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                        prior_length=prior_length, dates=data.index,
                        holidays=holidays,
                        rho=rho,
                        deltrend = deltrend,
                        delregn=delregn,
                        delseas=delseas,
                        delhol=delhol,
                        ret = ['model', 'forecast'])
```

    beginning forecasting


The default discount factors in PyBATS are fairly high. We've set them to $\delta=0.98$ for the trend, regression, and seasonal components. This will allow those coefficient to adapt more rapidly. The holiday component is left at $\delta=1$.

We also changed the parameter $\rho=0.3$. This is a special discount factor which is used only to increase the *forecast* uncertainty. This parameter is different than the other discount factor, and its range can go anywhere. The smaller $\rho$ is, the wider the forecast interval. It can even be set higher than $1$, if the forecast intervals are too wide.

```python
plot_length = 30
data_1step = data.loc[forecast_end-pd.DateOffset(30):forecast_end]
samples_1step = samples[:,-31:,0]
fig, ax = plt.subplots(1,1)
ax = plot_data_forecast(fig, ax,
                        data_1step.Sales,
                        median(samples_1step),
                        samples_1step,
                        data_1step.index,
                        credible_interval=75)
ax.set_ylim([0, 25])
```




    (0.0, 25.0)




![png](docs/images/output_60_1.png)


## In-Depth Sales Forecasting Example
