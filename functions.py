import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from proforma import proforma
import numpy as np

sns.set_style("darkgrid")

# scenarios = [1, 2, 3, 4, 5, 6]
# labels = ['Actual', 'Naive model', 'Mean model', 'Cycle model', 'Projection model', 'Re-scale model']
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#808080', 'black', '#000080']


def plot_bar(data, labels, xlabel, ylabel='Internal Rate of Return (IRR)', ylim=None, rotation=None):
  """
  Plot a bar chart of the results based on various expectation models.

  Parameters
  ----------
  scenarios: list
    A list of scenario numbers to be plotted
  data: list
    A list of IRR values calculated based on various expectation mechanisms.
  xlabel: str
    The lebel for the x-axis.
  ylabel: str
    The label for the y-axis.
  ylim: list
    Range of values for y-axis.
  rotation: int
    The rotation of the x-axis labels.
  """
    
  x_values = labels
  y_values = data
  data = {'x': x_values, 'y':y_values}
  df = pd.DataFrame(data)
  df['y'] = df['y'].apply(lambda x: round(x, 3))
  ax = sns.barplot(x='x', y='y', data=df, palette=colors[:len(y_values)+1])
  # Add numbers on top of bars
  for index, row in df.iterrows():
    plt.annotate(str(row['y']), xy=(index, row['y']), ha='center', va='bottom')
  if ylim is not None:
    ax.set(ylim=ylim)
  plt.gca().set_xlabel(xlabel)
  plt.gca().set_ylabel(ylabel)
  plt.xticks(rotation=rotation)
  plt.tight_layout()

def plot_line(scenarios, data, labels, xlabel, ylabel='Internal Rate of Return (IRR)', xlim=None, ylim=None, rotation=0, legsize=10):
  """
  Plot a line chart of the trends for various expectation models.

  Parameters
  ----------
  scenarios: list
    A list of scenario numbers to be plotted
  data: pd.DataFrame
    A dataframe of the trends for various models of expectation formation, returned by 'expectation_formation' function.
  xlabel: str
    The lebel for the x-axis.
  ylabel: str
    The label for the y-axis.
  xlim: list
    Range of values for x-axis.
  ylim: list
    Range of values for y-axis.
  rotation: int
    The rotation of the x-axis labels.
  legsize: int
    Size of the legend font.
  labels: list
    List of labels for data sets.
  """
  if data.index.name == 'REF_DATE' and type(data.index.values[0]) == str:
    data['DATE'] = pd.to_datetime(data.index.values)
    data.set_index('DATE', inplace=True)
  for i in range(len(scenarios)):
    plt.plot(data[scenarios[i]], label=labels[i], color=colors[i])
  plt.xticks(rotation=rotation)
  plt.gca().set_xlabel(xlabel)
  plt.gca().set_ylabel(ylabel)
  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)
  plt.legend(fontsize=legsize, frameon=True)
  plt.tight_layout()

def plot_hist(data, xlabel='Internal Rate of Return (IRR)', ylabel='Frequency', labels=None, legsize=10):
  """
  Plot a histogram of data.

  Parameters
  ----------
  data: list
    A list of results produced as a result of Monte-Carlo simulation experiments, returned by 'montecarlo_simulation' function.
  xlabel: str
    The lebel for the x-axis.
  ylabel: str
    The label for the y-axis.
  legsize: int
    Size of the legend font.
  labels: list
    List of labels for data sets.
  """
  
  if type(data[0]) is list:
    for i in range(len(data)):
      plt.hist(data[i], label=labels[i], alpha=0.4)
  else:
    plt.hist(data)
  plt.gca().set_xlabel(xlabel)
  plt.gca().set_ylabel(ylabel)
  plt.legend(fontsize=legsize, frameon=True)
  
def expectation_formation(sheet_name,
                          scenarios,
                          expectation,
                          construction_cost_data_df,
                          vacancy_data_df,
                          absorption_data_df,
                          rent_data_df,
                          sales_data_df,
                          interest_rate_data_df,
                          sheet_path='/content/drive/My Drive/Mitacs_Fall2022/Proforma analysis/input_data/proforma_inputs_analysis.xlsx',
                          cc_expectation=False, vr_expectation=False, rp_expectation=False, i_expectation=False
                          ):
  """
  Create a dataset of expectation formation results for various models of expectation formation.

  Parameters
  ----------
  sheet_name: srt
    Name of the sheet from the input xlsx file.
  scenarios: list
    A list of scenario numbers to be plotted
  expectation: str
    Notation for the expectation factor. 'cc' for construction costs; 'vr' for vacancy rates; 'rp' for rental prices.
  construction_cost_data_df: pd.DataFrame
    A dataframe of input data for construction costs.
  vacancy_data_df: pd.DataFrame
    A dataframe of input data for vacancy rates.
  absorption_data_df: pd.DataFrame
    A dataframe of input data for market absorption rates.
  rent_data_df: pd.DataFrame
    A dataframe of input data for rental prices.
  sales_data_df: pd.DataFrame
    A dataframe of input data for sales prices.
  interest_rate_data_df: pd.DataFrame
    A dataframe of input data for interest rates.
  sheet_path: str
    Path to the input xlsx file.
  cc_expectation: boolean
    If True, the construction costs are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the construction costs are considered to be constant over the project's liftime.
  vr_expectation: boolean
    If True, the vacancy rates are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the vacancy rate is considered to be constant over the project's liftime.
  rp_expectation: boolean
    If True, the rental prices are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the rental price is considered to be constant over the project's liftime.
  i_expectation: boolean
    If True, the interest rates are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the interest rates is considered to be constant over the project's liftime.
  """
  
  proforma_inputs_df_cc = pd.read_excel(sheet_path, sheet_name=sheet_name, index_col=0)
  results = pd.DataFrame([])
  proforma_inputs_df = proforma_inputs_df_cc.copy()
  sample_proforma = proforma(proforma_inputs_df, 
                          construction_cost_data_df=construction_cost_data_df, construction_cost_data_type='index', cc_price_base=275, 
                          vacancy_data_df=vacancy_data_df,
                          absorption_data_df=absorption_data_df,
                          rent_data_df=rent_data_df,
                          sales_data_df=sales_data_df,
                          interest_rate_data_df=interest_rate_data_df)
  sample_proforma.run(scenarios, cc_expectation=cc_expectation, vr_expectation=vr_expectation, rp_expectation=rp_expectation, i_expectation=i_expectation)
  for i in range(len(scenarios)):
    if proforma_inputs_df[scenarios[i]]['{}_model'.format(expectation)] == "Actual":
        expectation_df = eval('sample_proforma.{}_expectation_updated_table_{}[["REF_DATE", "VALUE"]].copy()'.format(expectation, scenarios[i]))
        expectation_df.rename(columns={'VALUE': scenarios[i]}, inplace=True)
        expectation_df.set_index("REF_DATE", inplace=True)
        results = pd.concat([results, expectation_df], axis=1)
    else:
        expectation_df = eval('sample_proforma.{}_estimation_period_updated_table_{}[["REF_DATE", "VALUE"]].copy()'.format(expectation, scenarios[i]))
        expectation_df.rename(columns={'VALUE': scenarios[i]}, inplace=True)
        expectation_df.set_index("REF_DATE", inplace=True)
        results = pd.concat([results, expectation_df], axis=1)
        results.iloc[len(results)-len(expectation_df)-1, results.columns.get_loc(scenarios[i])] = results.iloc[len(results)-len(expectation_df)-1][1]
  return results

def run_numerical_sensitivity(variable, value_range, scenarios,
                              proforma_inputs_df,
                              construction_cost_data_df,
                              vacancy_data_df,
                              absorption_data_df,
                              rent_data_df,
                              sales_data_df,
                              interest_rate_data_df,
                              sheet_path='/content/drive/My Drive/Mitacs_Fall2022/Proforma analysis/input_data/proforma_inputs_analysis.xlsx',
                              cc_expectation=False, vr_expectation=False, rp_expectation=False, i_expectation=False
                              ):
  """
  Run experiments with a variable within its plausible range of values.

  Parameters
  ----------
  variable: str
    The notation of the variable intended for the sensitivity analysis.
  value_range: list
    A list of values to be assigned to the variable. 
  scenarios: list
    A list of scenario numbers to be plotted
  proforma_inputs_df: pd.DataFrame
    A dataframe of input data to the proforma.
  vacancy_data_df: pd.DataFrame
    A dataframe of input data for vacancy rates.
  absorption_data_df: pd.DataFrame
    A dataframe of input data for market absorption rates.
  rent_data_df: pd.DataFrame
    A dataframe of input data for rental prices.
  sales_data_df: pd.DataFrame
    A dataframe of input data for sales prices.
  interest_rate_data_df: pd.DataFrame
    A dataframe of input data for interest rates.
  sheet_path: str
    Path to the input xlsx file.
  cc_expectation: boolean
    If True, the construction costs are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the construction costs are considered to be constant over the project's liftime.
  vr_expectation: boolean
    If True, the vacancy rates are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the vacancy rate is considered to be constant over the project's liftime.
  rp_expectation: boolean
    If True, the rental prices are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the rental price is considered to be constant over the project's liftime.
  i_expectation: boolean
    If True, the interest rates are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the interest rates is considered to be constant over the project's liftime.
  """
  
  results = pd.DataFrame([])
  for i in value_range:
    proforma_inputs_df.loc[variable] = i
    sample_proforma = proforma(proforma_inputs_df, 
                          construction_cost_data_df=construction_cost_data_df, construction_cost_data_type='index', cc_price_base=275, 
                          vacancy_data_df=vacancy_data_df,
                          absorption_data_df=absorption_data_df,
                          rent_data_df=rent_data_df,
                          sales_data_df=sales_data_df,
                          interest_rate_data_df=interest_rate_data_df)
    
    _results = sample_proforma.run(scenarios, cc_expectation=cc_expectation, vr_expectation=vr_expectation, rp_expectation=rp_expectation, i_expectation=i_expectation)
    _results_dict = {}
    for scenario_number in scenarios:
      _results_dict[scenario_number] = [_results[scenario_number-1]]
    results = pd.concat([results, pd.DataFrame(_results_dict, index=[i])])
  return results

def montecarlo_simulation(iterations, variables, scenario_number,
                              proforma_inputs_df,
                              construction_cost_data_df,
                              vacancy_data_df,
                              absorption_data_df,
                              rent_data_df,
                              sales_data_df,
                              interest_rate_data_df,
                              cc_expectation=False, vr_expectation=False, rp_expectation=False, i_expectation=False):
  """
  Conduct Monte-Carlo simulation for the number of specified interations and return the results in a list.

  Parameters
  ----------
  interations: int
    Number of iterations for the Monte Carlo simulation.
  variables: dict
    A dictionary with the variables with uncertainty as keys and their corresponding distribution function and its parameters as values (example: {'i': ['uniform', 0.01, 0.06]}).
  scenario_number: int
    The scenario number in the input sheet which is used to produce the results.
  proforma_inputs_df: pd.DataFrame
    A dataframe of input data for the proforma.
  construction_cost_data_df: pd.DataFrame
    A dataframe of input data for construction costs.
  vacancy_data_df: pd.DataFrame
    A dataframe of input data for vacancy rates.
  absorption_data_df: pd.DataFrame
    A dataframe of input data for market absorption rates.
  rent_data_df: pd.DataFrame
    A dataframe of input data for rental prices.
  sales_data_df: pd.DataFrame
    A dataframe of input data for sales prices.
  interest_rate_data_df: pd.DataFrame
    A dataframe of input data for interest rates.
  cc_expectation: boolean
    If True, the construction costs are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the construction costs are considered to be constant over the project's liftime.
  vr_expectation: boolean
    If True, the vacancy rates are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the vacancy rate is considered to be constant over the project's liftime.
  rp_expectation: boolean
    If True, the rental prices are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the rental price is considered to be constant over the project's liftime.
  i_expectation: boolean
    If True, the interest rates are estimated based on the expectation mechanims specified in the same scenario. 
    If False, the interest rates is considered to be constant over the project's liftime.
  """
  
  results = []
  for i in range(iterations):
    for variable in variables:
      proforma_inputs_df[scenario_number][variable] = eval('np.random.{}({})'.format(variables[variable][0], str(variables[variable][1:])[1:-1]))
    sample_proforma = proforma(proforma_inputs_df, 
                              construction_cost_data_df=construction_cost_data_df, construction_cost_data_type='index', cc_price_base=275, 
                              vacancy_data_df=vacancy_data_df,
                              absorption_data_df=absorption_data_df,
                              rent_data_df=rent_data_df,
                              sales_data_df=sales_data_df,
                              interest_rate_data_df=interest_rate_data_df)
    _result = sample_proforma.run(scenario_number, cc_expectation=cc_expectation, vr_expectation=vr_expectation, rp_expectation=rp_expectation, i_expectation=i_expectation)
    results.append(_result)
  return results

def required_iterations(results, error=0.01):
  """
  Estimate the number of required iterations for the Monte Carlo simulation based on the approach proposed by Banks (2005).

  Parameters
  ----------
  resukts: list
    A list containing the results of the initial experiments (usually eight experiments).
  error: float
    The expected error terms as a percentage of the mean value of the results of the initial experiments.
  """
  
  # R > ((t_alpha/2, R-1:2.36462 * S_0)/(epsilon))**2
  return ((2.36462*np.std(results))/(error*np.mean(results)))**2


def run_joint_numerical_sensitivity(variables_grid, scenario_number, proforma_inputs_df, 
                                    construction_cost_data_df, 
                                    vacancy_data_df,
                                    absorption_data_df,
                                    rent_data_df,
                                    sales_data_df,
                                    interest_rate_data_df,
                                    cc_expectation=False, vr_expectation=False, rp_expectation=False, i_expectation=False):
  results_df = pd.DataFrame()
  variable_1 = list(variables_grid.keys())[0]
  variable_2 = list(variables_grid.keys())[1]
  for i in variables_grid[variable_1]:
    for pi in variables_grid[variable_2]:
      proforma_inputs_df[scenario_number][variable_1] = i
      proforma_inputs_df[scenario_number][variable_2] = pi
      test_proforma = proforma(proforma_inputs_df,
                          construction_cost_data_df=construction_cost_data_df, construction_cost_data_type='index', cc_price_base=275,
                          vacancy_data_df=vacancy_rates_data_df,
                          absorption_data_df=absorption_rate_data_df,
                          rent_data_df=rent_data_df,
                          sales_data_df=sales_data_df,
                          interest_rate_data_df=interest_rate_data_df)
      irr = test_proforma.run(scenario_number, cc_expectation=cc_expectation, vr_expectation=vr_expectation, rp_expectation=rp_expectation, i_expectation=i_expectation)
      results_df.at[i, pi] = irr
  return results_df

def plot_heatmap(data, x_label, y_label, z_label='Internal Rate of Return (IRR)', cmap='coolwarm', vmin=None, vmax=None):
  fig, ax = plt.subplots()
  heatmap = sns.heatmap(data, annot=True, vmin=vmin, vmax=vmax, cmap=cmap)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  # Format the x-axis tick labels as prices with separators
  heatmap.set_xticklabels(['{:,.1f}'.format(x/1e6).format(x) for x in data.columns])
  colorbar = heatmap.collections[0].colorbar
  colorbar.set_label(z_label)
  
  