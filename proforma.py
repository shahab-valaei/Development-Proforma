import pandas as pd
import numpy as np
import numpy_financial as npf
from sklearn.linear_model import LinearRegression
from arch import arch_model


class proforma():
  def __init__(self, proforma_inputs_df,
               construction_cost_data_df=None, construction_cost_data_type=None, cc_price_base=None, cc_price_models=None,
               vacancy_data_df=None,
               absorption_data_df=None,
               rent_data_df=None,
               sales_data_df=None,
               interest_rate_data_df=None):
    """
    The base class for a development project.

    Parameters
    ----------
    proforma_inputs_df: pd.DataFrame
        Table of inputs to the proforma
    construction_cost_data_df: pd.DataFrame
        Data for construction costs if available
    construction_cost_data_type: str
        Type of the data inputted for construction cost ["index" or "price"]
    cc_price_base: float
        Average construction cost per square foot for the reference period
    cc_price_models: dic
        A dictionary of price expectation models and the corresponding model parameter
    vacancy_data_df: pd.DataFrame
        Data for vacancies if available
    absorption_data_df: pd.DataFrame
        Data for absorption rates if available
    rent_data_df: pd.DataFrame
        Data for unit rental prices if available
    sales_data_df: pd.DataFrame
        Data for unit sales prices if available
    interest_rate_data_df: pd.DataFrame
        Data for interest rate if available
    """

    self.proforma_inputs_df = proforma_inputs_df
    self.construction_cost_data_df = construction_cost_data_df
    self.construction_cost_data_type = construction_cost_data_type
    self.cc_price_base = cc_price_base
    self.cc_observation_period_table = None
    self.vacancy_data_df = vacancy_data_df
    self.absorption_data_df = absorption_data_df
    self.rental_price_df = rent_data_df
    self.sales_price_df = sales_data_df
    self.interest_rate_df = interest_rate_data_df

  def _read_variables(self, scenario_number, list_of_variables):
    """
    Read the list of input variables from the specified scenario number.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input data sheet
    list_of_variables: list
        A list containing the notation of the variables
    """

    for variable in list_of_variables:
      exec("self.{} = self.proforma_inputs_df[{}]['{}']".format(variable, str(int(scenario_number)), variable))

  def _write_feature(self, scenario_number, feature_name, value):
    exec('self.{}_{} = {}'.format(feature_name, scenario_number, value))

  def land_table(self, scenario_number):
    """
    Create a table representing the characteristics of the land.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scneario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['A', 'L'])
    _land_df = pd.DataFrame({
        'value': [self.A, 43650*self.A, self.L, self.L/(43650*self.A)]
    }, index=['Land size', 'Total square feet', 'Total land costs', 'Land costs per square foot'])
    exec('self.land_table_{} = _land_df'.format(scenario_number))
    return _land_df

  def construction_cost_table(self, scenario_number, cc_expectation=False):
    """
    Create a table representing the details on the estimated construction costs.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    cc_expectation: Boolean
        If True, the construction costs are estimated based on the expectation mechanims specified in the same scenario.
        If False, the construction costs are considered to be constant over the project's liftime.
    """

    scneario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['R', 'PT', 'tau', 'N', 'Ns', 'Nl', 'A_a', 'cc_model', 'N_cc_model', 'C_cc_model'])
    if cc_expectation:
      _cc_expectation_df = self.construction_cost_expectation_table(scenario_number)
      _cc_expectation_df_updated = self.update_expectation_table(scenario_number, 'cc', self.cc_model, self.N_cc_model, self.C_cc_model)
      _total_construction_costs = 0
      for i in range(int(self.N_cc_model), len(_cc_expectation_df_updated)):
        _total_construction_costs += _cc_expectation_df_updated['Units constructed'][i+1] * _cc_expectation_df_updated['VALUE'][i+1] * self.A_a
    else:
      _total_construction_costs = self.tau*self.A_a*self.N
    _construction_cost_df = pd.DataFrame({
        'value': [self.PT, 'Rental' if self.R else 'Sales', self.tau, self.Ns, self.Nl, self.N, self.A_a, self.tau*self.A_a, _total_construction_costs]
    }, index=['Price type', 'Unit type', 'Construction cost per square foot', 'Bachelor and 1 bedroom units', '2+ bedroom units', 'Total units', 'Avg unit size', 'Construction cost per unit', 'Total units construction cost'])
    exec('self.construction_cost_table_{} = _construction_cost_df'.format(scenario_number))
    return _construction_cost_df

  def soft_costs_table(self, scenario_number):
    """
    Create a table representing the soft development costs.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scneario_number = str(int(scenario_number))
    _construction_cost_df = eval('self.construction_cost_table_{}'.format(scenario_number))
    self._read_variables(scenario_number, ['zeta', 'P', 'Ds', 'Dl'])
    _soft_costs_df = pd.DataFrame({
        'value': [self.Ds, self.Ds*_construction_cost_df['value']['Bachelor and 1 bedroom units'], self.Dl, self.Dl*_construction_cost_df['value']['2+ bedroom units'],
                  self.zeta*(self.Ds*_construction_cost_df['value']['Bachelor and 1 bedroom units']+self.Dl*_construction_cost_df['value']['2+ bedroom units']),
                  self.P, (1+self.zeta)*(self.Ds*_construction_cost_df['value']['Bachelor and 1 bedroom units']+self.Dl*_construction_cost_df['value']['2+ bedroom units'])+self.P]
    }, index=['DCs per small units', 'Total DCs for small units', 'DCs per large units', 'Total DCs for large units', 'Other soft fees',  'Planning', 'Total soft costs'])
    exec('self.soft_costs_table_{} = _soft_costs_df'.format(scenario_number))
    return _soft_costs_df

  def total_development_costs_table(self, scenario_number):
    """
    Create a table representing the total development costs.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scneario_number = str(int(scenario_number))
    _construction_cost_df = eval('self.construction_cost_table_{}'.format(scenario_number))
    _soft_cost_df = eval('self.soft_costs_table_{}'.format(scenario_number))
    self._read_variables(scenario_number, ['D_f', 'L'])
    TPC = _construction_cost_df['value']['Total units construction cost']+_soft_cost_df['value']['Total soft costs']
    _total_development_costs_df = pd.DataFrame({
        'value': [TPC, self.D_f, TPC*self.D_f, TPC*(1+self.D_f)+self.L]
    }, index=['Total project costs', 'Developer fee rate', 'Developer fee', 'Total development costs'])
    exec('self.total_development_costs_table_{} = _total_development_costs_df'.format(scenario_number))
    return _total_development_costs_df

  def income_and_expenses_table(self, scenario_number):
    """
    Create a table representing the assumptions on the source of income and expenses.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scneario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['R', 'pi', 'alpha', 'omega', 'i_r', 'i_e', 'B_r', 'theta'])
    if self.R:
      _income_and_expenses_df = pd.DataFrame({
          'value': [self.pi, self.alpha, self.omega, self.i_r, self.i_e, self.theta]
      }, index=['Monthly rent', 'Monthly operating expenses', 'Vacancies', 'Annual rent inflations', 'Anual expense inflations', 'Annual tax rate'])
    else:
      _income_and_expenses_df = pd.DataFrame({
          'value': [self.pi, self.B_r, self.theta]
      }, index=['Unit sales price', 'Brokers fee', 'Annual tax rate'])
    exec('self.income_and_expenses_table_{} = _income_and_expenses_df'.format(scenario_number))
    return _income_and_expenses_df

  def loan_terms_table(self, scenario_number):
    """
    Create a table representing the loan terms and project sales assumptions.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scenario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['Y', 'i', 'c', 'C_e'])
    _loan_terms_df = pd.DataFrame({
        'value': [self.Y, self.Y*12, self.i, self.c, self.C_e]
    }, index=['Loan years', 'Loan months', 'Annual interest rate', 'Cap rate', 'Closing costs'])
    exec('self.loan_terms_table_{} = _loan_terms_df'.format(scenario_number))
    return _loan_terms_df

  def funding_sources_table(self, scenario_number):
    """
    Create a table representing the sources of funds, including the equity and debt.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scneario_number = str(int(scenario_number))
    _total_development_costs_df = eval('self.total_development_costs_table_{}'.format(scenario_number))
    _loan_terms_df = eval('self.loan_terms_table_{}'.format(scenario_number))
    self._read_variables(scenario_number, ['eta'])
    TDC = _total_development_costs_df['value']['Total development costs']
    PMT = npf.pmt(_loan_terms_df['value']['Annual interest rate']/12, _loan_terms_df['value']['Loan months'], TDC*(1-self.eta))
    _funding_sources_df = pd.DataFrame({
        'value': [self.eta, self.eta*TDC, TDC*(1-self.eta), PMT, PMT*12, (TDC*(1-self.eta))/TDC, self.eta*TDC+TDC*(1-self.eta)]
    }, index=['Equity percent', 'Equity amount', 'Loan amount', 'Monthly payment', 'Annual payment', 'Loan to development cost ratio', 'Total sources'])
    exec('self.funding_sources_table_{} = _funding_sources_df'.format(scenario_number))
    return _funding_sources_df

  def loan_schedule_table(self, scenario_number, i_expectation):
    """
    Create a table representing the loan payments according to the loan amount and a dynamic interst rate.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scneario_number = str(int(scenario_number))
    _funding_sources_df = eval('self.funding_sources_table_{}'.format(scenario_number))
    self._read_variables(scenario_number, ['i_model', 'N_i_model', 'C_i_model', 'Y', 'R', 'i'])
    _i_expectation_df = self.interest_rate_expectation_table(scenario_number)
    _i_expectation_df = self.update_expectation_table(scenario_number, 'i', self.i_model, self.N_i_model, self.C_i_model)
    _i_estimation_df = eval('self.i_estimation_period_updated_table_{}'.format(scenario_number))
    if not i_expectation:
      _i_estimation_df['VALUE'] = self.i
    _i_estimation_df = _i_estimation_df.reset_index()
    _i_estimation_df.index = _i_estimation_df.index + 1
    # Columns for the _loan_schedule_df are 'PER_NUM', 'REF_DATE', 'VALUE', 'Monthly payment', 'Interest', 'Principal pmt', 'Principal balance'
    _i_estimation_df['Monthly payment'] = [np.NaN for i in range(len(_i_estimation_df))]
    _i_estimation_df['Interest'] = [np.NaN for i in range(len(_i_estimation_df))]
    _i_estimation_df['Principal pmt'] = [np.NaN for i in range(len(_i_estimation_df))]
    _i_estimation_df['Principal balance'] = [np.NaN for i in range(len(_i_estimation_df))]

    new_row = pd.DataFrame(np.nan, index=[0], columns=_i_estimation_df.columns)   # Create a new row filled with NaN values
    _i_estimation_df = pd.concat([new_row, _i_estimation_df]).reset_index(drop=True)  # Concatenate the new row with the original dataframe
    _i_estimation_df['Principal balance'][0] = _funding_sources_df['value']['Loan amount']
    if self.R:
      pmt_start_month = 37    # Assumption for start of loan repayments for a rental project
    else:
      pmt_start_month = 13    # Assumption for start of loan repayments for a sales project
    for j in range(pmt_start_month-1):
      _i_estimation_df['Interest'][j+1] = _i_estimation_df['Principal balance'][j] * (_i_estimation_df['VALUE'][j+1]/12)
      _i_estimation_df['Principal balance'][j+1] = _i_estimation_df['Principal balance'][j] + _i_estimation_df['Interest'][j+1]
    for j in range(pmt_start_month-1, len(_i_estimation_df)-1):
      _i_estimation_df['Monthly payment'][j+1] = npf.pmt(_i_estimation_df['VALUE'][j+1]/12, self.Y*12-(j-(pmt_start_month-1)), -_i_estimation_df['Principal balance'][j])
      _i_estimation_df['Interest'][j+1] = _i_estimation_df['Principal balance'][j] * (_i_estimation_df['VALUE'][j+1]/12)
      _i_estimation_df['Principal pmt'][j+1] = _i_estimation_df['Monthly payment'][j+1] - _i_estimation_df['Interest'][j+1]
      _i_estimation_df['Principal balance'][j+1] = _i_estimation_df['Principal balance'][j] - _i_estimation_df['Principal pmt'][j+1]
    exec('self.loan_schedule_table_{} = _i_estimation_df'.format(scenario_number))
    return _i_estimation_df

  def cash_flow_analysis(self, scenario_number, cc_expectation=False, vr_expectation=False, rp_expectation=False, i_expectation=False):
    """
    Create a table representing the cashflow of the project revenues and costs.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    cc_expectation: boolean
        If True, the construction costs are estimated based on the expectation mechanims specified in the same scenario.
        If False, the construction costs are considered to be constant over the project's liftime.
    vr_expectation: boolean
        If True, the vacancy rates are estimated based on the expectation mechanims specified in the same scenario.
        If False, the vacancy rate is considered to be constant over the project's liftime.
    rp_expectation: boolean
        If True, the unit prices are estimated based on the expectation mechanims specified in the same scenario.
        If False, the unit prices are considered to be constant over the project's liftime.
    i_expectation: boolean
        If True, the interest rates are estimated based on the expectation mechanims specified in the same scenario.
        If False, the interest rates is considered to be constant over the project's liftime.
    """

    # Load the data
    scenario_number = str(int(scenario_number))
    _land_df = eval('self.land_table_{}'.format(scenario_number))
    _construction_cost_df = eval('self.construction_cost_table_{}'.format(scenario_number))
    _soft_costs_df = eval('self.soft_costs_table_{}'.format(scenario_number))
    _total_development_costs_df = eval('self.total_development_costs_table_{}'.format(scenario_number))
    _income_and_expenses_df = eval('self.income_and_expenses_table_{}'.format(scenario_number))
    _loan_terms_df = eval('self.loan_terms_table_{}'.format(scenario_number))
    _funding_sources_table = eval('self.funding_sources_table_{}'.format(scenario_number))
    self._read_variables(scenario_number, ['R', 'N', 'kappa', 'L', 'P',
                                           'cc_model', 'N_cc_model', 'C_cc_model',
                                           'vr_model', 'N_vr_model', 'C_vr_model',
                                           'rp_model', 'N_rp_model', 'C_rp_model',
                                           'A_a', 'tau', 'theta', 'pi', 'omega', 'c', 'C_e', 'd', 'B_r', 'S_L', 'S_B', 'U_c', 'E', 'M'])

    # Construction period
    self.d = int(self.d)
    _construction_period = int(1 + np.ceil(self.N/(self.kappa*4)))
    _construction_period_dict = {}
    for i in range (_construction_period+int(self.d)):
      i = str(int(i+1))
      _empty_list = []
      for j in range(11 if self.R else 17):
        _empty_list.append(np.nan)
      _construction_period_dict['Year{}'.format(i)] = _empty_list

    if self.R:
      _construction_period_df = pd.DataFrame(_construction_period_dict, index=[
          'Units constructed',
          # Revenues
          'Equity investment',
          'Developer fee',
          # Expenses
          'Land acquisition',
          'Planning and entitlements',
          'Building construction',
          'Fees and other soft costs',
          'Taxes',
          # Summation
          'Net operative income',
          'Annual loan payment',
          'Cash flow total'
      ])
    else:
      _construction_period_df = pd.DataFrame(_construction_period_dict, index=[
        'Units constructed',
        'Units closed',
        # Revenues
        'Gross sales',
        'Less brokers fee',
        'Net sales revenue',
        'Equity investment',
        'Developer fee',
        # Expenses
        'Land acquisition',
        'Planning and entitlements',
        'Other initial costs',
        'Building construction',
        'Fees and other soft costs',
        'Management and overhead',
        'Taxes',
        # Summation
        'Net operative income',
        'Annual loan payment',
        'Cash flow total'
      ])

    for i in range(2+self.d, _construction_period+self.d):
      _construction_period_df.at['Units constructed', 'Year{}'.format(i)] = 4*self.kappa
    _construction_period_df.at['Units constructed', 'Year{}'.format(str(int(_construction_period+self.d)))] = self.N-self.kappa*4*(_construction_period+self.d-2)

    ## Construction period revenues
    _construction_period_df.at['Equity investment', 'Year{}'.format(str(int(_construction_period+self.d)))] = _funding_sources_table['value']['Equity amount']
    if not self.R:
      _construction_period_df.at['Developer fee', 'Year{}'.format(str(int(_construction_period+self.d)))] = _total_development_costs_df['value']['Developer fee']
      if vr_expectation:
        _vr_expectation_df = self.absorption_rates_expectation_table(scenario_number)
        _vr_expectation_df = self.update_expectation_table(scenario_number, 'vr', self.vr_model, self.N_vr_model, self.C_vr_model)
      if rp_expectation:
        _rp_expectation_df = self.sales_expectation_table(scenario_number)
        _rp_expectation_df = self.update_expectation_table(scenario_number, 'rp', self.rp_model, self.N_rp_model, self.C_rp_model)
      for i in range(1, _construction_period+int(self.d+1)):
        if vr_expectation:
          units_closed_per_year = 0
          for j in range(4):
            units_closed_per_year += _vr_expectation_df['Units constructed'][int(j+1+self.N_vr_model+4*(i-1))] * _vr_expectation_df['VALUE'][int(j+1+self.N_vr_model+4*(i-1))]
        else:
          try:
            units_closed_per_year = _construction_period_df['Year{}'.format(i)]['Units constructed'] * self.omega
          except ValueError:
            units_closed_per_year = 0
        _construction_period_df.at['Units closed', 'Year{}'.format(i)] = units_closed_per_year
        gross_sales_per_year = 0
        if rp_expectation and vr_expectation:
          for j in range(4):
            gross_sales_per_year += _rp_expectation_df['VALUE'][int(j+1+self.N_rp_model+4*(i-1))] * _rp_expectation_df['Units constructed'][int(j+1+self.N_rp_model+4*(i-1))] * _vr_expectation_df['VALUE'][int(j+1+self.N_vr_model+4*(i-1))]
        elif rp_expectation:
          for j in range(4):
            gross_sales_per_year += _rp_expectation_df['VALUE'][int(j+1+self.N_rp_model+4*(i-1))] * _rp_expectation_df['Units constructed'][int(j+1+self.N_rp_model+4*(i-1))] * self.omega
        else:
          gross_sales_per_year = _construction_period_df['Year{}'.format(i)]['Units closed'] * self.pi
        if np.isnan(gross_sales_per_year):
          gross_sales_per_year = 0
        _construction_period_df.at['Gross sales', 'Year{}'.format(i)] = gross_sales_per_year
        _construction_period_df.at['Less brokers fee', 'Year{}'.format(i)] = _construction_period_df['Year{}'.format(i)]['Gross sales'] * self.B_r
        _construction_period_df.at['Net sales revenue', 'Year{}'.format(i)] = _construction_period_df['Year{}'.format(i)]['Gross sales'] * (1 - self.B_r)

    ## Construction period expenses
    _construction_period_df.at['Land acquisition', 'Year1'] = self.L
    _construction_period_df.at['Planning and entitlements', 'Year{}'.format(int(1+self.d))] = _soft_costs_df['value']['Planning']
    if not self.R:
      _construction_period_df.at['Other initial costs', 'Year1'] = self.E
      for i in range(1, _construction_period+int(self.d+1)):
        _construction_period_df.at['Management and overhead', 'Year{}'.format(i)] = _construction_period_df['Year{}'.format(i)]['Gross sales'] * self.M
    if cc_expectation:
      _cc_expectation_df = self.construction_cost_expectation_table(scenario_number)
      _cc_expectation_df = self.update_expectation_table(scenario_number, 'cc', self.cc_model, self.N_cc_model, self.C_cc_model)
    for i in range(int(2+self.d), _construction_period+int(self.d+1)):
      if cc_expectation:
        construction_cost_per_year = 0
        for j in range(4):
          construction_cost_per_year += _cc_expectation_df['Units constructed'][int(j+1+self.N_cc_model+4*(i-1))] * _cc_expectation_df['VALUE'][int(j+1+self.N_cc_model+4*(i-1))] * self.A_a
      else:
        construction_cost_per_year = _construction_period_df['Year{}'.format(i)]['Units constructed'] * self.A_a * self.tau
      _construction_period_df.at['Building construction', 'Year{}'.format(i)] = construction_cost_per_year
    _construction_period_df.at['Fees and other soft costs', 'Year{}'.format(int(1+self.d))] = _soft_costs_df['value']['Total DCs for small units']+_soft_costs_df['value']['Total DCs for large units']+_soft_costs_df['value']['Other soft fees']
    for i in range(1, _construction_period+int(self.d+1)):
      _construction_period_df.at['Taxes', 'Year{}'.format(i)] = self.theta * self.L

    ## Construction period summation
      if self.R:
        _construction_period_df.at['Net operative income', 'Year{}'.format(i)] = -_construction_period_df['Year{}'.format(i)][['Land acquisition', 'Planning and entitlements', 'Building construction', 'Fees and other soft costs', 'Taxes']].sum()
      else:
        _construction_period_df.at['Net operative income', 'Year{}'.format(i)] = _construction_period_df['Year{}'.format(i)]['Net sales revenue']-_construction_period_df['Year{}'.format(i)][['Land acquisition', 'Planning and entitlements', 'Building construction', 'Fees and other soft costs', 'Taxes', 'Management and overhead', 'Other initial costs']].sum()
    _i_expectation_df = self.loan_schedule_table(scenario_number, i_expectation)
    for i in range(int(1+self.d), _construction_period+int(self.d+1)):
      loan_pmt_per_year = 0
      for j in range(12):
        loan_pmt_per_year += _i_expectation_df['Monthly payment'][j+1+12*(i-1)]
      if np.isnan(loan_pmt_per_year):
        loan_pmt_per_year = 0
      _construction_period_df.at['Annual loan payment', 'Year{}'.format(i)] = -loan_pmt_per_year
    for i in range(1, _construction_period+int(self.d+1)):
      nop_and_loan = _construction_period_df['Year{}'.format(i)][['Net operative income', 'Annual loan payment']].sum()
      if np.isnan(nop_and_loan):
        nop_and_loan = 0
      other_costs = _construction_period_df['Year{}'.format(i)]['Equity investment'] - _construction_period_df['Year{}'.format(i)]['Developer fee']
      if np.isnan(other_costs):
        other_costs = 0
      _construction_period_df.at['Cash flow total', 'Year{}'.format(i)] = nop_and_loan + other_costs

    exec('self.construction_period_table_{} = _construction_period_df'.format(scenario_number))

    if not self.R:
      new_indexes = ['Units constructed', 'Units closed', 'Gross sales', 'Less brokers fee', 'Net sales revenue', 'Equity investment', 'Developer fee', 'Land acquisition',
                     'Planning and entitlements', 'Other initial costs', 'Building construction', 'Fees and other soft costs', 'Management and overhead', 'Taxes', 'Net operative income',
                     'Annual loan payment', 'Cash flow total']
      _construction_period_df = _construction_period_df.reindex(new_indexes)
      exec('self.cash_flow_table_{} = _construction_period_df'.format(scenario_number))
      return _construction_period_df

    # Operation period
    _operation_start = int(_construction_period + self.d + 1)
    _operation_end = int(_operation_start + 9)

    _operation_period_dict = {}
    for i in range (_operation_start, _operation_end+1):
      _empty_list = []
      for j in range(10):
        _empty_list.append(np.nan)
      _operation_period_dict['Year{}'.format(i)] = _empty_list

    _operation_period_df = pd.DataFrame(_operation_period_dict, index=[
        # Revenues
        'Gross potential rent',
        'Less vacancies',
        'Effective gross income',
        'Sales in last year',
        'Developer fee',
        # Expenses
        'Operating expenses',
        'Closing costs in last year',
        # Summation
        'Net operative income',
        'Annual loan payment',
        'Cash flow total'
    ])

    ## Operation period revenues, expenses, and summation
    if rp_expectation:
      _rp_expectation_df = self.rent_expectation_table(scenario_number)
      _rp_expectation_updated_df = self.update_expectation_table(scenario_number, 'rp', self.rp_model, self.N_rp_model, self.C_rp_model)
    if vr_expectation:
      _vr_expectation_df = self.vacancy_rates_expectation_table(scenario_number)
      _vr_expectation_updated_df = self.update_expectation_table(scenario_number, 'vr', self.vr_model, self.N_vr_model, self.C_vr_model)
    for i in range(_operation_start, _operation_end+1):
      if rp_expectation:
        _operation_period_df.at['Gross potential rent', 'Year{}'.format(i)] = 12*_rp_expectation_updated_df['VALUE'][i+self.N_rp_model]*self.N*((1+self.i_r)**(i-_operation_start))
      else:
        _operation_period_df.at['Gross potential rent', 'Year{}'.format(i)] = 12*self.pi*self.N*((1+self.i_r)**(i-_operation_start))
      if vr_expectation:
        _operation_period_df.at['Less vacancies', 'Year{}'.format(i)] = 12*self.pi*self.N*_vr_expectation_updated_df['VALUE'][i+self.N_vr_model]*((1+self.i_r)**(i-_operation_start))
      else:
        _operation_period_df.at['Less vacancies', 'Year{}'.format(i)] = 12*self.pi*self.N*self.omega*((1+self.i_r)**(i-_operation_start))
      effective_gross_income = _operation_period_df.at['Gross potential rent', 'Year{}'.format(i)] - _operation_period_df.at['Less vacancies', 'Year{}'.format(i)]
      _operation_period_df.at['Effective gross income', 'Year{}'.format(i)] = effective_gross_income
      _operation_period_df.at['Operating expenses', 'Year{}'.format(i)] = effective_gross_income*self.alpha
      _operation_period_df.at['Net operative income', 'Year{}'.format(i)] = _operation_period_df['Year{}'.format(i)]['Effective gross income'] - _operation_period_df.at['Operating expenses', 'Year{}'.format(i)]
      loan_pmt_per_year = 0
      for j in range(12):
        loan_pmt_per_year += _i_expectation_df['Monthly payment'][j+1+12*(i-1)]
        _construction_period_df.at['Annual loan payment', 'Year{}'.format(i)] = -loan_pmt_per_year
    _operation_period_df.at['Developer fee', 'Year{}'.format(_operation_start)] = _total_development_costs_df['value']['Developer fee']
    if self.U_c:
      _operation_period_df.at['Sales in last year', 'Year{}'.format(_operation_end)] = _operation_period_df['Year{}'.format(_operation_end)]['Net operative income']/self.c
    else:
      _operation_period_df.at['Sales in last year', 'Year{}'.format(_operation_end)] = self.S_B + self.S_L
    _operation_period_df.at['Closing costs in last year', 'Year{}'.format(_operation_end)] = _operation_period_df['Year{}'.format(_operation_end)]['Sales in last year'] * self.C_e
    for i in range (_operation_start, _operation_end+1):
      rents_and_expenses = _operation_period_df['Year{}'.format(i)][['Net operative income', 'Annual loan payment']].sum()
      if np.isnan(rents_and_expenses):
        rents_and_expenses = 0
      sales_and_developer = _operation_period_df['Year{}'.format(i)][['Sales in last year', 'Developer fee']].sum()
      if np.isnan(sales_and_developer):
        sales_and_developer = 0
      closing = _operation_period_df['Year{}'.format(i)]['Closing costs in last year']
      if np.isnan(closing):
        closing = 0
      other_cash_flows = sales_and_developer - closing

      if np.isnan(other_cash_flows):
        other_cash_flows = 0
      _operation_period_df.at['Cash flow total', 'Year{}'.format(i)] = rents_and_expenses + other_cash_flows

    exec('self.operation_period_table_{} = _operation_period_df'.format(scenario_number))

    _cash_flow_df = _construction_period_df.join(_operation_period_df, how='outer')
    if self.R:
      new_indexes = ['Units constructed', 'Gross potential rent', 'Less vacancies', 'Effective gross income', 'Sales in last year', 'Equity investment', 'Developer fee', 'Land acquisition',
                     'Planning and entitlements', 'Building construction', 'Fees and other soft costs', 'Operating expenses', 'Closing costs in last year', 'Taxes', 'Net operative income',
                     'Annual loan payment', 'Cash flow total']
      _cash_flow_df = _cash_flow_df.reindex(new_indexes)
    exec('self.cash_flow_table_{} = _cash_flow_df'.format(scenario_number))

    return _cash_flow_df

  def calculate_irr(self, scenario_number):
    """
    Calculate the Internal Rate of Reuturn based on the cashflow of the project.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet.
    """

    _cash_flow_df = eval('self.cash_flow_table_{}'.format(scenario_number))
    irr = npf.irr(_cash_flow_df.loc['Cash flow total'])
    exec('self.irr_{} = irr'.format(scenario_number))
    return irr

  def calculate_npv(self, scenario_number, rate):
    """
    Calculate the Net Present Value based on the cashflow and a rate of ruturn.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet.
    rate: float
        The rate of return used to calculate the NPV.
    """

    _cash_flow_df = eval('self.cash_flow_table_{}'.format(scenario_number))
    npv = npf.npv(rate, _cash_flow_df.loc['Cash flow total'])
    return npv

  def run(self, scenario_numbers, cc_expectation=False, vr_expectation=False, rp_expectation=False, i_expectation=False):
    """
    Run the cashflow analysis for the proforma and return the estimated IRR or a list of estimated IRR values for scenarios.

    Parameters
    ----------
    scenario_numbers: int or list
        The scenario number(s) in the input sheet which is used to produce the table.
    cc_expectation: boolean
        If True, the construction costs are estimated based on the expectation mechanims specified in the same scenario.
        If False, the construction costs are considered to be constant over the project's liftime.
    vr_expectation: boolean
        If True, the vacancy rates are estimated based on the expectation mechanims specified in the same scenario.
        If False, the vacancy rate is considered to be constant over the project's liftime.
    rp_expectation: boolean
        If True, the rental prices are estimated based on the expectation mechanims specified in the same scenario.
        If False, the rental price is considered to be constant over the project's liftime.
    """

    if type(scenario_numbers) == type([]):
      _output_list = []
      for scenario_number in scenario_numbers:
        self.land_table(scenario_number)
        self.construction_cost_table(scenario_number, cc_expectation)
        self.soft_costs_table(scenario_number)
        self.total_development_costs_table(scenario_number)
        self.income_and_expenses_table(scenario_number)
        self.loan_terms_table(scenario_number)
        self.funding_sources_table(scenario_number)
        self.cash_flow_analysis(scenario_number, cc_expectation, vr_expectation, rp_expectation, i_expectation)
        _irr = self.calculate_irr(scenario_number)
        _output_list.append(_irr)
        print("Scenario {} completed . . .".format(scenario_number))
      return _output_list

    else:
      scenario_number = int(scenario_numbers)
      self.land_table(scenario_number)
      self.construction_cost_table(scenario_number, cc_expectation)
      self.soft_costs_table(scenario_number)
      self.total_development_costs_table(scenario_number)
      self.income_and_expenses_table(scenario_number)
      self.loan_terms_table(scenario_number)
      self.funding_sources_table(scenario_number)
      self.cash_flow_analysis(scenario_number, cc_expectation, vr_expectation, rp_expectation, i_expectation)
      _irr = self.calculate_irr(scenario_number)
      return _irr

  def construction_cost_expectation_table(self, scenario_number):
    """
    Create a table representing the expectation formation for construction costs based on the specified mechanism corresponding to the scenario.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scenario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['t_init', 'N_cc_model', 'kappa', 'N', 'd'])
    _construction_cost_data_df = self.construction_cost_data_df[['REF_DATE', 'VALUE']]
    if self.construction_cost_data_type == 'index':
      _construction_cost_data_df['VALUE'] = _construction_cost_data_df['VALUE']*(self.cc_price_base/100)

    # observation table
    _cc_observation_period_df = _construction_cost_data_df.copy()
    _start_date_estimation_index = int(_cc_observation_period_df[_cc_observation_period_df['REF_DATE'] == str(self.t_init)[:7]].index.values)
    exec("self._start_date_estimation_{} = _cc_observation_period_df['REF_DATE'][_start_date_estimation_index]".format(scenario_number))
    _end_date_estimation_index = int(_start_date_estimation_index+int((1 + np.ceil(self.N/(self.kappa*4)))*4-1)+4*self.d)
    exec("self._end_date_estimation_{} = _cc_observation_period_df['REF_DATE'][_end_date_estimation_index]".format(scenario_number))
    _end_date_observation_index = int(_start_date_estimation_index-1)
    exec("self._end_date_observation_{} = _cc_observation_period_df['REF_DATE'][_end_date_observation_index]".format(scenario_number))
    _start_date_observation_index = int(_end_date_observation_index-self.N_cc_model+1)
    exec("self._end_date_observation_{} = _cc_observation_period_df['REF_DATE'][_start_date_observation_index]".format(scenario_number))

    _cc_observation_period_df = _cc_observation_period_df[_start_date_observation_index:_end_date_observation_index+1]
    _cc_observation_period_df['PER_NUM'] = [i+1 for i in range(len(_cc_observation_period_df))]
    _cc_observation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.cc_observation_period_table_{} = _cc_observation_period_df'.format(scenario_number))

    # expectation table
    _cc_estimation_period_df = _construction_cost_data_df.copy()
    _cc_estimation_period_df = _cc_estimation_period_df[_start_date_estimation_index:_end_date_estimation_index+1]
    _cc_estimation_period_df['PER_NUM'] = [i for i in range(len(_cc_observation_period_df)+1, len(_cc_observation_period_df)+len(_cc_estimation_period_df)+1)]
    _cc_estimation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.cc_estimation_period_table_{} = _cc_estimation_period_df'.format(scenario_number))

    _cc_expectation_df = pd.concat([_cc_observation_period_df, _cc_estimation_period_df])

    _units_constructed = []
    for i in range(len(_cc_expectation_df)):
      if i < self.N_cc_model+4+4*self.d:
        _units_constructed.append(0)
      elif i == len(_cc_expectation_df)-1:
        _units_constructed.append(self.N%self.kappa)
      else:
        _units_constructed.append(self.kappa)
    _cc_expectation_df['Units constructed'] = _units_constructed

    exec('self.cc_expectation_table_{} = _cc_expectation_df'.format(scenario_number))

    return _cc_expectation_df

  def vacancy_rates_expectation_table(self, scenario_number):
    """
    Create a table representing the expectation formation for vacancy rates based on the specified mechanism corresponding to the scenario.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scenario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['t_init', 'N_vr_model', 'kappa', 'N', 'd'])
    _vacancy_data_df = self.vacancy_data_df[['REF_DATE', 'Total']].rename(columns={'Total': 'VALUE'})

    # observation table
    _vr_observation_period_df = _vacancy_data_df.copy()

    _end_observation_year = int(str(self.t_init)[:4])-1
    _end_observation_year_index = int(_vr_observation_period_df[_vr_observation_period_df['REF_DATE'] == _end_observation_year].index.values)
    _start_observation_year_index = int(_end_observation_year_index - self.N_vr_model + 1)
    _vr_observation_period_df = _vr_observation_period_df[_start_observation_year_index:_end_observation_year_index+1]
    _vr_observation_period_df['PER_NUM'] = [int(i+1) for i in range(len(_vr_observation_period_df))]
    _vr_observation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.vr_observation_period_table_{} = _vr_observation_period_df'.format(scenario_number))

    # expectation table
    _vr_estimation_period_df = _vacancy_data_df.copy()
    _start_estimation_year_index = _end_observation_year_index + 1
    _end_estimation_year_index = _start_estimation_year_index + int(1 + np.ceil(self.N/(self.kappa*4))) + int(self.d) + 9
    _vr_estimation_period_df = _vr_estimation_period_df[_start_estimation_year_index: _end_estimation_year_index+1]
    _vr_estimation_period_df['PER_NUM'] = [int(i+self.N_vr_model+1) for i in range(len(_vr_estimation_period_df))]
    _vr_estimation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.vr_estimation_period_table_{} = _vr_estimation_period_df'.format(scenario_number))

    _vr_expectation_df = pd.concat([_vr_observation_period_df, _vr_estimation_period_df])
    exec('self.vr_expectation_table_{} = _vr_expectation_df'.format(scenario_number))
    return _vr_expectation_df

  def absorption_rates_expectation_table(self, scenario_number):
    """
    Create a table representing the expectation formation for absorption rates based on the specified mechanism corresponding to the scenario.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scenario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['t_init', 'N_vr_model', 'kappa', 'N', 'd'])
    _absorption_rate_data_df = self.absorption_data_df[['REF_DATE', 'Total']].rename(columns={'Total': 'VALUE'})

    # observation table
    _vr_observation_period_df = _absorption_rate_data_df.copy()
    _start_date_estimation_index = int(_vr_observation_period_df[_vr_observation_period_df['REF_DATE'] == str(self.t_init)[:7]].index.values)
    exec("self._start_date_estimation_{} = _vr_observation_period_df['REF_DATE'][_start_date_estimation_index]".format(scenario_number))
    _end_date_estimation_index = int(_start_date_estimation_index+int((1 + np.ceil(self.N/(self.kappa*4)))*4-1)+4*self.d)
    exec("self._end_date_estimation_{} = _vr_observation_period_df['REF_DATE'][_end_date_estimation_index]".format(scenario_number))
    _end_date_observation_index = int(_start_date_estimation_index-1)
    exec("self._end_date_observation_{} = _vr_observation_period_df['REF_DATE'][_end_date_observation_index]".format(scenario_number))
    _start_date_observation_index = int(_end_date_observation_index-self.N_vr_model+1)
    exec("self._end_date_observation_{} = _vr_observation_period_df['REF_DATE'][_start_date_observation_index]".format(scenario_number))

    _vr_observation_period_df = _vr_observation_period_df[_start_date_observation_index:_end_date_observation_index+1]
    _vr_observation_period_df['PER_NUM'] = [i+1 for i in range(len(_vr_observation_period_df))]
    _vr_observation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.vr_observation_period_table_{} = _vr_observation_period_df'.format(scenario_number))

    # expectation table
    _vr_estimation_period_df = _absorption_rate_data_df.copy()
    _vr_estimation_period_df = _vr_estimation_period_df[_start_date_estimation_index:_end_date_estimation_index+1]
    _vr_estimation_period_df['PER_NUM'] = [i for i in range(len(_vr_observation_period_df)+1, len(_vr_observation_period_df)+len(_vr_estimation_period_df)+1)]
    _vr_estimation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.vr_estimation_period_table_{} = _vr_estimation_period_df'.format(scenario_number))

    _vr_expectation_df = pd.concat([_vr_observation_period_df, _vr_estimation_period_df])

    _units_constructed = []
    for i in range(len(_vr_expectation_df)):
      if i < self.N_vr_model+4+4*self.d:
        _units_constructed.append(0)
      elif i == len(_vr_expectation_df)-1:
        _units_constructed.append(self.N%self.kappa)
      else:
        _units_constructed.append(self.kappa)
    _vr_expectation_df['Units constructed'] = _units_constructed
    exec('self.vr_expectation_table_{} = _vr_expectation_df'.format(scenario_number))

    return _vr_expectation_df

  def rent_expectation_table(self, scenario_number):
    """
    Create a table representing the expectation formation for rental prices based on the specified mechanism corresponding to the scenario.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scenario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['t_init', 'N_rp_model', 'kappa', 'N'])
    _rp_data_df = self.rental_price_df[['REF_DATE', 'Total']].rename(columns={'Total': 'VALUE'})

    # observation table
    _rp_observation_period_df = _rp_data_df.copy()
    _end_observation_year = int(str(self.t_init)[:4])-1
    _end_observation_year_index = int(_rp_observation_period_df[_rp_observation_period_df['REF_DATE'] == _end_observation_year].index.values)
    _start_observation_year_index = int(_end_observation_year_index - self.N_rp_model + 1)
    _rp_observation_period_df = _rp_observation_period_df[_start_observation_year_index:_end_observation_year_index+1]
    _rp_observation_period_df['PER_NUM'] = [int(i+1) for i in range(len(_rp_observation_period_df))]
    _rp_observation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.rp_observation_period_table_{} = _rp_observation_period_df'.format(scenario_number))

    # expectation table
    _rp_estimation_period_df = _rp_data_df.copy()
    _start_estimation_year_index = _end_observation_year_index + 1
    _end_estimation_year_index = _start_estimation_year_index + int(1 + np.ceil(self.N/(self.kappa*4))) + int(self.d) + 9
    _rp_estimation_period_df = _rp_estimation_period_df[_start_estimation_year_index: _end_estimation_year_index+1]
    _rp_estimation_period_df['PER_NUM'] = [int(i+self.N_rp_model+1) for i in range(len(_rp_estimation_period_df))]
    _rp_estimation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.rp_estimation_period_table_{} = _rp_estimation_period_df'.format(scenario_number))

    _rp_expectation_df = pd.concat([_rp_observation_period_df, _rp_estimation_period_df])
    exec('self.rp_expectation_table_{} = _rp_expectation_df'.format(scenario_number))
    return _rp_expectation_df

  def sales_expectation_table(self, scenario_number):
    """
    Create a table representing the expectation formation for sales price based on the specified mechanism corresponding to the scenario.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scenario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['t_init', 'N_rp_model', 'kappa', 'N', 'd'])
    _sales_data_df = self.sales_price_df[['REF_DATE', 'Average']].rename(columns={'Average': 'VALUE'})

    # observation table
    _rp_observation_period_df = _sales_data_df.copy()
    _start_date_estimation_index = int(_rp_observation_period_df[_rp_observation_period_df['REF_DATE'] == str(self.t_init)[:7]].index.values)
    exec("self._start_date_estimation_{} = _rp_observation_period_df['REF_DATE'][_start_date_estimation_index]".format(scenario_number))
    _end_date_estimation_index = int(_start_date_estimation_index+int((1 + np.ceil(self.N/(self.kappa*4)))*4-1)+4*self.d)
    exec("self._end_date_estimation_{} = _rp_observation_period_df['REF_DATE'][_end_date_estimation_index]".format(scenario_number))
    _end_date_observation_index = int(_start_date_estimation_index-1)
    exec("self._end_date_observation_{} = _rp_observation_period_df['REF_DATE'][_end_date_observation_index]".format(scenario_number))
    _start_date_observation_index = int(_end_date_observation_index-self.N_rp_model+1)
    exec("self._end_date_observation_{} = _rp_observation_period_df['REF_DATE'][_start_date_observation_index]".format(scenario_number))

    _rp_observation_period_df = _rp_observation_period_df[_start_date_observation_index:_end_date_observation_index+1]
    _rp_observation_period_df['PER_NUM'] = [i+1 for i in range(len(_rp_observation_period_df))]
    _rp_observation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.rp_observation_period_table_{} = _rp_observation_period_df'.format(scenario_number))

    # expectation table
    _rp_estimation_period_df = _sales_data_df.copy()
    _rp_estimation_period_df = _rp_estimation_period_df[_start_date_estimation_index:_end_date_estimation_index+1]
    _rp_estimation_period_df['PER_NUM'] = [i for i in range(len(_rp_observation_period_df)+1, len(_rp_observation_period_df)+len(_rp_estimation_period_df)+1)]
    _rp_estimation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.rp_estimation_period_table_{} = _rp_estimation_period_df'.format(scenario_number))

    _rp_expectation_df = pd.concat([_rp_observation_period_df, _rp_estimation_period_df])

    _units_constructed = []
    for i in range(len(_rp_expectation_df)):
      if i < self.N_rp_model+4+4*self.d:
        _units_constructed.append(0)
      elif i == len(_rp_expectation_df)-1:
        _units_constructed.append(self.N%self.kappa)
      else:
        _units_constructed.append(self.kappa)
    _rp_expectation_df['Units constructed'] = _units_constructed

    exec('self.rp_expectation_table_{} = _rp_expectation_df'.format(scenario_number))

    return _rp_expectation_df

  def interest_rate_expectation_table(self, scenario_number):
    """
    Create a table representing the expectation formation for interest rates based on the specified mechanism corresponding to the scenario.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    """

    scenario_number = str(int(scenario_number))
    self._read_variables(scenario_number, ['t_init', 'N_i_model', 'kappa', 'N', 'd'])
    _interest_rate_data_df = self.interest_rate_df[['REF_DATE', 'Interest rate']].rename(columns={'Interest rate': 'VALUE'})

    # observation table
    _i_observation_period_df = _interest_rate_data_df.copy()
    _start_date_estimation_index = int(_i_observation_period_df[_i_observation_period_df['REF_DATE'] == str(self.t_init)[:7]].index.values)
    exec("self._start_date_estimation_{} = _i_observation_period_df['REF_DATE'][_start_date_estimation_index]".format(scenario_number))
    _end_date_estimation_index = int(_start_date_estimation_index+int((1 + np.ceil(self.N/(self.kappa*4)))*12-1)+12*self.d)
    exec("self._end_date_estimation_{} = _i_observation_period_df['REF_DATE'][_end_date_estimation_index]".format(scenario_number))
    _end_date_observation_index = int(_start_date_estimation_index-1)
    exec("self._end_date_observation_{} = _i_observation_period_df['REF_DATE'][_end_date_observation_index]".format(scenario_number))
    _start_date_observation_index = int(_end_date_observation_index-self.N_i_model+1)
    exec("self._end_date_observation_{} = _i_observation_period_df['REF_DATE'][_start_date_observation_index]".format(scenario_number))

    _i_observation_period_df = _i_observation_period_df[_start_date_observation_index:_end_date_observation_index+1]
    _i_observation_period_df['PER_NUM'] = [i+1 for i in range(len(_i_observation_period_df))]
    _i_observation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.i_observation_period_table_{} = _i_observation_period_df'.format(scenario_number))

    # expectation table
    _i_estimation_period_df = _interest_rate_data_df.copy()
    _i_estimation_period_df = _i_estimation_period_df[_start_date_estimation_index:_end_date_estimation_index+1]
    _i_estimation_period_df['PER_NUM'] = [i for i in range(len(_i_observation_period_df)+1, len(_i_observation_period_df)+len(_i_estimation_period_df)+1)]
    _i_estimation_period_df.set_index('PER_NUM', inplace=True)
    exec('self.i_estimation_period_table_{} = _i_estimation_period_df'.format(scenario_number))

    _i_expectation_df = pd.concat([_i_observation_period_df, _i_estimation_period_df])
    exec('self.i_expectation_table_{} = _i_expectation_df'.format(scenario_number))
    return _i_expectation_df

  def update_expectation_table(self, scenario_number, variable_notation,
                               expectation_model, N_expectation_model, C_expectation_model):
    """
    Update the expectation formation table by estimating the trends over the prediction period and return the predictions.

    Parameters
    ----------
    scenario_number: int
        The scenario number in the input sheet which is used to produce the table.
    variable_notation: str
        The notation for the expectation factor to be updated. 'cc' for construction costs; or 'vr' for vacancy rates; or 'rp' for rental prices.
    expectation_model: str
        The expectation formation model to update the expectation table.
    N_expectation_model: int
        Number of observation periods for the available data used to form the expectations over the prediction period.
    C_expectation_model: float
        Characteristic of the expectation formation model.
    """

    scneario_number = str(int(scenario_number))
    t_d = self._read_variables(scenario_number, ['t_d', 't_init'])
    _expectation_df = eval('self.{}_expectation_table_{}.copy()'.format(variable_notation, scenario_number))
    _estimation_period_df = eval('self.{}_estimation_period_table_{}.copy()'.format(variable_notation, scenario_number))
    if expectation_model == 'Actual':
      _estimation_period_updated_df = _estimation_period_df.copy()
      _expectation_updated_df = _expectation_df.copy()
    else:
      _estimation_period_updated_df, _expectation_updated_df = self.expectation_formation(variable_notation, _expectation_df, expectation_model, N_expectation_model, C_expectation_model, self.t_init, self.t_d)
    exec('self.{}_estimation_period_updated_table_{} = _estimation_period_updated_df'.format(variable_notation, scenario_number))
    exec('self.{}_expectation_updated_table_{} = _expectation_updated_df'.format(variable_notation, scenario_number))
    return _expectation_updated_df

  def expectation_formation(self, variable_notation, expectation_df, expectation_model, N_expectation_model, C_expectation_model, t_init, t_d):
    """
    Estimate the trends over the prediction period and return the predictions.

    Parameters
    ----------
    expectation_df: pd.DataFrame
        Table of the available data for the observation period (expectation table).
    expectation_model: str
        The expectation formation model to update the expectation table.
    N_expectation_model: int
        Number of observation periods for the available data used to form the expectations over the prediction period.
    C_expectation_model: float
        Characteristic of the expectation formation model.
    """

    _expectation_df = expectation_df.copy()
    _estimation_df = _expectation_df[int(N_expectation_model):]
    _estimation_period = int(len(_expectation_df) - N_expectation_model)
    i_start = _expectation_df[_expectation_df['REF_DATE'] == str(t_d)[:7]].index[0]
    i_end = _estimation_period + N_expectation_model + 1
    decision_delay = _expectation_df[_expectation_df['REF_DATE'] == str(t_d)[:7]].index[0] - _expectation_df[_expectation_df['REF_DATE'] == str(t_init)[:7]].index[0]
    if expectation_model == 'Naive model':
      for i in range(i_start, i_end):
        _estimation_df['VALUE'][i] = _expectation_df['VALUE'][i-1]
        _expectation_df['VALUE'][i] = _expectation_df['VALUE'][i-1]
    elif expectation_model == 'Mean model':
      for i in range(i_start, i_end):
        _estimation_df['VALUE'][i] = _expectation_df['VALUE'][int(i-C_expectation_model-1):int(i-1)].mean()
        _expectation_df['VALUE'][i] = _expectation_df['VALUE'][int(i-C_expectation_model-1):int(i-1)].mean()
    elif expectation_model == 'Cycle model':
      for i in range(i_start, i_end):
        _estimation_df['VALUE'][i] = _expectation_df['VALUE'][int(i-C_expectation_model)]
        _expectation_df['VALUE'][i] = _expectation_df['VALUE'][int(i-C_expectation_model)]
    elif expectation_model == 'Projection model':
      _expectation_df['PER_NUM'] = _expectation_df.index.values
      for i in range(i_start, i_end):
        x = _expectation_df['PER_NUM'][int(i-C_expectation_model-1):int(i-1)].to_numpy().reshape((-1, 1))
        y = _expectation_df['VALUE'][int(i-C_expectation_model-1):int(i-1)].to_numpy()
        model = LinearRegression().fit(x, y)
        _estimation_df['VALUE'][i] = model.predict([[_expectation_df['PER_NUM'][i]]])
        _expectation_df['VALUE'][i] = model.predict([[_expectation_df['PER_NUM'][i]]])
    elif expectation_model == 'Re-scale model':
      for i in range(i_start, i_end):
        _estimation_df['VALUE'][i] = C_expectation_model*_expectation_df['VALUE'][i-1]
        _expectation_df['VALUE'][i] = C_expectation_model*_expectation_df['VALUE'][i-1]
    elif expectation_model == 'Adaptive expectations':
      _expectation_df['Actual'] = _expectation_df['VALUE']
      _estimation_df['Actual'] = _estimation_df['VALUE']
      for i in range(2, i_start):
        _estimation_df['Actual'][i] = C_expectation_model*_expectation_df['VALUE'][i-1] + (1-C_expectation_model)*_expectation_df['Actual'][i-1]
        _expectation_df['Actual'][i] = C_expectation_model*_expectation_df['VALUE'][i-1] + (1-C_expectation_model)*_expectation_df['Actual'][i-1]
      last_observed_data = _expectation_df['Actual'][N_expectation_model+decision_delay]
      _estimation_df['VALUE'][i_start] = C_expectation_model*last_observed_data + (1-C_expectation_model)*_expectation_df['Actual'][i_start-1]
      _expectation_df['VALUE'][i_start] = C_expectation_model*last_observed_data + (1-C_expectation_model)*_expectation_df['Actual'][i_start-1]
      for i in range(i_start+1, i_end):
        _estimation_df['VALUE'][i] = C_expectation_model*last_observed_data + (1-C_expectation_model)*_expectation_df['VALUE'][i-1]
        _expectation_df['VALUE'][i] = C_expectation_model*last_observed_data + (1-C_expectation_model)*_expectation_df['VALUE'][i-1]
      _expectation_df = _expectation_df.drop('Actual', axis=1)
      _estimation_df = _estimation_df.drop('Actual', axis=1)
    elif expectation_model == 'Trend following':
      _expectation_df['Actual'] = _expectation_df['VALUE']
      for i in range(i_start, i_end):
        _estimation_df['VALUE'][i] = _expectation_df['VALUE'][i-1] + C_expectation_model*(_expectation_df['VALUE'][i-1]-_expectation_df['VALUE'][i-2])
        _expectation_df['VALUE'][i] = _expectation_df['VALUE'][i-1] + C_expectation_model*(_expectation_df['VALUE'][i-1]-_expectation_df['VALUE'][i-2])
      _expectation_df = _expectation_df.drop('Actual', axis=1)
    elif expectation_model == 'Anchor and adjustment':
      _expectation_df['Actual'] = _expectation_df['VALUE']
      for i in range(i_start, i_end):
        p_av = np.mean(_expectation_df['VALUE'][int(i-C_expectation_model-1):int(i-1)].to_numpy())
        _estimation_df['VALUE'][i] = (p_av+_expectation_df['VALUE'][i-1])/2 + (_expectation_df['VALUE'][i-1]-_expectation_df['VALUE'][i-2])
        _expectation_df['VALUE'][i] = (p_av+_expectation_df['VALUE'][i-1])/2 + (_expectation_df['VALUE'][i-1]-_expectation_df['VALUE'][i-2])
      _expectation_df = _expectation_df.drop('Actual', axis=1)
    elif expectation_model == 'garch':
      returns = _expectation_df['VALUE'][int(i_start-C_expectation_model-1):int(i_start-1)].pct_change().dropna()
      model = arch_model(returns, vol='Garch', p=1, q=1)
      model_fit = model.fit(disp='off')
      forecast = model_fit.forecast(start=0, horizon=len(_estimation_df))
      forecasted_returns = forecast.mean.iloc[-1] + np.random.normal(0, forecast.variance.iloc[-1] ** 0.5, len(_estimation_df))
      forecasted_data = _expectation_df['VALUE'][i_start-1] * (1 + forecasted_returns).cumprod()
      for i in range(i_start, i_end):
        _estimation_df['VALUE'][i] = forecasted_data[i-i_start]
        _expectation_df['VALUE'][i] = forecasted_data[i-i_start]
    if variable_notation == 'vr' or variable_notation == 'i':
      _estimation_df['VALUE'] = _estimation_df['VALUE'].clip(0, 1)
      _expectation_df['VALUE'] = _expectation_df['VALUE'].clip(0, 1)
    elif variable_notation == 'cc' or variable_notation == 'rp':
      _estimation_df['VALUE'] = _estimation_df['VALUE'].clip(0)
      _expectation_df['VALUE'] = _expectation_df['VALUE'].clip(0)
    return _estimation_df, _expectation_df