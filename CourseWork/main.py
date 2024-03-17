import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import ttest_ind


def show_distribution(df, parameter, parameter_name):
    plt.figure(figsize=(9, 6))

    sns.distplot(df[parameter], hist=True, kde=True,
                 color='blue',
                 hist_kws={'edgecolor': 'black'},
                 label=parameter_name)

    plt.title(f'Distribution of parameter - {parameter_name}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    plt.show()


def show_scatter(df, parameter_x, parameter_name_x, parameter_y, parameter_name_y):
    sns.scatterplot(x=parameter_x, y=parameter_y, data=df, color='blue', alpha=0.5, label=parameter_name_x)

    plt.xlabel(parameter_name_x)
    plt.ylabel(parameter_name_y)
    plt.title('Scatter Plot')
    plt.grid(True)
    plt.legend()
    plt.show()


def show_box_plot(df, parameter, parameter_name):
    sns.boxplot(y=parameter, data=df)

    plt.xlabel(parameter_name)
    plt.title('Box Plot')
    plt.grid(True)
    plt.show()


def calculate_median(df, df_yes, df_no, parameter_name):
    median_all = df[parameter_name].median()
    median_yes = df_yes[parameter_name].median()
    median_no = df_no[parameter_name].median()
    print(f'median_all of parameter {parameter_name} = {median_all}')
    print(f'median_yes of parameter {parameter_name} = {median_yes}')
    print(f'median_no of parameter {parameter_name} = {median_no}')
    print('\n')
    return median_all


def calculate_average_value_and_standard_deviation(df, df_yes, df_no, parameter_name):
    average_value_all = df[parameter_name].mean()
    standard_deviation_all = df[parameter_name].std()
    average_value_yes = df_yes[parameter_name].mean()
    standard_deviation_yes = df_yes[parameter_name].std()
    average_value_no = df_no[parameter_name].mean()
    standard_deviation_no = df_no[parameter_name].std()
    print(f'average_value_all of parameter {parameter_name} = {average_value_all}')
    print(f'standard_deviation_all of parameter {parameter_name} = {standard_deviation_all}')
    print(f'average_value_yes of parameter {parameter_name} = {average_value_yes}')
    print(f'standard_deviation_yes of parameter {parameter_name} = {standard_deviation_yes}')
    print(f'average_value_no of parameter {parameter_name} = {average_value_no}')
    print(f'standard_deviation_no of parameter {parameter_name} = {standard_deviation_no}')
    print('\n')


def evaluate_class_differences(df_yes, df_no, parameter_name):
    t_statistic, p_value = ttest_ind(df_yes[parameter_name], df_no[parameter_name])

    print("Значение t-статистики:", t_statistic)
    print("p-значение:", p_value)
    # Определим уровень статистической значимости (обычно 0.05)
    alpha = 0.05
    # Проверим, является ли полученное p-значение меньше уровня значимости
    if p_value < alpha:
        print(f"Различия в классах по параметру {parameter_name} статистически значимы.")
    else:
        print(f"Нет статистически значимых различий в классах по параметру {parameter_name}.")
    print('\n')


def main():
    file_path = 'D:\\leti\\8sem\\статистика\\StatisticsCode\\Resources\\Employee.csv'

    df = pd.read_csv(file_path)
    df_yes = df.loc[df[df.columns[-1]] == 1]
    df_no = df.loc[df[df.columns[-1]] == 0]

    age = 'Age'
    joining_year = 'JoiningYear'
    leave_or_not = 'LeaveOrNot'

    show_distribution(df, age, age)
    show_distribution(df, joining_year, joining_year)

    age_median = calculate_median(df, df_yes, df_no, age)
    joining_year_median = calculate_median(df, df_yes, df_no, joining_year)

    age_greater_than_median = df[df[age] > age_median]
    age_less_than_median = df[df[age] < age_median]
    show_distribution(age_greater_than_median, age, 'AgeGreaterThanMedian')
    show_distribution(age_less_than_median, age, 'AgeLessThanMedian')

    joining_year_greater_than_median = df[df[joining_year] > joining_year_median]
    joining_year_less_than_median = df[df[joining_year] < joining_year_median]
    show_distribution(joining_year_greater_than_median, joining_year, 'JoiningYearGreaterThanMedian')
    show_distribution(joining_year_less_than_median, joining_year, 'JoiningYearLessThanMedian')

    calculate_average_value_and_standard_deviation(df, df_yes, df_no, age)
    calculate_average_value_and_standard_deviation(df, df_yes, df_no, joining_year)

    evaluate_class_differences(df_yes, df_no, age)
    evaluate_class_differences(df_yes, df_no, joining_year)

    show_scatter(age_greater_than_median, age, 'AgeGreaterThanMedian', leave_or_not, leave_or_not)
    show_scatter(age_less_than_median, age, 'AgeLessThanMedian', leave_or_not, leave_or_not)
    show_scatter(joining_year_greater_than_median, joining_year, 'JoiningYearGreaterThanMedian', leave_or_not, leave_or_not)
    show_scatter(joining_year_less_than_median, joining_year, 'JoiningYearLessThanMedian', leave_or_not, leave_or_not)

    show_box_plot(age_greater_than_median, age, 'AgeGreaterThanMedian')
    show_box_plot(age_less_than_median, age, 'AgeLessThanMedian')
    show_box_plot(joining_year_greater_than_median, joining_year, 'JoiningYearGreaterThanMedian')
    show_box_plot(joining_year_less_than_median, joining_year, 'JoiningYearLessThanMedian')


if __name__ == "__main__":
    main()