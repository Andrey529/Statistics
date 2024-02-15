import csv


def task1(data):
    sum_num = 0
    for i in range(1, len(data)):
        sum_num += int(data[i][0])
    avg = sum_num / (len(data) - 1)
    print("task1:")
    print(avg)


def task2(data):
    male_avg_age = 0
    male_count = 0
    female_avg_age = 0
    female_count = 0
    for i in range(1, len(data)):
        if data[i][1] == "Male":
            male_avg_age += int(data[i][0])
            male_count += 1
        else:
            female_avg_age += int(data[i][0])
            female_count += 1
    male_avg_age /= male_count
    female_avg_age /= female_count
    print("task2")
    print(male_avg_age)
    print(female_avg_age)


def task3(data):
    obesities = []
    diabets = []
    ob_and_diab = []
    healthy = []
    for i in range(1, len(data)):
        if (data[i][-1] == "Positive") and (data[i][-2] == "Yes"):
            ob_and_diab.append(i)
        if (data[i][-1] == "Negative") and (data[i][-2] == "No"):
            healthy.append(i)
        if (data[i][-1] == "Positive") and (data[i][-2] == "No"):
            obesities.append(i)
        if (data[i][-1] == "Negative") and (data[i][-2] == "Yes"):
            diabets.append(i)
    print(ob_and_diab)
    print(healthy)
    print(obesities)
    print(diabets)

with open('diabetes_data_upload.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

print(data)
task1(data)
task2(data)
task3(data)
