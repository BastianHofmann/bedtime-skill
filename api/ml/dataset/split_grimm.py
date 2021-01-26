import csv

with open('grimms_fairytales.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in reader:
        if line_count > 0:
            num = str(line_count-1).rjust(2, '0')

            f = open(f'./grimm/grimm_{num}.txt', 'w')
            f.write(row[2])
            f.close()

        line_count += 1
