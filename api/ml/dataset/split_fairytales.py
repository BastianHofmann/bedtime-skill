with open('./merged_clean.txt') as f:
    num_files = 0
    num_newlines = 0
    data = ''

    for line in f:
        data += line

        if len(line) <= 1:
            num_newlines += 1
        else:
            num_newlines = 0

        if num_newlines >= 4:
            with open('./dataset/fairytales_' + str(num_files).rjust(3, '0') + '.txt', 'w') as out_f:
                out_f.write(data.strip())
                out_f.close()

            data = ''
            num_newlines = 0
            num_files += 1
