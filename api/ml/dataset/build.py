import os
import re
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dataset_path = './stories/'
    data = []
    for file_path in os.listdir(dataset_path):
        with open(dataset_path + file_path) as infile:
            try:
                content = infile.read()
                clean_content = re.sub(r'[_|-|$|&|*|%|@|(|)|~]', '', content)
                data.append(re.sub(r'(\\n|\s)+', ' ', clean_content))
            except:
                print(file_path)
            infile.close()

    data = [story for story in data if len(story) > 100]

    # 80, 10, 10 split
    train, test = train_test_split(data, test_size=0.2)
    val, test = train_test_split(test, test_size=0.5)

    with open('./train_dataset.txt', 'w') as outfile:
        outfile.write("\n".join(train))
        outfile.close()

    with open('./val_dataset.txt', 'w') as outfile:
        outfile.write("\n".join(val))
        outfile.close()

    with open('./test_dataset.txt', 'w') as outfile:
        outfile.write("\n".join(test))
        outfile.close()
