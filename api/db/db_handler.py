import json
import re
from datetime import date
from random import choice

import mysql.connector as mysql

from .db_config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER


class DBHandler:

    def __init__(self):
        self.db = mysql.connect(
            host = DB_HOST,
            user = DB_USER,
            passwd = DB_PASS,
            database= DB_NAME)

        cursor = self.db.cursor()
        # how to use json in database https://www.sitepoint.com/use-json-data-fields-mysql-databases/
        create_table_models = "CREATE TABLE IF NOT EXISTS models \
            (model_id CHAR(3) NOT NULL PRIMARY KEY, class_name VARCHAR(50), \
            creation TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, parameters JSON DEFAULT NULL)"
        create_table_predictions = "CREATE TABLE IF NOT EXISTS predictions \
            (id INT NOT NULL PRIMARY KEY AUTO_INCREMENT, model_id CHAR(3) NOT NULL,\
            keyword1 VARCHAR(255) NOT NULL, keyword2 VARCHAR(30), keyword3 VARCHAR(30), output TEXT,\
                FULLTEXT (keyword1, keyword2, keyword3))"

        cursor.execute(create_table_models)
        cursor.execute(create_table_predictions)

        cursor.close()

    def add_model(self, model_id: str, class_name: str, parameters: str = None) -> bool:
        """Add a model to the db

        :param model_id: 3 char unique id for the model and the parameters
        :type model_id: str
        :param class_name: the class name of the model e.g GPT2
        :type class_name: str
        :param parameters: Multiple parameters, which will be saved as json in db, defaults to None
        :type parameters: str in Json format, optional
        :return: If creation was successful
        :rtype: bool
        """
        cursor = self.db.cursor()

        # id id exists already or id has not the right format abort
        if len(model_id) != 3 or self.id_exist(model_id):
            return False

        if parameters:
            param_json = json.dumps(parameters)
            add_model = "INSERT INTO models (model_id, class_name, parameters) VALUES (%s, %s, %s)"
            values = (model_id, class_name, param_json)
        else:
            add_model = "INSERT INTO models (model_id, class_name) VALUES (%s, %s)"
            values = (model_id, class_name)

        cursor.execute(add_model, values)

        # commit changes to db
        self.db.commit()
        cursor.close()

        return cursor.rowcount == 1

    def add_prediction(self, model_id: str, keyword1: str, text: str, keyword2: str = None, keyword3: str = None) -> bool:
        """Add a prediction/text to the db

        :param model_id: 3 char unique id, has to exist in model table
        :type model_id: str
        :param keyword1: first keyword of the text, max len 255
        :type keyword1: str
        :param text: the generated output text
        :type text: str
        :param keyword2: optional second keyword, max len 50, defaults to None
        :type keyword2: str, optional
        :param keyword3: optional third keyword, max len 50, defaults to None
        :type keyword3: str, optional
        :return: True if successful creation
        :rtype: bool
        """
        cursor = self.db.cursor()

        # id id exists already or id has not the right format abort
        if (len(model_id) != 3 or not self.id_exist(model_id) or
            len(keyword1) > 255 or (keyword2 and len(keyword2) > 30) or (keyword3 and len(keyword3) > 30)):
            return False

        add_prediction = "INSERT INTO predictions (model_id, keyword1, keyword2, keyword3, output) VALUES (%s, %s, %s, %s, %s)"
        values = (model_id, keyword1, keyword2, keyword3, text)

        cursor.execute(add_prediction, values)

        # commit changes to db
        self.db.commit()
        cursor.close()

        return cursor.rowcount == 1

    def id_exist(self, model_id: str) -> bool:
        """Checks if model id is already used

        :param model_id: 3 char unique model id
        :type model_id: str
        :return: if the model id already exists
        :rtype: bool
        """
        cursor = self.db.cursor()

        if len(model_id) != 3:
            return False

        exist = "SELECT 1 FROM models WHERE model_id = %s"
        cursor.execute(exist, (model_id,))

        data = cursor.fetchone()
        return bool(data)

    def get_model(self, model_id: str):
        """Return the model settings

        :param model_id: unique 3 char model id
        :type model_id: str
        :return: the class name, creation date of the model entry, parameters
        :rtype: str, str, str
        """
        cursor = self.db.cursor(dictionary=True)

        if len(model_id) != 3:
            return None

        get_model = "SELECT * FROM models WHERE model_id = %s"
        cursor.execute(get_model, (model_id,))

        data = cursor.fetchone()
        # parameters = json.loads(data["parameters"])
        return data["class_name"], data["creation"], data["parameters"]

    def create_unique_id(self) -> str:
        """Create a new unique 3 char id

        :return: new model id
        :rtype: str
        """
        allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        numbers = "0123456789"
        # prevent to get stuck
        for i in range(1, 1000):
            model_id = ""
            model_id += choice(allowed_chars)
            model_id += choice(allowed_chars)
            model_id += choice(allowed_chars+numbers)
            if not self.id_exist(model_id):
                return model_id
        return None

    def get_prediction(self, keyword1: str, keyword2: str = None, keyword3: str = None):
        """Get a generated text with matches to the keywords

        :param keyword1: required key in order to find text, more important than
        the other two keys
        :param keyword2: optional is also used for the search if not None
        :param keyword3: same as keyword2
        :return: All matches in a list dict fashion
        :rtype: list[dict]
        """
        # mysql fulltext indexing feature
        # https://www.w3resource.com/mysql/mysql-full-text-search-functions.php
        # https://dev.mysql.com/doc/refman/8.0/en/fulltext-fine-tuning.html min len
        # https://stackoverflow.com/questions/41177154/best-mysql-search-query-for-multiple-keywords-in-multiple-columns

        keywords = []
        if keyword1:
            keywords.append(keyword1)
        else:
            return None
        if keyword2:
            keywords.append(keyword2)
        if keyword3:
            keywords.append(keyword3)

        # Replace all non word characters with spaces
        # https://stackoverflow.com/questions/26507087/escape-string-for-use-in-mysql-fulltext-search
        non_word = re.compile('[^\w\d_]+')
        for keyword in keywords:
            keyword = non_word.sub("", keyword)
        # Replace characters-operators with spaces
        operator = re.compile('[+\-><\(\)~*\"@]+')
        for keyword in keywords:
            keyword = operator.sub("", keyword)
        # https://dev.mysql.com/doc/refman/8.0/en/fulltext-boolean.html
        # * find rows that contain words such as keyword
        search_condition = "+" #first keyword is required to match otherwise everything would be returned
        for keyword in keywords:
            search_condition += f"{keyword}* "

        # print(search_condition)
        # search in db
        cursor = self.db.cursor(dictionary=True)
        search = "SELECT keyword1, keyword2, keyword3, output, MATCH (keyword1, keyword2, keyword3)\
            AGAINST (%s IN BOOLEAN MODE) AS score FROM predictions \
            WHERE MATCH (keyword1, keyword2, keyword3) AGAINST (%s IN BOOLEAN MODE) ORDER BY score DESC"
        cursor.execute(search, (search_condition,search_condition))

        data = cursor.fetchall()
        return data

    def __del__(self):
        if self.db:
            self.db.close()
