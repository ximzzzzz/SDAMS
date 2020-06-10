from mysql.connector import MySQLConnection, Error
from db_config import read_db_config
import pandas as pd

class GetSdamsDB:
    def __init__(self):
        self.dbconfig = read_db_config()
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = MySQLConnection(**self.dbconfig)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, e_type, e_val, tb):
        try:
            self.cursor.close()
            self.conn.close()
        except AttributeError:
            return True

    def getDataPD(self, sql):
        data = pd.read_sql(sql, con=self.conn)
        return data

    def runSQL_Insert(self, sql, args):
        self.cursor.execute(sql, args)
        self.conn.commit()

    def runSQL_Insert_ID(self, sql, args):
        self.cursor.execute(sql,args)
        lastId = self.cursor.lastrowid
        self.conn.commit()

        return lastId

    def runSQL_Update(self, sql, args):
        self.cursor.execute(sql, args)
        self.conn.commit()
                                                                              