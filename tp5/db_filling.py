import pandas as pd
df = pd.read_csv('bank-full.csv')
sqls = []
for i in range(0, df.shape[0]):
  sql = 'INSERT INTO customer VALUES (\'{}\','.format(i)
  for each in df.columns:
    sql += '\''
    sql += str(df.loc[[i]][each][i])
    sql += '\','
  sql = sql[:-1]
  sql += ')'
  sqls.append(sql)

import psycopg2
HOST = "localhost"
USER = "postgres"
PASSWORD = "password"
DATABASE = "marketing"
conn = psycopg2.connect("host={} dbname={} user={} password={}".format(HOST, DATABASE, USER, PASSWORD))
cur = conn.cursor()
for each in sqls:
	cur.execute(each)
conn.commit()
sql = "SELECT * FROM customer"
cur.execute(sql)
print(cur.fetchall())
conn.close()