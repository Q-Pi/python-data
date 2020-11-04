CREATE DATABASE marketing;

CREATE TABLE customer (
  id INT PRIMARY KEY,
  age INT,
  job VARCHAR,
  marital VARCHAR,
  education VARCHAR,
  default_ BOOLEAN,
  balance INT,
  housing BOOLEAN,
  loan BOOLEAN,
  contact VARCHAR,
  day INT,
  month VARCHAR,
  duration INT,
  campaign INT,
  pdays INT,
  previous INT,
  poutcome VARCHAR,
  subscribed BOOLEAN
);