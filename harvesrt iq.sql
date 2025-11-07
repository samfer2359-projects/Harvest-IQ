create database harvestiq;

\c harvestiq

create table users(user_id serial primary key,username text not null unique,password text not null);