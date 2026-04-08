# OpenLIFU Example Database

This directory represents an example of the current database structure
prescribed in `../src/openlifu/db`. 

## Users Login Credentials

Note that the User information stored in in `users/` contains password hashes,
and so for testing purposes, the text form of the passwords for each `.json`
file storing a User are listed below (in `username`/`password` format):

complex\_user/complex\_user.json:
- `complex_user`/`complex_user_hash`

example\_admin/example\_admin.json:
- `example_admin`/`example_admin_hash`

example\_operator/example\_operator.json:
- `example_operator`/`example_operator_hash`

example\_restricted/example\_restricted.json:
- `example_restricted`/`example_restricted_hash`
