#!/usr/bin/env bash

gunicorn server:app --log-file -
