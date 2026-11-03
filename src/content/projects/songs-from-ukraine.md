---
title: Songs from Ukraine
url: /songsfromukraine.pdf
date: 2022-06-01
---

I scraped several hundred gigabytes of social media posts and analyzed the
music found in them. Videos were scraped from the Telegram social media
platform. Music metadata was retrieved from the videos using Shazam.

Videos were retrieved over the course of months. The scraping process was
scheduled using SystemD units on a linux server. CSV metrics on the data were
updated asynchronously via file changes. The code for the scraper and data
processing was written in Python.
