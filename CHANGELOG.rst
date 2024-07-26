Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 0.3.0 (2024-07-26)
--------------------------

* Added: table preview for each table of a dataset
  on the datacard
* Added: ``audbcards.Dataset.segments``
  which returns the number of unique segments of a dataset
* Added: ``audbcards.Datasets.segment_durations``,
  which returns a list of all segment durations
* Added: ``audbcards.Datacard.segment_duration_distribution``
* Changed: don't show media examples
  for datasets that store,
  on average,
  more than 100 files per archive
* Changed: show video examples as video instead of audio
* Changed: depend on ``audeer>=2.2.0``
* Changed: depend on ``audiofile>=1.5.0``


Version 0.2.0 (2024-05-15)
--------------------------

* Added: ``audbcards.config.CACHE_ROOT``
  to configure the default cache root
* Added: store the result of ``audb.available()``
  in the sphinx extension
  to make it reusable
* Added: ``audbcards.Dataset.example_media``
* Added: ``cache_root`` argument to ``audbcards.Datacard``
* Added: support for Python 3.11
* Changed: speedup caching of ``audbcards.Dataset``
* Changed: cache resulting files
  of ``audbcards.Datacard.file_duration_distribution()``
  and ``audbcards.Datacard.player()``
* Changed: depend on ``audb>=1.7.0``
* Fixed: skip duration distribution plots
  for datasets
  that only contain files with the same duration
* Fixed: support ``|`` character
  in dataset description
* Fixed: remove ``audbcards.Dataset.prop``
  from API documentation
* Removed: ``audbcards.Datacard.example_media``,
  use ``audbcards.Dataset.example_media`` instead


Version 0.1.0 (2024-03-27)
--------------------------

* Added: initial release,
  including the classes
  ``audbcards.Datacard``
  and ``audbcards.Dataset``,
  and the ``audbcards.sphinx`` extension


.. _Keep a Changelog:
    https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning:
    https://semver.org/spec/v2.0.0.html
