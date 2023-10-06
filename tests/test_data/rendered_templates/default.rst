.. _db:

db
--

Created by H Wierstorf, C Geng, B E Abrougui

============= ======================
version       1.0.0
license       `CC0-1.0 <https://creativecommons.org/publicdomain/zero/1.0/>`__
source        https://github.com/audeering/audbcards
usage         unrestricted
languages     eng, deu
format        wav
channel       1
sampling rate 8000
bit depth     16
duration      0 days 00:05:02
files         2
repository    `data-local <.../data-local/db>`__
published     2023-04-05 by author
============= ======================

Description
^^^^^^^^^^^

Example database.

Example
^^^^^^^

:file:`data/f0.wav`

.. image:: ../db.png

.. raw:: html

    <p><audio controls src="db/data/f0.wav"></audio></p>

Tables
^^^^^^

.. csv-table::
   :header: ID,Type,Columns
   :widths: 20, 10, 70

    "files", "filewise", "speaker"
    "segments", "segmented", "emotion"
    "speaker", "misc", "age, gender"

Schemes
^^^^^^^

.. csv-table::
    :header-rows: 1

    "ID", "Dtype", "Min", "Labels", "Mappings"
    "age", "int", "", ""
    "emotion", "str", "", "angry, happy, neutral"
    "gender", "str", "", "female, male"
    "speaker", "int", "", "0, 1", "age, gender"
