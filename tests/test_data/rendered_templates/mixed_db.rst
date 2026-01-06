.. _datasets-mixed_db:

mixed_db
--------

Created by H Wierstorf, C Geng, B E Abrougui

============= ======================
version       1.0.0

source        https://github.com/audeering/audbcards
usage         unrestricted
languages     
format        json, wav
channel       1
sampling rate 8000
bit depth     16
duration      0 days 00:00:00.100000
files         3, duration distribution: each file is 0.1 s


============= ======================

Description
^^^^^^^^^^^

Mixed database.

Example
^^^^^^^

:file:`c0.json`

.. code:: json

  [
    {
      "role": "human",
      "audio": "f0.wav",
      "transcription": "Hello World"
    },
    {
      "role": "assistant",
      "audio": "f1.wav"
    }
  ]


Tables
^^^^^^

Click on a row to toggle a preview.

.. raw:: html

    <table class="clickable docutils align-default">
        <thead>
    <tr class="row-odd grid header">
        <th class="head"><p>ID</p></th>
        <th class="head"><p>Type</p></th>
        <th class="head"><p>Columns</p></th>
        </tr>
    </thead>
        <tbody>
                    <tr onClick="toggleRow(this)" class="row-even clickable grid">
        <td><p>audio</p></td>
        <td><p>filewise</p></td>
        <td><p>transcription</p></td>
        <td class="expanded-row-content hide-row">

    
    <table class="docutils field-list align-default preview">
    <thead>
    <tr>
        <th class="head"><p>file</p></th>
        <th class="head"><p>transcription</p></th>
        </tr>
    </thead>
    <tbody>
                    <tr>
        <td><p>f0.wav</p></td>
        <td><p>Hello World</p></td>
        </tr>
                <tr>
        <td><p>f1.wav</p></td>
        <td><p></p></td>
        </tr>
            <tr><td><p class="table-statistic">2 rows x 1 column</p></td></tr>
    </tbody>
    </table>

    
    </td>
    </tr>
                <tr onClick="toggleRow(this)" class="row-odd clickable grid">
        <td><p>json</p></td>
        <td><p>filewise</p></td>
        <td><p>turns</p></td>
        <td class="expanded-row-content hide-row">

    
    <table class="docutils field-list align-default preview">
    <thead>
    <tr>
        <th class="head"><p>file</p></th>
        <th class="head"><p>turns</p></th>
        </tr>
    </thead>
    <tbody>
                    <tr>
        <td><p>c0.json</p></td>
        <td><p>2</p></td>
        </tr>
            <tr><td><p class="table-statistic">1 row x 1 column</p></td></tr>
    </tbody>
    </table>

    
    </td>
    </tr>
            </tbody>
    </table>


Schemes
^^^^^^^

.. csv-table::
    :header-rows: 1

    "ID", "Dtype"
    "transcription", "str"
    "turns", "int"
