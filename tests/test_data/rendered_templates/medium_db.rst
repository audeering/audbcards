.. |medium_db-1.0.0-file-duration-distribution| image:: ./medium_db/medium_db-1.0.0-file-duration-distribution.png
.. |medium_db-1.0.0-segment-duration-distribution| image:: ./medium_db/medium_db-1.0.0-segment-duration-distribution.png

.. _datasets-medium_db:

medium_db
---------

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
files         2, duration distribution: 1.0 s |medium_db-1.0.0-file-duration-distribution| 301.0 s
segments      4, duration distribution: 0.5 s |medium_db-1.0.0-segment-duration-distribution| 151.0 s
repository    `data-local <.../data-local/medium_db>`__
published     2023-04-05 by author
============= ======================

Description
^^^^^^^^^^^

Medium database. \| Some description \|.

Example
^^^^^^^

:file:`data/f0.wav`

.. image:: ./medium_db/medium_db-1.0.0-player-waveform.png

.. raw:: html

    <p><audio controls src="./medium_db/data/f0.wav"></audio></p>

Tables
^^^^^^

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
        <td><p>files</p></td>
        <td><p>filewise</p></td>
        <td><p>speaker</p></td>
        <td class="expanded-row-content hide-row">

    
    <table class="docutils field-list align-default preview">
    <thead>
    <tr>
        <th class="head"><p>file</p></th>
        <th class="head"><p>speaker</p></th>
        </tr>
    </thead>
    <tbody>
                    <tr>
        <td><p>data/f0.wav</p></td>
        <td><p>0</p></td>
                    <tr>
        <td><p>data/f1.wav</p></td>
        <td><p>1</p></td>
                </tbody>
    </table>

    
    </td>
    </tr>
                <tr onClick="toggleRow(this)" class="row-odd clickable grid">
        <td><p>segments</p></td>
        <td><p>segmented</p></td>
        <td><p>emotion</p></td>
        <td class="expanded-row-content hide-row">

    
    <table class="docutils field-list align-default preview">
    <thead>
    <tr>
        <th class="head"><p>file</p></th>
        <th class="head"><p>start</p></th>
        <th class="head"><p>end</p></th>
        <th class="head"><p>emotion</p></th>
        </tr>
    </thead>
    <tbody>
                    <tr>
        <td><p>data/f0.wav</p></td>
        <td><p>0 days 00:00:00</p></td>
        <td><p>0 days 00:00:00.500000</p></td>
        <td><p>neutral</p></td>
                    <tr>
        <td><p>data/f0.wav</p></td>
        <td><p>0 days 00:00:00.500000</p></td>
        <td><p>0 days 00:00:01</p></td>
        <td><p>neutral</p></td>
                    <tr>
        <td><p>data/f1.wav</p></td>
        <td><p>0 days 00:00:00</p></td>
        <td><p>0 days 00:02:30</p></td>
        <td><p>happy</p></td>
                    <tr>
        <td><p>data/f1.wav</p></td>
        <td><p>0 days 00:02:30</p></td>
        <td><p>0 days 00:05:01</p></td>
        <td><p>angry</p></td>
                </tbody>
    </table>

    
    </td>
    </tr>
                <tr onClick="toggleRow(this)" class="row-even clickable grid">
        <td><p>speaker</p></td>
        <td><p>misc</p></td>
        <td><p>age, gender</p></td>
        <td class="expanded-row-content hide-row">

    
    <table class="docutils field-list align-default preview">
    <thead>
    <tr>
        <th class="head"><p>speaker</p></th>
        <th class="head"><p>age</p></th>
        <th class="head"><p>gender</p></th>
        </tr>
    </thead>
    <tbody>
                    <tr>
        <td><p>0</p></td>
        <td><p>23</p></td>
        <td><p>female</p></td>
                    <tr>
        <td><p>1</p></td>
        <td><p>49</p></td>
        <td><p>male</p></td>
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

    "ID", "Dtype", "Min", "Labels", "Mappings"
    "age", "int", "0", "", ""
    "emotion", "str", "", "angry, happy, neutral", ""
    "gender", "str", "", "female, male", ""
    "speaker", "int", "", "0, 1", "age, gender"
