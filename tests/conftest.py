import os

import numpy as np
import pandas as pd
import pytest

import audb
import audeer
import audformat
import audiofile


pytest.NAME = 'db'
pytest.VERSION = '1.0.0'


@pytest.fixture
def db(tmpdir, scope='function'):
    r"""Publish a test database.

    The database will use ``pytest.NAME`` as name
    and ``pytest.VERSION`` as version.
    It will be published to a file-system repository
    in a tmp folder.
    ``audb.config.REPOSITORIES`` will be adjusted
    to point to the repository
    the database was published to.

    """
    db_path = audeer.mkdir(audeer.path(tmpdir, pytest.NAME))

    db = audformat.Database(
        name=pytest.NAME,
        source='https://github.com/audeering/audbcards',
        usage='unrestricted',
        expires=None,
        languages='eng',
        description='Example database.',
        author='H Wierstorf, C Geng, B E Abrougui',
        organization='audEERING',
        license=audformat.define.License.CC0_1_0,
    )

    # Misc table 'speaker'
    db.schemes['age'] = audformat.Scheme(
        'int',
        minimum=0,
        description='Age of speaker',
    )
    db.schemes['gender'] = audformat.Scheme(
        'str',
        labels=['female', 'male'],
        description='Gender of speaker',
    )
    index = pd.Index(
        [0, 1],
        dtype='Int64',
        name='speaker',
    )
    db['speaker'] = audformat.MiscTable(index)
    db['speaker']['age'] = audformat.Column(scheme_id='age')
    db['speaker']['age'].set([23, 49])
    db['speaker']['gender'] = audformat.Column(scheme_id='gender')
    db['speaker']['gender'].set(['female', 'male'])

    # Table 'files'
    db.schemes['speaker'] = audformat.Scheme(
        'int',
        labels='speaker',
        description='Speaker IDs.',
    )
    index = audformat.filewise_index(['data/f0.wav', 'data/f1.wav'])
    db['files'] = audformat.Table(index)
    db['files']['speaker'] = audformat.Column(scheme_id='speaker')
    db['files']['speaker'].set([0, 1])

    # Table 'segments'
    db.schemes['emotion'] = audformat.Scheme(
        'str',
        labels=['angry', 'happy', 'neutral'],
        description='Emotional class.',
    )
    index = audformat.segmented_index(
        files=['data/f0.wav', 'data/f0.wav', 'data/f1.wav', 'data/f1.wav'],
        starts=[0, 1, 0, 1],
        ends=[1, 2, 1, 2],
    )
    db['segments'] = audformat.Table(index)
    db['segments']['emotion'] = audformat.Column(scheme_id='emotion')
    db['segments']['emotion'].set(['neutral', 'neutral', 'happy', 'angry'])

    # Create audio files and store database
    np.random.seed(1)
    sampling_rate = 8000
    duration = 2
    for file in list(db['files'].index):
        path = audeer.path(db_path, file)
        audeer.mkdir(os.path.dirname(path))
        signal = np.random.normal(0, .1, (1, duration * sampling_rate))
        audiofile.write(path, signal, sampling_rate, normalize=True)
    db.save(db_path)

    # Publish database
    host = audeer.mkdir(audeer.path(tmpdir, 'repo'))
    repository = audb.Repository(
        name='data-local',
        host=host,
        backend='file-system',
    )
    audb.config.REPOSITORIES = [repository]
    audb.publish(db_path, pytest.VERSION, repository)

    return db