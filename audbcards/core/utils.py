import typing

import matplotlib.pyplot as plt

import audeer
import audformat


def format_schemes(
    schemes: typing.Dict[str, audformat.Scheme],
    excludes: typing.Sequence[str] = ["duration"],
    max_schemes: int = 15,
) -> str:
    """Convert schemes object into string.

    It lists the main annotation schemes
    of the datasets,
    and collects additional information
    on schemes named ``'emotion'`` and ``'speaker'``,
    e.g. ``'speaker: [age, gender, language]'``.

    """
    # Filter schemes
    filtered_schemes = []
    emotion = []
    speaker = []
    for scheme in schemes:
        if scheme in excludes:
            continue
        # schemes[scheme] entries are always dictionaries,
        # so we don't have to check for that
        if scheme == "emotion":
            try:
                labels = schemes[scheme]._labels_to_list()
                labels = audeer.flatten_list(labels)
                emotion = [{scheme: labels}]
                max_schemes -= 1
            except KeyError:
                emotion = [scheme]
        elif scheme == "speaker":
            try:
                labels = schemes[scheme]._labels_to_dict()
                labels = list(labels.values())
                # Store the dictionary keys for speaker
                # as those are gender, language, ...
                # Keys are the same for all entries,
                # using the first one is enough
                labels = list(labels[0].keys())
                speaker = [{scheme: labels}]
                max_schemes -= 1
            except KeyError:
                emotion = [scheme]
            except AttributeError:
                emotion = [scheme]
        else:
            filtered_schemes.append(scheme)
    # Force emotion and speaker to the beginning of the list
    filtered_schemes = emotion + speaker + filtered_schemes
    # Limit to maximum number of schemes and add '...' for longer once
    max_schemes = max(max_schemes, 2)
    filtered_schemes = limit_presented_samples(filtered_schemes, max_schemes)
    # Format the information for display
    info_str = ""
    for scheme in filtered_schemes:
        if isinstance(scheme, dict):
            key = list(scheme.keys())[0]
            info_str += f"{key}: ["
            for label in scheme[key]:
                info_str += f"{label}, "
            info_str = info_str[:-2] + "], "
        else:
            info_str += f"{scheme}, "
    info_str = info_str[:-2]

    return info_str


def limit_presented_samples(
    samples: typing.Sequence,
    limit: int,
    replacement_text: str = "...",
) -> typing.List:
    r"""Limit the printing of sequences.

    If the sequence contains too many samples,
    they will be cut out in the center.

    Args:
        samples: sequence of samples to list on screen
        limit: maximum number to present
        replacement_text: text shown instead of removed samples

    Returns:
        string listing the samples

    """
    if len(samples) >= limit:
        samples = samples[: limit // 2] + [replacement_text] + samples[-limit // 2 :]
    return samples


def set_plot_margins(
    *,
    left=0,
    bottom=0,
    right=1,
    top=1,
    wspace=0,
    hspace=0,
):
    r"""Set the margins in a plot.

    As default it will remove all margins.
    For details on arguments,
    see :func:`matplotlib.pyplot.subplots_adjust`.

    """
    plt.subplots_adjust(
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
