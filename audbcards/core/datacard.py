import functools
import os

import jinja2

from audbcards.core.dataset import Dataset


class Datacard(object):

    def __init__(self, dataset: Dataset):

        self._dataset = dataset

    @functools.cached_property
    def content(self):
        """Property Accessor for rendered jinja2 content."""

        return self._render_template()

    def _render_template(self):

        t_dir = os.path.join(os.path.dirname(__file__), 'templates')
        environment = jinja2.Environment(loader=jinja2.FileSystemLoader(t_dir),
                                         trim_blocks=True)
        # Provide Jinja filter access to Python build-ins/functions
        environment.filters.update(
            zip=zip,
            tw=self._trim_trailing_whitespace,
        )
        template = environment.get_template("datacard.j2")
        content = template.render(self._dataset.properties())
        return content

    @staticmethod
    def _trim_trailing_whitespace(x: list):
        """J2 filter to get rid of trailing empty table entries within a row.

        Trims last entry if present.

        Args:
            x: untrimmed single scheme table row
        Returns:
            trimmed single scheme table row
        """

        if x[-1] == '':
            x.pop()

        return x

    def save(self, ofpath: str = None):
        """Save content of rendered template to rst.

        Args:
            ofpath: filepath to save rendered template to
        Returns:
            None

        if ofpath is specified, the directory must exist.
        """

        if ofpath is None:
            ofpath = f'datasets/{self._dataset.name}.rst'

        with open(ofpath, mode="w", encoding="utf-8") as fp:
            fp.write(self.content)
            print(f"... wrote {ofpath}")
