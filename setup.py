from setuptools import setup

package_data = {'audbcards': ['core/templates/*']}

setup(
    use_scm_version=True,
    package_data=package_data
)
