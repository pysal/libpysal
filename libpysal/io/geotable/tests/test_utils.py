import pytest


@pytest.mark.skip("skpping converters and metadata inserters")
class TestUtils:
    def test_converters(self):
        """Make a round trip to geodataframe and back."""

        raise Exception

    def test_insert_metadata(self):
        """Add an attribute to a dataframe and see if it is pervasive over copies."""

        raise Exception
