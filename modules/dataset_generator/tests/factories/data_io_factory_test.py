import unittest
from modules.dataset_generator.factories.data_io_factory import DataIOFactory
from modules.dataset_generator.helpers.data_readers import CsvIO, TxtIO, XmlIO

class TestDataIOFactory(unittest.TestCase):
    """
    Unit tests for the DataIOFactory class.
    """

    def test_create_csv(self) -> None:
        """
        Test that the factory creates a CsvIO instance when 'csv' type is provided.
        """
        data_io = DataIOFactory.create("csv")
        self.assertIsInstance(data_io, CsvIO)

    def test_create_txt(self) -> None:
        """
        Test that the factory creates a TxtIO instance when 'txt' type is provided.
        """
        data_io = DataIOFactory.create("txt")
        self.assertIsInstance(data_io, TxtIO)

    def test_create_xml(self) -> None:
        """
        Test that the factory creates an XmlIO instance when 'xml' type is provided.
        """
        data_io = DataIOFactory.create("xml")
        self.assertIsInstance(data_io, XmlIO)

    def test_create_unsupported_type(self) -> None:
        """
        Test that the factory raises a ValueError when an unsupported file type is provided.
        """
        with self.assertRaises(ValueError) as context:
            DataIOFactory.create("json")
        self.assertEqual(str(context.exception), "Unsupported file type: json")


if __name__ == "__main__":
    unittest.main()
