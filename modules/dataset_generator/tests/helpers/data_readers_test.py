import unittest
import pandas as pd
import tempfile
import os
from modules.dataset_generator.helpers.data_readers import CsvIO, TxtIO, XmlIO


class TestDataIO(unittest.TestCase):
    """
    Unit tests for CsvIO, TxtIO, and XmlIO classes.
    """

    def setUp(self) -> None:
        # Create temporary CSV file
        self.csv_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", mode="w"
        )
        self.csv_file.write("column1,column2,column3\n1,2,3\n4,5,6\n")
        self.csv_file.close()

        # Create temporary TXT file
        self.txt_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w"
        )
        self.txt_file.write("column1\tcolumn2\tcolumn3\n1\t2\t3\n4\t5\t6\n")
        self.txt_file.close()

        # Create temporary XML file
        self.xml_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".xml", mode="w"
        )
        self.xml_file.write(
            """
        <data>
            <row>
                <column1>1</column1>
                <column2>2</column2>
                <column3>3</column3>
            </row>
            <row>
                <column1>4</column1>
                <column2>5</column2>
                <column3>6</column3>
            </row>
        </data>
        """
        )
        self.xml_file.close()

    def tearDown(self) -> None:
        # Remove temporary files
        os.unlink(self.csv_file.name)
        os.unlink(self.txt_file.name)
        os.unlink(self.xml_file.name)

    def test_csv_io(self) -> None:
        """
        Test CsvIO reads the correct data from a CSV file.
        """
        csv_io = CsvIO()
        df = csv_io.read_df_from_path(
            self.csv_file.name, columns={"column1": {}, "column2": {}}
        )
        expected_df = pd.DataFrame({"column1": [1, 4], "column2": [2, 5]})
        pd.testing.assert_frame_equal(df, expected_df)

    def test_txt_io(self) -> None:
        """
        Test TxtIO reads the correct data from a TXT file.
        """
        txt_io = TxtIO()
        df = txt_io.read_df_from_path(
            self.txt_file.name, columns={"column1": {}, "column2": {}}
        )
        expected_df = pd.DataFrame({"column1": [1, 4], "column2": [2, 5]})
        pd.testing.assert_frame_equal(df, expected_df)

    def test_xml_io(self) -> None:
        """
        Test XmlIO reads the correct data from an XML file.
        """
        xml_io = XmlIO()
        df = xml_io.read_df_from_path(
            self.xml_file.name, columns={"column1": {}, "column2": {}}
        )
        expected_df = pd.DataFrame({"column1": [1, 4], "column2": [2, 5]})
        pd.testing.assert_frame_equal(df, expected_df)

    def test_csv_io_exception(self) -> None:
        """
        Test CsvIO raises a ValueError when a non-existing column is requested.
        """
        csv_io = CsvIO()
        with self.assertRaises(ValueError):
            csv_io.read_df_from_path(
                self.csv_file.name, columns={"non_existent_column": {}}
            )

    def test_txt_io_exception(self) -> None:
        """
        Test TxtIO raises a ValueError when a non-existing column is requested.
        """
        txt_io = TxtIO()
        with self.assertRaises(ValueError):
            txt_io.read_df_from_path(
                self.txt_file.name, columns={"non_existent_column": {}}
            )

    def test_xml_io_exception(self) -> None:
        """
        Test XmlIO raises an IOError when a non-existing column is requested.
        """
        xml_io = XmlIO()
        with self.assertRaises(ValueError):
            xml_io.read_df_from_path(
                self.xml_file.name, columns={"non_existent_column": {}}
            )


if __name__ == "__main__":
    unittest.main()
