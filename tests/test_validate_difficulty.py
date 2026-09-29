import unittest
from xml.etree import ElementTree as ET

from notaear.validate_difficulty import _first_part_xml


class DifficultyValidationTests(unittest.TestCase):
    def test_concone_first_part_extraction_removes_accompaniment(self):
        xml = b"""<score-partwise>
          <part-list>
            <score-part id='P1'><part-name>Voice</part-name></score-part>
            <score-part id='P2'><part-name>Piano</part-name></score-part>
          </part-list>
          <part id='P1'><measure number='1'/></part>
          <part id='P2'><measure number='1'/></part>
        </score-partwise>"""
        root = ET.fromstring(_first_part_xml(xml))
        parts = root.findall("part")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].get("id"), "P1")


if __name__ == "__main__":
    unittest.main()
